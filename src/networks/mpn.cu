// Full MPN: multi-plasticity network (Aitken & Mihalas 2023, eLife 12:e83035,
// Eq. 1) with puffer's gated highway. Per layer with input x (H), a plastic
// matrix M (H x H) per agent, W = [W_a; W_g] (2H x H, bias-free) and learned
// scalars rho, eta (lambda, c as in mpn_common.cu):
//   M~_{t-1} = (1 - d_t) M_{t-1}                 (reset at episode boundary)
//   a_t   = (W_a ⊙ (1 + M~_{t-1})) x_t,  g_t = W_g x_t
//   h_t   = tanh(a_t),  s_t = sigmoid(g_t)
//   out_t = s*h + (1-s)*x_t
//   M_t   = lambda*M~_{t-1} + c * h_t x_t^T      (Hebbian: post h, pre x)
// Only W_a is plastic; the gate is static. W x is one time-parallel GEMM and
// the sequential scan only adds (W_a ⊙ M~) x and updates M.
//
// Rows of M are independent given x: row i needs only x_t and h_{t,i}. One
// warp owns one (agent, row), keeps that row of M (and of W_a) in registers
// for the whole T loop, and needs no cross-warp sync. Lane l holds columns
// j = l + 32k for k < K = H/32.
//
// Training stores every post-reset M~_{t-1} (B, T, H*H) for the backward
// scan. Cross-row sums in the backward (dx and dW of the plastic term) use
// f32 atomics. This is the simple, exact version; see MPN.md for the plan to
// checkpoint/recompute and to replace the atomics.
// Included from algo.cu; selected with [policy] network = mpn.
#include "mpn_common.cu"

#define MPN_WARPS_PER_BLOCK 8
// Hidden sizes with a compiled scan: 32 * {1, 2, 4, 8, 16, 32}.
#define MPN_K_CASES(X) X(1) X(2) X(4) X(8) X(16) X(32)

static bool mpn_hidden_supported(int H) {
    if (H % 32 != 0) {
        return false;
    }
    int K = H / 32;
    return K >= 1 && K <= 32 && (K & (K - 1)) == 0;
}

__device__ __forceinline__ float mpn_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

// Scan over T for warp (b, i).
// combined (B*T, 2H) holds W x on entry; its a-half is overwritten with
//   a_t = W_a x_t + (W_a ⊙ M~) x_t, rounded to precision_t. h uses the rounded a
// so the Hebbian update and the output gate (mpn_post) see the same value.
// M is rounded to precision_t every step so training reproduces the state
// sequence rollout stores. state_in (B, H*H) is M_0 (row-major M[i][j]).
// m_cache (B, T, H*H) gets M~_{t-1} when non-null (training). state_out gets
// M_T when non-null (rollout; may alias state_in since each warp reads its row
// before writing it). terminals (B, T) may be null.
template <int K>
__global__ void mpn_scan_forward(precision_t* __restrict__ combined,
        precision_t* __restrict__ m_cache, const precision_t* __restrict__ x,
        const precision_t* state_in, precision_t* state_out,
        const precision_t* __restrict__ terminals, const precision_t* __restrict__ W,
        const precision_t* __restrict__ scal, int B, int T, int H) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= B * H) {
        return;  // warp-uniform
    }
    int b = warp / H, i = warp % H;
    float lam, c;
    mpn_coeffs(scal, H, &lam, &c);

    float w[K], m[K], xv[K];
    long srow = ((long)b * H + i) * H;
    #pragma unroll
    for (int k = 0; k < K; k++) {
        int j = lane + 32 * k;
        w[k] = to_float(W[(long)i * H + j]);
        m[k] = to_float(state_in[srow + j]);
    }

    for (int t = 0; t < T; t++) {
        // terminal[t] and observation[t] are emitted together after the prior step.
        if (terminals != NULL && to_float(terminals[b * T + t]) != 0.0f) {
            #pragma unroll
            for (int k = 0; k < K; k++) {
                m[k] = 0.0f;
            }
        }
        long bt = (long)b * T + t;
        long xrow = bt * H;
        long ai = bt * 2 * H + i;
        float base = to_float(combined[ai]);
        float part = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            int j = lane + 32 * k;
            xv[k] = to_float(x[xrow + j]);
            part += w[k] * m[k] * xv[k];
        }
        if (m_cache != NULL) {
            long crow = (bt * H + i) * H;
            #pragma unroll
            for (int k = 0; k < K; k++) {
                m_cache[crow + lane + 32 * k] = from_float(m[k]);
            }
        }
        // The shuffle reduction syncs the warp, so every lane has read
        // combined[ai] before lane 0 overwrites it.
        float a = to_float(from_float(base + mpn_warp_sum(part)));
        float h = tanhf(a);
        if (lane == 0) {
            combined[ai] = from_float(a);
        }
        #pragma unroll
        for (int k = 0; k < K; k++) {
            m[k] = to_float(from_float(lam * m[k] + c * h * xv[k]));
        }
    }

    if (state_out != NULL) {
        #pragma unroll
        for (int k = 0; k < K; k++) {
            state_out[srow + lane + 32 * k] = from_float(m[k]);
        }
    }
}

// Time-parallel part of the output-gate backward. grad_combined gets
// [s*dL/dout | dL/dg]: the a-half is dL/dh before the plastic path adds to it
// (mpn_scan_backward turns it into dL/da). grad_hw = (1-s)*dL/dout (highway).
__global__ void mpn_post_backward(precision_t* __restrict__ grad_combined,
        precision_t* __restrict__ grad_hw, const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ combined, const precision_t* __restrict__ x,
        int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H) {
        return;
    }
    int r = idx / H, j = idx % H;
    float h = tanhf(to_float(combined[r * 2 * H + j]));
    float s = sigmoid(to_float(combined[r * 2 * H + H + j]));
    float go = to_float(grad_out[idx]);
    float xv = to_float(x[idx]);
    grad_combined[r * 2 * H + j] = from_float(go * s);
    grad_combined[r * 2 * H + H + j] = from_float(go * (h - xv) * s * (1.0f - s));
    grad_hw[idx] = from_float(go * (1.0f - s));
}

// Reverse scan for warp (b, i). G = dL/dM_t (row i) is carried in registers;
// truncated BPTT: G = 0 after the last step, and nothing flows into M_0.
// Per step t (descending), with M~ = M~_{t-1} from m_cache:
//   dh   = grad_combined[a-half] + c * (G x_t)_i
//   da   = dh * (1 - h^2)                       -> grad_combined[a-half]
//   dx_mod[t, j] += c*G_ij*h_i + W_ij*M~_ij*da_i   (f32 atomics across rows)
//   dW_a[i, j]   += da_i * x_j * M~_ij              (summed in registers)
//   dlambda += sum_j G_ij*M~_ij,  dc += h_i * (G x_t)_i
//   G <- (1 - d_t) * (lambda*G + W_ij*da_i*x_j)
// The unmodulated terms (da x^T for dW, W^T da for dx) are left to the GEMMs.
template <int K>
__global__ void mpn_scan_backward(precision_t* __restrict__ grad_combined,
        float* __restrict__ dx_mod, float* __restrict__ dw_mod,
        float* __restrict__ scal_acc, const precision_t* __restrict__ combined,
        const precision_t* __restrict__ m_cache, const precision_t* __restrict__ x,
        const precision_t* __restrict__ terminals, const precision_t* __restrict__ W,
        const precision_t* __restrict__ scal, int B, int T, int H) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= B * H) {
        return;  // warp-uniform
    }
    int b = warp / H, i = warp % H;
    float lam, c;
    mpn_coeffs(scal, H, &lam, &c);

    float w[K], g[K], dw[K], xv[K], mp[K];
    #pragma unroll
    for (int k = 0; k < K; k++) {
        w[k] = to_float(W[(long)i * H + lane + 32 * k]);
        g[k] = 0.0f;
        dw[k] = 0.0f;
    }
    float d_lam = 0.0f, d_c = 0.0f;

    for (int t = T - 1; t >= 0; t--) {
        long bt = (long)b * T + t;
        long xrow = bt * H;
        long crow = (bt * H + i) * H;
        long ai = bt * 2 * H + i;
        float h = tanhf(to_float(combined[ai]));
        float dh_out = to_float(grad_combined[ai]);
        float gx_part = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            int j = lane + 32 * k;
            xv[k] = to_float(x[xrow + j]);
            mp[k] = to_float(m_cache[crow + j]);
            gx_part += g[k] * xv[k];
            d_lam += g[k] * mp[k];
        }
        // Shuffle syncs the warp: all lanes read grad_combined[ai] before lane 0 writes.
        float gx = mpn_warp_sum(gx_part);
        float da = (dh_out + c * gx) * (1.0f - h * h);
        if (lane == 0) {
            grad_combined[ai] = from_float(da);
            d_c += h * gx;
        }
        #pragma unroll
        for (int k = 0; k < K; k++) {
            int j = lane + 32 * k;
            atomicAdd(&dx_mod[xrow + j], c * g[k] * h + w[k] * mp[k] * da);
            dw[k] += da * xv[k] * mp[k];
            g[k] = lam * g[k] + w[k] * da * xv[k];
        }
        // M~_{t-1} was reset to 0 before step t: no flow into earlier state.
        if (terminals != NULL && to_float(terminals[b * T + t]) != 0.0f) {
            #pragma unroll
            for (int k = 0; k < K; k++) {
                g[k] = 0.0f;
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < K; k++) {
        atomicAdd(&dw_mod[(long)i * H + lane + 32 * k], dw[k]);
    }
    d_lam = mpn_warp_sum(d_lam);
    if (lane == 0) {
        atomicAdd(&scal_acc[0], d_lam);
        atomicAdd(&scal_acc[1], d_c);
    }
}

// wgrad (2H, H) <- [dW_a plastic term (f32) ; 0]; the GEMM then adds grad_combined^T x.
__global__ void mpn_wgrad_init(precision_t* __restrict__ wgrad,
        const float* __restrict__ dw_mod, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * H * H) {
        return;
    }
    wgrad[idx] = from_float(idx < H * H ? dw_mod[idx] : 0.0f);
}

// dx <- (GEMM result W^T [da; dg]) + highway + plastic terms.
__global__ void mpn_dx_finalize(precision_t* __restrict__ dx,
        const precision_t* __restrict__ grad_hw, const float* __restrict__ dx_mod, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    dx[idx] = from_float(to_float(dx[idx]) + to_float(grad_hw[idx]) + dx_mod[idx]);
}

static dim3 mpn_scan_grid(int B, int H) {
    long warps = (long)B * H;
    return dim3((unsigned)((warps + MPN_WARPS_PER_BLOCK - 1) / MPN_WARPS_PER_BLOCK));
}

static void mpn_launch_scan_forward(precision_t* combined, precision_t* m_cache,
        const precision_t* x, const precision_t* state_in, precision_t* state_out,
        const precision_t* terminals, const precision_t* W, const precision_t* scal,
        int B, int T, int H, cudaStream_t stream) {
    dim3 grid = mpn_scan_grid(B, H), block(32 * MPN_WARPS_PER_BLOCK);
    switch (H / 32) {
#define MPN_CASE(K) case K: mpn_scan_forward<K><<<grid, block, 0, stream>>>( \
        combined, m_cache, x, state_in, state_out, terminals, W, scal, B, T, H); break;
    MPN_K_CASES(MPN_CASE)
#undef MPN_CASE
    default:
        assert(0 && "mpn: unsupported hidden size");
    }
}

static void mpn_launch_scan_backward(precision_t* grad_combined, float* dx_mod,
        float* dw_mod, float* scal_acc, const precision_t* combined,
        const precision_t* m_cache, const precision_t* x, const precision_t* terminals,
        const precision_t* W, const precision_t* scal, int B, int T, int H,
        cudaStream_t stream) {
    dim3 grid = mpn_scan_grid(B, H), block(32 * MPN_WARPS_PER_BLOCK);
    switch (H / 32) {
#define MPN_CASE(K) case K: mpn_scan_backward<K><<<grid, block, 0, stream>>>( \
        grad_combined, dx_mod, dw_mod, scal_acc, combined, m_cache, x, terminals, \
        W, scal, B, T, H); break;
    MPN_K_CASES(MPN_CASE)
#undef MPN_CASE
    default:
        assert(0 && "mpn: unsupported hidden size");
    }
}

struct MPNActivations {
    // Rollout
    Prec* combined;       // (B rollout, 2H)[num_layers]
    Prec out;             // (B rollout, H)
    // Training
    Prec* saved_inputs;   // (B, TT, H)[num_layers]
    Prec* combined_bufs;  // (B*TT, 2H)[num_layers] W x; a-half becomes a_t in the scan
    Prec* m_cache;        // (B, TT, H*H)[num_layers] post-reset M~_{t-1}
    Prec* outs;           // (B, TT, H)[num_layers]
    Prec grad_combined;   // (B*TT, 2H) [dL/da | dL/dg]
    Prec grad_hw;         // (B, TT, H) highway dx
    Prec grad_input_buf;  // (B*TT, H) total dx
    Float dx_mod;         // (B, TT, H) f32 plastic-path dx
    Float dw_mod;         // (H, H) f32 plastic-path dW_a
    Float scal_acc;       // (num_layers, 2) f32 dlambda, dc accumulators
    Prec* wgrad_scratch;  // (2H, H)[num_layers]
    Prec* sgrad_scratch;  // (MPN_SCALARS)[num_layers]
    precision_t* terminals;  // (B, TT) from forward_train, read in backward
};

void mpn_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    MPNWeights* m = (MPNWeights*)w;
    MPNActivations* a = (MPNActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT, L = m->num_layers;
    a->saved_inputs = (Prec*)calloc(L, sizeof(Prec));
    a->combined_bufs = (Prec*)calloc(L, sizeof(Prec));
    a->m_cache = (Prec*)calloc(L, sizeof(Prec));
    a->outs = (Prec*)calloc(L, sizeof(Prec));
    a->wgrad_scratch = (Prec*)calloc(L, sizeof(Prec));
    a->sgrad_scratch = (Prec*)calloc(L, sizeof(Prec));
    a->grad_combined = {.shape = {B_TT, 2 * H}};
    a->grad_hw = {.shape = {B, TT, H}};
    a->grad_input_buf = {.shape = {B_TT, H}};
    a->dx_mod = {.shape = {B, TT, H}};
    a->dw_mod = {.shape = {H, H}};
    a->scal_acc = {.shape = {L, 2}};
    alloc_register(acts, &a->grad_combined);
    alloc_register(acts, &a->grad_hw);
    alloc_register(acts, &a->grad_input_buf);
    alloc_register(acts, &a->dx_mod);
    alloc_register(acts, &a->dw_mod);
    alloc_register(acts, &a->scal_acc);
    for (int i = 0; i < L; i++) {
        a->saved_inputs[i]  = {.shape = {B, TT, H}};
        a->combined_bufs[i] = {.shape = {B_TT, 2 * H}};
        a->m_cache[i]       = {.shape = {B, TT, (long)H * H}};
        a->outs[i]          = {.shape = {B, TT, H}};
        a->wgrad_scratch[i] = {.shape = {2 * H, H}};
        a->sgrad_scratch[i] = {.shape = {MPN_SCALARS}};
        alloc_register(acts, &a->saved_inputs[i]);
        alloc_register(acts, &a->combined_bufs[i]);
        alloc_register(acts, &a->m_cache[i]);
        alloc_register(acts, &a->outs[i]);
        // Grads only, in param order (W, scalars per layer): Muon assumes it.
        alloc_register(grads, &a->wgrad_scratch[i]);
        alloc_register(grads, &a->sgrad_scratch[i]);
    }
    double gb = (double)L * B_TT * H * H * sizeof(precision_t) / 1e9;
    fprintf(stderr, "mpn: plastic-state training cache %.2f GB (%d layers x %d x %dx%d)\n",
        gb, L, B_TT, H, H);
}

void mpn_reg_rollout(void* weights, void* activations,
        Allocator* alloc, int B_inf) {
    MPNWeights* w = (MPNWeights*)weights;
    MPNActivations* a = (MPNActivations*)activations;
    int H = w->hidden;
    a->combined = (Prec*)calloc(w->num_layers, sizeof(Prec));
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 2 * H}};
        alloc_register(alloc, &a->combined[i]);
    }
    a->out = {.shape = {B_inf, H}};
    alloc_register(alloc, &a->out);
}

// One rollout step. state (L, B, H*H) is updated in place; the trainer zeroes
// terminated agents' state (zero_term_state) before this runs.
Prec mpn_forward(void* w, Prec x, Prec state,
        void* activations, cudaStream_t stream) {
    MPNWeights* m = (MPNWeights*)w;
    MPNActivations* a = (MPNActivations*)activations;
    int B = state.shape[1];
    int H = m->hidden;
    for (int i = 0; i < m->num_layers; i++) {
        Prec state_i = mpn_state_layer(state, i, 0, B);
        puf_mm(&x, &m->weights[i], &a->combined[i], stream);
        mpn_launch_scan_forward(a->combined[i].data, NULL, x.data,
            state_i.data, state_i.data, NULL, m->weights[i].data,
            m->scalars[i].data, B, 1, H, stream);
        mpn_post<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->combined[i].data, x.data, B, H);
        x = a->out;
    }
    return x;
}

Prec mpn_forward_train(void* w, Prec x, Prec state, Prec terminals,
        void* activations, int agent_off, cudaStream_t stream) {
    MPNWeights* m = (MPNWeights*)w;
    MPNActivations* a = (MPNActivations*)activations;
    int B = (int)x.shape[0], TT = (int)x.shape[1], H = m->hidden;
    a->terminals = terminals.data;
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(&a->saved_inputs[i], &x, stream);
        Prec state_i = mpn_state_layer(state, i, agent_off, B);
        puf_mm(&a->saved_inputs[i], &m->weights[i], &a->combined_bufs[i], stream);
        mpn_launch_scan_forward(a->combined_bufs[i].data, a->m_cache[i].data,
            a->saved_inputs[i].data, state_i.data, NULL, a->terminals,
            m->weights[i].data, m->scalars[i].data, B, TT, H, stream);
        mpn_post<<<grid_size(B * TT * H), BLOCK_SIZE, 0, stream>>>(
            a->outs[i].data, a->combined_bufs[i].data, a->saved_inputs[i].data,
            B * TT, H);
        x = a->outs[i];
    }
    return x;
}

Prec mpn_backward(void* w, Prec grad, void* activations, cudaStream_t stream) {
    MPNWeights* m = (MPNWeights*)w;
    MPNActivations* a = (MPNActivations*)activations;
    int B = (int)a->grad_hw.shape[0], TT = (int)a->grad_hw.shape[1], H = m->hidden;
    int N = B * TT;
    cudaMemsetAsync(a->scal_acc.data, 0, numel(a->scal_acc.shape) * sizeof(float), stream);
    for (int i = m->num_layers - 1; i >= 0; i--) {
        mpn_post_backward<<<grid_size(N * H), BLOCK_SIZE, 0, stream>>>(
            a->grad_combined.data, a->grad_hw.data, grad.data,
            a->combined_bufs[i].data, a->saved_inputs[i].data, N, H);
        cudaMemsetAsync(a->dx_mod.data, 0, numel(a->dx_mod.shape) * sizeof(float), stream);
        cudaMemsetAsync(a->dw_mod.data, 0, numel(a->dw_mod.shape) * sizeof(float), stream);
        float* acc = a->scal_acc.data + 2 * i;
        mpn_launch_scan_backward(a->grad_combined.data, a->dx_mod.data, a->dw_mod.data,
            acc, a->combined_bufs[i].data, a->m_cache[i].data, a->saved_inputs[i].data,
            a->terminals, m->weights[i].data, m->scalars[i].data, B, TT, H, stream);
        // dW = [plastic dW_a ; 0] + grad_combined^T x (on main: it accumulates
        // onto the plastic term, so it cannot run before the scan finishes).
        mpn_wgrad_init<<<grid_size(2 * H * H), BLOCK_SIZE, 0, stream>>>(
            a->wgrad_scratch[i].data, a->dw_mod.data, H);
        puf_mm_tn(&a->grad_combined, &a->saved_inputs[i], &a->wgrad_scratch[i],
            stream, 1.0f, 1.0f);
        // grad is consumed by mpn_post_backward above, so grad_input_buf is free.
        puf_mm_nn(&a->grad_combined, &m->weights[i], &a->grad_input_buf, stream);
        mpn_dx_finalize<<<grid_size(N * H), BLOCK_SIZE, 0, stream>>>(
            a->grad_input_buf.data, a->grad_hw.data, a->dx_mod.data, N * H);
        mpn_scalar_grad<<<1, MPN_SCALARS, 0, stream>>>(
            a->sgrad_scratch[i].data, acc, m->scalars[i].data, H);
        grad = a->grad_input_buf;
    }
    return grad;
}
