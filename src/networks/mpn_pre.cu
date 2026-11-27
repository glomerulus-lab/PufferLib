// MPN: multi-plasticity network, MPNpre variant (Aitken & Mihalas 2023,
// eLife 12:e83035, Eq. 1 + Eq. 13), plus the highway connection used by puffer.
// Per layer with input x (H), state m (H) and learned scalars rho, eta:
//   lambda = MPN_LAMBDA_MAX * sigmoid(rho),  c = eta / sqrt(H)
//   u_t    = x_t * (1 + m_{t-1})        // == (W ⊙ (1+M)) x when all rows of M are equal
//   [a|g]  = u_t @ W^T                   // W: (2H, H), bias-free
//   out_t  = s*tanh(a) + (1-s)*x_t,  s = sigmoid(g)
//   m_t    = lambda*m_{t-1} + c*x_t
// m depends only on inputs (no h feedback), so training is a linear scan over T.
// Included from algo.cu; selected with [policy] network = mpn_pre.
#include "mpn_common.cu"

// Rollout pre-step: u = x*(1+m), then m <- lambda*m + c*x in place.
__global__ void mpnpre_step_pre(precision_t* __restrict__ u,
        precision_t* __restrict__ state, const precision_t* __restrict__ x,
        const precision_t* __restrict__ scal, int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    float lam, c;
    mpn_coeffs(scal, H, &lam, &c);
    float m = to_float(state[idx]);
    float xv = to_float(x[idx]);
    u[idx] = from_float(xv * (1.0f + m));
    state[idx] = from_float(lam * m + c * xv);
}

// Train scan over T, one thread per (b, j). Caches post-reset m_{t-1} for bwd
// and rounds m to precision_t each step to match rollout's stored state.
__global__ void mpnpre_scan_forward(precision_t* __restrict__ u,
        precision_t* __restrict__ m_prev, const precision_t* __restrict__ x,
        const precision_t* __restrict__ state, const precision_t* __restrict__ terminals,
        const precision_t* __restrict__ scal, int B, int T, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    int b = idx / H, j = idx % H;
    float lam, c;
    mpn_coeffs(scal, H, &lam, &c);
    float m = to_float(state[idx]);
    long base = (long)b * T * H + j;
    for (int t = 0; t < T; t++) {
        // terminal[t] and observation[t] are emitted together after the prior step.
        if (terminals != NULL && to_float(terminals[b * T + t]) != 0.0f) {
            m = 0.0f;
        }
        long i = base + (long)t * H;
        float xv = to_float(x[i]);
        m_prev[i] = from_float(m);
        u[i] = from_float(xv * (1.0f + m));
        m = to_float(from_float(lam * m + c * xv));
    }
}

// Backward of mpnpre_post: grad_combined = [dL/da | dL/dg], grad_hw = dL/dx (highway).
__global__ void mpnpre_post_backward(precision_t* __restrict__ grad_combined,
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
    grad_combined[r * 2 * H + j] = from_float(go * s * (1.0f - h * h));
    grad_combined[r * 2 * H + H + j] = from_float(go * (h - xv) * s * (1.0f - s));
    grad_hw[idx] = from_float(go * (1.0f - s));
}

// Reverse linear scan. g = dL/dm_t carried backward (truncated: 0 at t=T).
// grad_u holds dL/du on entry; overwritten with total dL/dx:
//   dx = dx_highway + du*(1+m_prev) + c*g;  g <- lambda*g + du*x
// Also accumulates dL/dlambda, dL/dc into scal_acc[0..1] (f32 atomics).
__global__ void mpnpre_scan_backward(precision_t* __restrict__ grad_u,
        const precision_t* __restrict__ grad_hw, const precision_t* __restrict__ x,
        const precision_t* __restrict__ m_prev, const precision_t* __restrict__ terminals,
        const precision_t* __restrict__ scal, float* __restrict__ scal_acc,
        int B, int T, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float d_lam = 0.0f, d_c = 0.0f;
    if (idx < B * H) {
        int b = idx / H, j = idx % H;
        float lam, c;
        mpn_coeffs(scal, H, &lam, &c);
        long base = (long)b * T * H + j;
        float g = 0.0f;
        for (int t = T - 1; t >= 0; t--) {
            long i = base + (long)t * H;
            float du = to_float(grad_u[i]);
            float xv = to_float(x[i]);
            float mp = to_float(m_prev[i]);
            grad_u[i] = from_float(to_float(grad_hw[i]) + du * (1.0f + mp) + c * g);
            d_lam += g * mp;
            d_c += g * xv;
            g = lam * g + du * xv;
            // m_{t-1} was reset to 0 before step t: no flow into earlier state.
            if (terminals != NULL && to_float(terminals[b * T + t]) != 0.0f) {
                g = 0.0f;
            }
        }
    }
    // All lanes reach here (no early return) so full-warp shuffles are safe.
    for (int off = 16; off > 0; off >>= 1) {
        d_lam += __shfl_down_sync(0xffffffff, d_lam, off);
        d_c += __shfl_down_sync(0xffffffff, d_c, off);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&scal_acc[0], d_lam);
        atomicAdd(&scal_acc[1], d_c);
    }
}

struct MPNPreActivations {
    // Rollout
    Prec* combined;       // (B rollout, 2H)[num_layers]
    Prec u;               // (B rollout, H)
    Prec out;             // (B rollout, H)
    // Training
    Prec* saved_inputs;   // (B, TT, H)[num_layers]
    Prec* u_bufs;         // (B, TT, H)[num_layers] GEMM input, reused for dW
    Prec* m_prev;         // (B, TT, H)[num_layers] post-reset m_{t-1}
    Prec* combined_bufs;  // (B*TT, 2H)[num_layers]
    Prec* outs;           // (B, TT, H)[num_layers]
    Prec* grad_combined;  // (B*TT, 2H)[num_layers] per layer: side-stream dW reads it
    Prec grad_hw;         // (B, TT, H) highway dx
    Prec grad_input_buf;  // (B*TT, H) du, then total dx
    Float scal_acc;       // (num_layers, 2) f32 dlambda, dc accumulators
    Prec* wgrad_scratch;  // (2H, H)[num_layers]
    Prec* sgrad_scratch;  // (MPN_SCALARS)[num_layers]
    precision_t* terminals;  // (B, TT) from forward_train, read in backward
};

void mpnpre_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    MPNWeights* m = (MPNWeights*)w;
    MPNPreActivations* a = (MPNPreActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT, L = m->num_layers;
    a->saved_inputs = (Prec*)calloc(L, sizeof(Prec));
    a->u_bufs = (Prec*)calloc(L, sizeof(Prec));
    a->m_prev = (Prec*)calloc(L, sizeof(Prec));
    a->combined_bufs = (Prec*)calloc(L, sizeof(Prec));
    a->outs = (Prec*)calloc(L, sizeof(Prec));
    a->grad_combined = (Prec*)calloc(L, sizeof(Prec));
    a->wgrad_scratch = (Prec*)calloc(L, sizeof(Prec));
    a->sgrad_scratch = (Prec*)calloc(L, sizeof(Prec));
    a->grad_hw = {.shape = {B, TT, H}};
    a->grad_input_buf = {.shape = {B_TT, H}};
    a->scal_acc = {.shape = {L, 2}};
    alloc_register(acts, &a->grad_hw);
    alloc_register(acts, &a->grad_input_buf);
    alloc_register(acts, &a->scal_acc);
    for (int i = 0; i < L; i++) {
        a->saved_inputs[i]  = {.shape = {B, TT, H}};
        a->u_bufs[i]        = {.shape = {B, TT, H}};
        a->m_prev[i]        = {.shape = {B, TT, H}};
        a->combined_bufs[i] = {.shape = {B_TT, 2 * H}};
        a->outs[i]          = {.shape = {B, TT, H}};
        a->grad_combined[i] = {.shape = {B_TT, 2 * H}};
        a->wgrad_scratch[i] = {.shape = {2 * H, H}};
        a->sgrad_scratch[i] = {.shape = {MPN_SCALARS}};
        alloc_register(acts, &a->saved_inputs[i]);
        alloc_register(acts, &a->u_bufs[i]);
        alloc_register(acts, &a->m_prev[i]);
        alloc_register(acts, &a->combined_bufs[i]);
        alloc_register(acts, &a->outs[i]);
        alloc_register(acts, &a->grad_combined[i]);
        alloc_register(grads, &a->wgrad_scratch[i]);
        alloc_register(grads, &a->sgrad_scratch[i]);
    }
}

void mpnpre_reg_rollout(void* weights, void* activations,
        Allocator* alloc, int B_inf) {
    MPNWeights* w = (MPNWeights*)weights;
    MPNPreActivations* a = (MPNPreActivations*)activations;
    int H = w->hidden;
    a->combined = (Prec*)calloc(w->num_layers, sizeof(Prec));
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 2 * H}};
        alloc_register(alloc, &a->combined[i]);
    }
    a->u = {.shape = {B_inf, H}};
    a->out = {.shape = {B_inf, H}};
    alloc_register(alloc, &a->u);
    alloc_register(alloc, &a->out);
}

Prec mpnpre_forward(void* w, Prec x, Prec state,
        void* activations, cudaStream_t stream) {
    MPNWeights* m = (MPNWeights*)w;
    MPNPreActivations* a = (MPNPreActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
    for (int i = 0; i < m->num_layers; i++) {
        Prec state_i = mpn_state_layer(state, i, 0, B);
        Prec u = {.data = a->u.data, .shape = {B, H}};
        mpnpre_step_pre<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
            u.data, state_i.data, x.data, m->scalars[i].data, B * H, H);
        puf_mm(&u, &m->weights[i], &a->combined[i], stream);
        mpn_post<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->combined[i].data, x.data, B, H);
        x = a->out;
    }
    return x;
}

Prec mpnpre_forward_train(void* w, Prec x, Prec state, Prec terminals,
        void* activations, int agent_off, cudaStream_t stream) {
    MPNWeights* m = (MPNWeights*)w;
    MPNPreActivations* a = (MPNPreActivations*)activations;
    int B = (int)x.shape[0], TT = (int)x.shape[1], H = m->hidden;
    a->terminals = terminals.data;
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(&a->saved_inputs[i], &x, stream);
        Prec state_i = mpn_state_layer(state, i, agent_off, B);
        mpnpre_scan_forward<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
            a->u_bufs[i].data, a->m_prev[i].data, a->saved_inputs[i].data,
            state_i.data, a->terminals, m->scalars[i].data, B, TT, H);
        puf_mm(&a->u_bufs[i], &m->weights[i], &a->combined_bufs[i], stream);
        mpn_post<<<grid_size(B * TT * H), BLOCK_SIZE, 0, stream>>>(
            a->outs[i].data, a->combined_bufs[i].data, a->saved_inputs[i].data,
            B * TT, H);
        x = a->outs[i];
    }
    return x;
}

Prec mpnpre_backward(void* w, Prec grad, void* activations, cudaStream_t stream) {
    MPNWeights* m = (MPNWeights*)w;
    MPNPreActivations* a = (MPNPreActivations*)activations;
    int B = (int)a->grad_hw.shape[0], TT = (int)a->grad_hw.shape[1], H = m->hidden;
    int N = B * TT;
    cudaMemsetAsync(a->scal_acc.data, 0, numel(a->scal_acc.shape) * sizeof(float), stream);
    for (int i = m->num_layers - 1; i >= 0; i--) {
        mpnpre_post_backward<<<grid_size(N * H), BLOCK_SIZE, 0, stream>>>(
            a->grad_combined[i].data, a->grad_hw.data, grad.data,
            a->combined_bufs[i].data, a->saved_inputs[i].data, N, H);
        // dW on side stream (per-layer scratch); dU on main continues the bwd chain.
        puf_mm_tn_async_after(&a->grad_combined[i], &a->u_bufs[i],
            &a->wgrad_scratch[i], stream);
        puf_mm_nn(&a->grad_combined[i], &m->weights[i], &a->grad_input_buf, stream);
        float* acc = a->scal_acc.data + 2 * i;
        mpnpre_scan_backward<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
            a->grad_input_buf.data, a->grad_hw.data, a->saved_inputs[i].data,
            a->m_prev[i].data, a->terminals, m->scalars[i].data, acc, B, TT, H);
        mpn_scalar_grad<<<1, MPN_SCALARS, 0, stream>>>(
            a->sgrad_scratch[i].data, acc, m->scalars[i].data, H);
        grad = a->grad_input_buf;
    }
    return grad;
}

