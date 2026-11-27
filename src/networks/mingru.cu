// MinGRU recurrent network (the 5.0 default), restored verbatim from 5.0 src/algo.cu.
// Included from algo.cu; selected with [policy] network = mingru.
// Per layer: [hidden|gate|proj] = x @ W^T (W: 3H x H),
//   h = lerp(h, h_tilde(hidden), sigmoid(gate)),  out = s*h + (1-s)*x, s = sigmoid(proj).

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}

// Rollout MinGRU step: h = (1-z)*h + z*h_tilde, highway out = s*h + (1-s)*x.
__global__ void mingru_gate(precision_t* out, precision_t* next_state,
        const precision_t* combined, const precision_t* state_in,
        const precision_t* x_in, int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;
    if (idx >= N) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int combined_base = b * 3 * H;
    float hidden = to_float(combined[combined_base + h]);
    float gate = to_float(combined[combined_base + H + h]);
    float proj = to_float(combined[combined_base + 2*H + h]);
    float state = to_float(state_in[idx]);
    float x = to_float(x_in[idx]);

    float z = sigmoid(gate);
    float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
    float h_out = lerp(state, h_tilde, z);
    next_state[idx] = from_float(h_out);

    float s = sigmoid(proj);
    out[idx] = from_float(s * h_out + (1.0f - s) * x);
}

// Train scan buffers. Cache post-reset h inputs; recompute outputs in bwd.
struct PrefixScan {
    precision_t* combined_ptr = NULL;
    precision_t* state_ptr = NULL;
    precision_t* input_ptr = NULL;  // (B, T, H) pre-projection input (highway)
    precision_t* terminals_ptr = NULL;  // (B, T), reset before step t if terminal[t]
    int B = 0, T = 0, H = 0;
    Prec scan_h;  // (B, T, H), recurrent input to each step
    Prec out, next_state;
    Prec grad_combined, grad_state;
    Prec grad_input;
};

// Train scan: match rollout's precision_t recurrent state between steps.
__global__ void mingru_scan_forward(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ out = scan.out.data;
    precision_t* __restrict__ next_state = scan.next_state.data;
    precision_t* __restrict__ scan_h = scan.scan_h.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const precision_t* __restrict__ terminals = scan.terminals_ptr;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bH = b * H;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int bHT = bH * T_seq;
    const int out_base = bHT + h;
    int cbase = 3 * bHT;

    // h_0
    float h_t = to_float(state[bH + h]);
    int h_base = b * T_seq * H + h;

    const precision_t* __restrict__ combined_h_base = &combined[cbase + h];
    const precision_t* __restrict__ combined_g_base = &combined[cbase + H + h];
    const precision_t* __restrict__ combined_p_base = &combined[cbase + H2 + h];

    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 0; t < T_seq; t++) {
        // terminal[t] and observation[t] are emitted together after the prior step.
        if (terminals != NULL && to_float(terminals[b * T_seq + t]) != 0.0f) {
            h_t = 0.0f;
        }
        scan_h[h_base + t * H] = from_float(h_t);

        float hidden_val = to_float(__ldg(&combined_h_base[t_offset]));
        float gate_val = to_float(__ldg(&combined_g_base[t_offset]));
        float proj_val = to_float(__ldg(&combined_p_base[t_offset]));
        float x_val = to_float(__ldg(&input[out_base + t * H]));

        // h = (1-z)*h + z*h_tilde  (exact sigmoid; matches prior train log-space)
        float z = sigmoid(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid(hidden_val);
        h_t = lerp(h_t, h_tilde, z);

        float proj_sigmoid = sigmoid(proj_val);
        out[out_curr] = from_float(proj_sigmoid * h_t + (1.0f - proj_sigmoid) * x_val);
        h_t = to_float(from_float(h_t));

        out_curr += H;
        t_offset += H3;
    }

    next_state[bH + h] = from_float(h_t);
}

// Linear reverse scan. Cache is bf16 h only; a,z,h_tilde recomputed from gates in f32.
__global__ void mingru_scan_backward(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ grad_combined = scan.grad_combined.data;
    precision_t* __restrict__ grad_state = scan.grad_state.data;
    precision_t* __restrict__ grad_input = scan.grad_input.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const precision_t* __restrict__ terminals = scan.terminals_ptr;
    const precision_t* __restrict__ scan_h = scan.scan_h.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int H2 = 2 * H;
    const int state_idx = b * H + h;
    const int out_base = bHT + h;
    const int h_base = b * T_seq * H + h;

    const precision_t* __restrict__ combined_h_base = &combined[cbase + h];
    const precision_t* __restrict__ combined_g_base = &combined[cbase + H + h];
    const precision_t* __restrict__ combined_p_base = &combined[cbase + H2 + h];

    precision_t* __restrict__ grad_combined_h_base = &grad_combined[cbase + h];
    precision_t* __restrict__ grad_combined_g_base = &grad_combined[cbase + H + h];
    precision_t* __restrict__ grad_combined_p_base = &grad_combined[cbase + H2 + h];

    // dh flowing into h_t from the future (and grad_next at t=T).
    float dh = to_float(grad_next_state[state_idx]);

    for (int t = T_seq; t >= 1; --t) {
        int t0 = t - 1;  // 0-based step
        int t_offset = t0 * H3;
        int input_idx = out_base + t0 * H;

        float h_prev = to_float(__ldg(&scan_h[h_base + (t - 1) * H]));

        float hidden_val = to_float(__ldg(&combined_h_base[t_offset]));
        float gate_val = to_float(__ldg(&combined_g_base[t_offset]));
        float proj_val = to_float(__ldg(&combined_p_base[t_offset]));
        float x_val = to_float(__ldg(&input[input_idx]));
        float grad_out_val = to_float(__ldg(&grad_out[input_idx]));

        float z = sigmoid(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid(hidden_val);
        float h_t = lerp(h_prev, h_tilde, z);
        float proj_sigmoid = sigmoid(proj_val);

        // highway: out = s*h + (1-s)*x
        float dh_from_out = grad_out_val * proj_sigmoid;
        float grad_proj = grad_out_val * (h_t - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
        grad_input[input_idx] = from_float(grad_out_val * (1.0f - proj_sigmoid));
        grad_combined_p_base[t_offset] = from_float(grad_proj);

        float dh_total = dh + dh_from_out;

        // h = (1-z)*h_prev + z*h_tilde
        float d_h_prev = dh_total * (1.0f - z);
        float d_h_tilde = dh_total * z;
        float d_z = dh_total * (h_tilde - h_prev);

        // z = sigmoid(gate)
        float d_gate = d_z * z * (1.0f - z);
        // h_tilde = hidden+0.5  or  sigmoid(hidden)
        float d_hidden = (hidden_val >= 0.0f)
            ? d_h_tilde
            : d_h_tilde * h_tilde * (1.0f - h_tilde);

        grad_combined_h_base[t_offset] = from_float(d_hidden);
        grad_combined_g_base[t_offset] = from_float(d_gate);

        // terminal before this step zeroed h_prev contribution into this step;
        // after bwd through the step, cut gradient flow into pre-reset state.
        dh = d_h_prev;
        if (terminals != NULL &&
                to_float(terminals[b * T_seq + t0]) != 0.0f) {
            dh = 0.0f;
        }
    }

    grad_state[state_idx] = from_float(dh);
}

// Coalesced T-tile load + sequential apply. 32 h-threads apply after the load.
// Use when T is long and seq occupancy B*H is not already huge.
#define MINGRU_TILE 32

__global__ void mingru_scan_forward_row(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H;
    int hx = threadIdx.x, t_local = threadIdx.y;
    int h = blockIdx.y * MINGRU_TILE + hx;
    int h_ok = h < H;
    int b = blockIdx.x, H3 = 3 * H, bHT = b * H * T_seq, cbase = 3 * bHT;
    float h_t = h_ok ? to_float(scan.state_ptr[b * H + h]) : 0.0f;
    __shared__ float sm_z[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_ht[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_s[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_x[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_term[MINGRU_TILE][MINGRU_TILE];
    for (int c = 0; c < (T_seq + MINGRU_TILE - 1) / MINGRU_TILE; c++) {
        int t = c * MINGRU_TILE + t_local;
        float z = 0.0f, h_tilde = 0.0f, s = 0.0f, x = 0.0f, term = 0.0f;
        if (t < T_seq && h_ok) {
            int off = t * H3 + h;
            float hidden = to_float(scan.combined_ptr[cbase + off]);
            float gate = to_float(scan.combined_ptr[cbase + H + off]);
            float proj = to_float(scan.combined_ptr[cbase + 2 * H + off]);
            x = to_float(scan.input_ptr[bHT + t * H + h]);
            z = sigmoid(gate);
            h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
            s = sigmoid(proj);
            term = scan.terminals_ptr != NULL &&
                to_float(scan.terminals_ptr[b * T_seq + t]) != 0.0f;
        }
        sm_z[t_local][hx] = z;
        sm_ht[t_local][hx] = h_tilde;
        sm_s[t_local][hx] = s;
        sm_x[t_local][hx] = x;
        sm_term[t_local][hx] = term;
        __syncthreads();
        if (t_local == 0 && h_ok) {
            int nloc = min(MINGRU_TILE, T_seq - c * MINGRU_TILE);
            for (int i = 0; i < nloc; i++) {
                if (sm_term[i][hx] != 0.0f) {
                    h_t = 0.0f;
                }
                int tt = c * MINGRU_TILE + i;
                scan.scan_h.data[bHT + tt * H + h] = from_float(h_t);
                h_t = lerp(h_t, sm_ht[i][hx], sm_z[i][hx]);
                float sv = sm_s[i][hx];
                scan.out.data[bHT + tt * H + h] =
                    from_float(sv * h_t + (1.0f - sv) * sm_x[i][hx]);
                h_t = to_float(from_float(h_t));
            }
        }
        __syncthreads();
    }
    if (t_local == 0 && h_ok) {
        scan.next_state.data[b * H + h] = from_float(h_t);
    }
}

__global__ void mingru_scan_backward_row(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H;
    int hx = threadIdx.x, t_local = threadIdx.y;
    int h = blockIdx.y * MINGRU_TILE + hx;
    int h_ok = h < H;
    int b = blockIdx.x, H3 = 3 * H, bHT = b * H * T_seq, cbase = 3 * bHT;
    float dh = h_ok ? to_float(grad_next_state[b * H + h]) : 0.0f;
    __shared__ float sm_z[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_ht[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_s[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_x[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_hp[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_go[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_hid[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_term[MINGRU_TILE][MINGRU_TILE];
    for (int c = (T_seq + MINGRU_TILE - 1) / MINGRU_TILE - 1; c >= 0; c--) {
        int t = c * MINGRU_TILE + t_local;
        float z = 0.0f, h_tilde = 0.0f, s = 0.0f, x = 0.0f;
        float h_prev = 0.0f, gout = 0.0f, hidden = 0.0f, term = 0.0f;
        if (t < T_seq && h_ok) {
            int off = t * H3 + h;
            hidden = to_float(scan.combined_ptr[cbase + off]);
            float gate = to_float(scan.combined_ptr[cbase + H + off]);
            float proj = to_float(scan.combined_ptr[cbase + 2 * H + off]);
            x = to_float(scan.input_ptr[bHT + t * H + h]);
            h_prev = to_float(scan.scan_h.data[bHT + t * H + h]);
            gout = to_float(grad_out[bHT + t * H + h]);
            z = sigmoid(gate);
            h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
            s = sigmoid(proj);
            term = scan.terminals_ptr != NULL &&
                to_float(scan.terminals_ptr[b * T_seq + t]) != 0.0f;
        }
        sm_z[t_local][hx] = z;
        sm_ht[t_local][hx] = h_tilde;
        sm_s[t_local][hx] = s;
        sm_x[t_local][hx] = x;
        sm_hp[t_local][hx] = h_prev;
        sm_go[t_local][hx] = gout;
        sm_hid[t_local][hx] = hidden;
        sm_term[t_local][hx] = term;
        __syncthreads();
        if (t_local == 0 && h_ok) {
            for (int i = min(MINGRU_TILE, T_seq - c * MINGRU_TILE) - 1; i >= 0; i--) {
                int tt = c * MINGRU_TILE + i;
                z = sm_z[i][hx];
                s = sm_s[i][hx];
                gout = sm_go[i][hx];
                h_tilde = sm_ht[i][hx];
                h_prev = sm_hp[i][hx];
                float h_t = lerp(h_prev, h_tilde, z);
                float dh_total = dh + gout * s;
                float d_h_tilde = dh_total * z;
                int off = tt * H3 + h;
                scan.grad_input.data[bHT + tt * H + h] =
                    from_float(gout * (1.0f - s));
                scan.grad_combined.data[cbase + off] = from_float(
                    sm_hid[i][hx] >= 0.0f ? d_h_tilde
                    : d_h_tilde * h_tilde * (1.0f - h_tilde));
                scan.grad_combined.data[cbase + H + off] =
                    from_float(dh_total * (h_tilde - h_prev) * z * (1.0f - z));
                scan.grad_combined.data[cbase + 2 * H + off] =
                    from_float(gout * (h_t - sm_x[i][hx]) * s * (1.0f - s));
                dh = sm_term[i][hx] != 0.0f ? 0.0f : dh_total * (1.0f - z);
            }
        }
        __syncthreads();
    }
    if (t_local == 0 && h_ok) {
        scan.grad_state.data[b * H + h] = from_float(dh);
    }
}

struct MinGRUActivations {
    // Rollout
    Prec* combined;       // (B rollout, 3*T)[num_layers]
    Prec out;             // (B rollout, T)
    Prec next_state;      // (B rollout, T)
    // Training
    Prec* saved_inputs;   // (B, TT, T)[num_layers]
    PrefixScan* scan_bufs;           // [num_layers]
    Prec* combined_bufs;  // (B*TT, 3*T)[num_layers]
    Prec* wgrad_scratch;  // (3*T, T)[num_layers]
    Prec grad_input_buf;  // (B*TT, T)
    Prec grad_next_state; // (B, 1, T)
};

struct MinGRUWeights {
    int hidden, num_layers, horizon;
    Prec* weights;  // [num_layers]
};

Prec mingru_state_layer(Prec& state, int layer, int agent_off, int B) {
    long A = state.shape[1], H = state.shape[2];
    return {
        .data = state.data + ((long)layer * A + agent_off) * H,
        .shape = {B, H},
    };
}

void mingru_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        Prec w2d = {
            .data = m->weights[i].data,
            .shape = {3 * m->hidden, m->hidden},
        };
        puf_kaiming_init(&w2d, 1.0f, (*seed)++, stream);
    }
}

void mingru_reg_params(void* w, Allocator* alloc) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        m->weights[i] = {.shape = {3 * m->hidden, m->hidden}};
        alloc_register(alloc, &m->weights[i]);
    }
}

void mingru_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT;
    a->saved_inputs = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->scan_bufs = (PrefixScan*)calloc(m->num_layers, sizeof(PrefixScan));
    a->combined_bufs = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->wgrad_scratch = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->grad_input_buf = {.shape = {B_TT, H}};
    a->grad_next_state = {.shape = {B, 1, H}};
    alloc_register(acts, &a->grad_input_buf);
    alloc_register(acts, &a->grad_next_state);
    for (int i = 0; i < m->num_layers; i++) {
        a->scan_bufs[i] = {
            .B = B, .T = TT, .H = H,
            .scan_h =           {.shape = {B, TT, H}},
            .out =              {.shape = {B, TT, H}},
            .next_state =       {.shape = {B, 1, H}},
            .grad_combined =    {.shape = {B, TT, 3 * H}},
            .grad_state =       {.shape = {B, 1, H}},
            .grad_input =       {.shape = {B, TT, H}},
        };
        a->saved_inputs[i]  = {.shape = {B, TT, H}};
        a->combined_bufs[i] = {.shape = {B_TT, 3 * H}};
        a->wgrad_scratch[i] = {.shape = {3 * H, H}};
        alloc_register(acts, &a->saved_inputs[i]);
        alloc_register(acts, &a->combined_bufs[i]);
        alloc_register(acts, &a->scan_bufs[i].out);
        alloc_register(acts, &a->scan_bufs[i].next_state);
        alloc_register(acts, &a->scan_bufs[i].scan_h);
        alloc_register(acts, &a->scan_bufs[i].grad_combined);
        alloc_register(acts, &a->scan_bufs[i].grad_state);
        alloc_register(acts, &a->scan_bufs[i].grad_input);
        alloc_register(grads, &a->wgrad_scratch[i]);
    }
}

void mingru_reg_rollout(void* weights, void* activations,
        Allocator* alloc, int B_inf) {
    MinGRUWeights* w = (MinGRUWeights*)weights;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = w->hidden;
    a->combined = (Prec*)calloc(w->num_layers, sizeof(Prec));
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 3 * H}};
        alloc_register(alloc, &a->combined[i]);
    }
    a->out = {.shape = {B_inf, H}};
    a->next_state = {.shape = {B_inf, H}};
    alloc_register(alloc, &a->out);
    alloc_register(alloc, &a->next_state);
}

void* mingru_create_weights(void* self) {
    Network* n = (Network*)self;
    MinGRUWeights* mw = (MinGRUWeights*)calloc(1, sizeof(MinGRUWeights));
    mw->hidden = n->hidden;
    mw->num_layers = n->num_layers;
    mw->horizon = n->horizon;
    mw->weights = (Prec*)calloc(n->num_layers, sizeof(Prec));
    return mw;
}

Prec mingru_forward(void* w, Prec x, Prec state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
    for (int i = 0; i < m->num_layers; i++) {
        Prec state_i = mingru_state_layer(state, i, 0, B);
        puf_mm(&x, &m->weights[i], &a->combined[i], stream);
        mingru_gate<<<grid_size(B*H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, a->next_state.data,
            a->combined[i].data, state_i.data, x.data, H, B);
        puf_copy(&state_i, &a->next_state, stream);
        x = a->out;
    }
    return x;
}

Prec mingru_forward_train(void* w, Prec x, Prec state, Prec terminals,
        void* activations, int agent_off, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = (int)x.shape[0];
    for (int i = 0; i < m->num_layers; i++) {
        puf_copy(&a->saved_inputs[i], &x, stream);
        Prec state_i = mingru_state_layer(state, i, agent_off, B);
        puf_mm(&x, &m->weights[i], &a->combined_bufs[i], stream);
        PrefixScan& scan = a->scan_bufs[i];
        scan.combined_ptr = a->combined_bufs[i].data;
        scan.state_ptr = state_i.data;
        scan.input_ptr = a->saved_inputs[i].data;
        scan.terminals_ptr = terminals.data;
        if (scan.T >= 256 && scan.B * scan.H <= 32768
                || scan.T >= 128 && scan.H <= 32) {
            dim3 block(MINGRU_TILE, MINGRU_TILE);
            dim3 grid(scan.B, (scan.H + MINGRU_TILE - 1) / MINGRU_TILE);
            mingru_scan_forward_row<<<grid, block, 0, stream>>>(scan);
        } else {
            mingru_scan_forward<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0, stream>>>(scan);
        }
        x = scan.out;
    }
    return x;
}

__global__ void add_kernel(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
            idx += blockDim.x * gridDim.x) {
        dst[idx] = from_float(to_float(dst[idx]) + to_float(src[idx]));
    }
}

Prec mingru_backward(void* w, Prec grad, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    for (int i = m->num_layers - 1; i >= 0; i--) {
        PrefixScan& scan = a->scan_bufs[i];
        if (scan.T >= 256 && scan.B * scan.H <= 32768
                || scan.T >= 128 && scan.H <= 32) {
            dim3 block(MINGRU_TILE, MINGRU_TILE);
            dim3 grid(scan.B, (scan.H + MINGRU_TILE - 1) / MINGRU_TILE);
            mingru_scan_backward_row<<<grid, block, 0, stream>>>(
                scan, grad.data, a->grad_next_state.data);
        } else {
            mingru_scan_backward<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0,
                stream>>>(scan, grad.data, a->grad_next_state.data);
        }
        // dW on side stream (per-layer scratch); dX on main continues the bwd chain.
        puf_mm_tn_async_after(&scan.grad_combined, &a->saved_inputs[i],
            &a->wgrad_scratch[i], stream);
        puf_mm_nn(&scan.grad_combined, &m->weights[i], &a->grad_input_buf, stream);
        int n = numel(scan.grad_input.shape);
        add_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            a->grad_input_buf.data, scan.grad_input.data, n);
        grad = a->grad_input_buf;
    }
    return grad;
}
