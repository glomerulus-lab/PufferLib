// Shared by the MPN variants (networks/mpn_pre.cu, networks/mpn.cu).
// Both store per-layer plasticity scalars as a 1-D param [rho, eta, 0...]:
//   lambda = MPN_LAMBDA_MAX * sigmoid(rho)   (decay, always < 1)
//   c      = eta * (variant-specific scale)  (Hebbian learning rate)
#pragma once

#define MPN_LAMBDA_MAX 0.95f
// Padded to 8 so Muon's flat (unpadded) param offsets stay aligned with the
// allocator's 16-byte-aligned layout. Pad slots always get zero gradient.
#define MPN_SCALARS 8

__global__ void mpn_scalar_init(precision_t* scal, float rho, float eta) {
    int i = threadIdx.x;
    if (i < MPN_SCALARS) {
        scal[i] = from_float(i == 0 ? rho : (i == 1 ? eta : 0.0f));
    }
}

// View one layer's agents [agent_off, agent_off+B) of a (L, A, S) state tensor.
Prec mpn_state_layer(Prec& state, int layer, int agent_off, int B) {
    long A = state.shape[1], S = state.shape[2];
    return {
        .data = state.data + ((long)layer * A + agent_off) * S,
        .shape = {B, S},
    };
}

// Init: lambda = 0.95*sigmoid(3) ~= 0.905 (time constant ~10.5 steps), c = 1/sqrt(H).
#define MPN_INIT_RHO 3.0f
#define MPN_INIT_ETA 1.0f

__device__ __forceinline__ void mpn_coeffs(const precision_t* __restrict__ scal,
        int H, float* lam, float* c) {
    *lam = MPN_LAMBDA_MAX * sigmoid(to_float(scal[0]));
    *c = to_float(scal[1]) * rsqrtf((float)H);
}

// out = s*tanh(a) + (1-s)*x over rows of combined = [a(H) | g(H)].
// out may alias x (rollout reuses one buffer across layers).
__global__ void mpn_post(precision_t* out, const precision_t* __restrict__ combined,
        const precision_t* x, int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H) {
        return;
    }
    int r = idx / H, j = idx % H;
    float a = to_float(combined[r * 2 * H + j]);
    float s = sigmoid(to_float(combined[r * 2 * H + H + j]));
    float xv = to_float(x[idx]);
    out[idx] = from_float(s * tanhf(a) + (1.0f - s) * xv);
}

// Chain (dlambda, dc) to the stored params (rho, eta); zero the pad slots.
__global__ void mpn_scalar_grad(precision_t* __restrict__ sgrad,
        const float* __restrict__ scal_acc, const precision_t* __restrict__ scal, int H) {
    int i = threadIdx.x;
    if (i >= MPN_SCALARS) {
        return;
    }
    float v = 0.0f;
    if (i == 0) {
        float sg = sigmoid(to_float(scal[0]));
        v = scal_acc[0] * MPN_LAMBDA_MAX * sg * (1.0f - sg);
    } else if (i == 1) {
        v = scal_acc[1] * rsqrtf((float)H);
    }
    sgrad[i] = from_float(v);
}

// Both variants share the parameter layout: per layer W (2H, H) = [W_a; W_g],
// then the scalar vector. Order (W, scalars) per layer must match each variant's
// reg_train grads and the CPU loader.
struct MPNWeights {
    int hidden, num_layers, horizon;
    Prec* weights;  // (2H, H)[num_layers]
    Prec* scalars;  // (MPN_SCALARS)[num_layers]: [rho, eta, 0...]
};

void mpn_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MPNWeights* m = (MPNWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        Prec w2d = {
            .data = m->weights[i].data,
            .shape = {2 * m->hidden, m->hidden},
        };
        puf_kaiming_init(&w2d, 1.0f, (*seed)++, stream);
        mpn_scalar_init<<<1, MPN_SCALARS, 0, stream>>>(
            m->scalars[i].data, MPN_INIT_RHO, MPN_INIT_ETA);
    }
}

void mpn_reg_params(void* w, Allocator* alloc) {
    MPNWeights* m = (MPNWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        m->weights[i] = {.shape = {2 * m->hidden, m->hidden}};
        m->scalars[i] = {.shape = {MPN_SCALARS}};
        alloc_register(alloc, &m->weights[i]);
        alloc_register(alloc, &m->scalars[i]);
    }
}

void* mpn_create_weights(void* self) {
    Network* n = (Network*)self;
    MPNWeights* mw = (MPNWeights*)calloc(1, sizeof(MPNWeights));
    mw->hidden = n->hidden;
    mw->num_layers = n->num_layers;
    mw->horizon = n->horizon;
    mw->weights = (Prec*)calloc(n->num_layers, sizeof(Prec));
    mw->scalars = (Prec*)calloc(n->num_layers, sizeof(Prec));
    return mw;
}
