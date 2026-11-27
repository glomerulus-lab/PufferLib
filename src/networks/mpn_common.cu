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
