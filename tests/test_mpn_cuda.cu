// Test harness for the MPN networks (src/networks/mpn_pre.cu and mpn.cu). Thin
// ctypes wrapper around the real implementation; driven by tests/test_mpn.py,
// which builds this as a shared lib and checks it against a float64 PyTorch
// reference. Both variants share MPNWeights and the param layout, so one
// harness serves both; `network` picks the variant (NetworkType in algo.cu).
//
// All pointer args are device pointers (torch CUDA tensor data_ptr()).
// Build float (-DPRECISION_FLOAT) for gradient checks; bf16 for parity only.
#include "../src/pufferl.cu"

extern "C" {

static Network g_net;
static MPNWeights* g_w = nullptr;
static void* g_ta = nullptr;  // train activations
static void* g_ra = nullptr;  // rollout activations
static Allocator g_pa = {}, g_aa = {}, g_ga = {}, g_ra_alloc = {};
static int g_B = 0, g_T = 0, g_H = 0, g_L = 0, g_S = 0;

int mpn_precision_bytes() { return (int)sizeof(precision_t); }
int mpn_num_scalars() { return MPN_SCALARS; }
float mpn_lambda_max() { return MPN_LAMBDA_MAX; }

// network: 1 = mpn_pre, 2 = mpn (full). Returns the per-agent state size.
int mpn_init(int B, int T, int H, int L, unsigned long seed, int network) {
    cublas_init_handle();
    g_B = B; g_T = T; g_H = H; g_L = L;
    g_net = make_network((NetworkType)network, H, L, T);
    g_S = g_net.state_size;
    g_w = (MPNWeights*)g_net.create_weights(&g_net);
    g_pa = {}; g_aa = {}; g_ga = {}; g_ra_alloc = {};
    g_net.reg_params(g_w, &g_pa);
    alloc_create(&g_pa);
    g_ta = calloc(1, g_net.activation_size);
    g_net.reg_train(g_w, g_ta, &g_aa, &g_ga, B * T);
    alloc_create(&g_aa);
    alloc_create(&g_ga);
    g_ra = calloc(1, g_net.activation_size);
    g_net.reg_rollout(g_w, g_ra, &g_ra_alloc, B);
    alloc_create(&g_ra_alloc);
    ulong s = seed;
    g_net.init_weights(g_w, &s, 0);
    cudaDeviceSynchronize();
    return g_S;
}

// Flat param / grad buffers exactly as Muon sees them (layout check).
long mpn_param_elems() { return g_pa.total_elems; }
long mpn_grad_elems() { return g_ga.total_elems; }

static void copy_dev(void* dst, const void* src, long n) {
    cudaMemcpy(dst, src, n * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}
// Grads are read at the param's byte offset in the grads buffer: the layout
// Muon relies on, so the test exercises it directly.
static const void* grad_of(const void* param) {
    return (const char*)g_ga.mem + ((const char*)param - (const char*)g_pa.mem);
}
void mpn_get_w(int l, void* dst) { copy_dev(dst, g_w->weights[l].data, 2L * g_H * g_H); }
void mpn_set_w(int l, void* src) { copy_dev(g_w->weights[l].data, src, 2L * g_H * g_H); }
void mpn_get_scalars(int l, void* dst) { copy_dev(dst, g_w->scalars[l].data, MPN_SCALARS); }
void mpn_set_scalars(int l, void* src) { copy_dev(g_w->scalars[l].data, src, MPN_SCALARS); }
void mpn_get_wgrad(int l, void* dst) {
    copy_dev(dst, grad_of(g_w->weights[l].data), 2L * g_H * g_H);
}
void mpn_get_sgrad(int l, void* dst) {
    copy_dev(dst, grad_of(g_w->scalars[l].data), MPN_SCALARS);
}

// x: (B, T, H), state: (L, B, S) (read only), terminals: (B, T) -> out (B, T, H)
void mpn_train_forward(void* out, void* x, void* state, void* terminals) {
    Prec xp = {.data = (precision_t*)x, .shape = {g_B, g_T, g_H}};
    Prec sp = {.data = (precision_t*)state, .shape = {g_L, g_B, g_S}};
    Prec tp = {.data = (precision_t*)terminals, .shape = {g_B, g_T}};
    Prec r = g_net.forward_train(g_w, xp, sp, tp, g_ta, 0, 0);
    copy_dev(out, r.data, (long)g_B * g_T * g_H);
}

// grad_out: (B, T, H) -> dx (B, T, H); weight grads readable after.
void mpn_train_backward(void* dx, void* grad_out) {
    Prec g = {.data = (precision_t*)grad_out, .shape = {g_B, g_T, g_H}};
    Prec r = g_net.backward(g_w, g, g_ta, 0);
    puf_dw_join(0);
    copy_dev(dx, r.data, (long)g_B * g_T * g_H);
}

// One rollout step. x: (B, H), state: (L, B, S) updated in place -> out (B, H).
// Terminal zeroing is the trainer's job (zero_term_state); caller does it.
void mpn_rollout_step(void* out, void* x, void* state) {
    Prec xp = {.data = (precision_t*)x, .shape = {g_B, g_H}};
    Prec sp = {.data = (precision_t*)state, .shape = {g_L, g_B, g_S}};
    Prec r = g_net.forward(g_w, xp, sp, g_ra, 0);
    copy_dev(out, r.data, (long)g_B * g_H);
}

}
