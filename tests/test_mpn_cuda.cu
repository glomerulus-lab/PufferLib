// Test harness for the MPN-pre network (src/networks/mpn_pre.cu mpnpre_*). Thin ctypes wrapper
// around the real implementation; driven by tests/test_mpn.py, which builds
// this as a shared lib and checks it against a float64 PyTorch reference.
//
// All pointer args are device pointers (torch CUDA tensor data_ptr()).
// Build float (-DPRECISION_FLOAT) for gradient checks; bf16 for parity only.
#include "../src/pufferl.cu"

extern "C" {

static Network g_net;
static MPNPreWeights* g_w = nullptr;
static MPNPreActivations* g_ta = nullptr;  // train activations
static MPNPreActivations* g_ra = nullptr;  // rollout activations
static Allocator g_pa = {}, g_aa = {}, g_ga = {}, g_ra_alloc = {};
static int g_B = 0, g_T = 0, g_H = 0, g_L = 0;

int mpn_precision_bytes() { return (int)sizeof(precision_t); }
int mpn_num_scalars() { return MPN_SCALARS; }
float mpn_lambda_max() { return MPN_LAMBDA_MAX; }

void mpn_init(int B, int T, int H, int L, unsigned long seed) {
    cublas_init_handle();
    g_B = B; g_T = T; g_H = H; g_L = L;
    g_net = make_network(NETWORK_MPN_PRE, H, L, T);
    g_w = (MPNPreWeights*)g_net.create_weights(&g_net);
    g_pa = {}; g_aa = {}; g_ga = {}; g_ra_alloc = {};
    g_net.reg_params(g_w, &g_pa);
    alloc_create(&g_pa);
    g_ta = (MPNPreActivations*)calloc(1, sizeof(MPNPreActivations));
    g_net.reg_train(g_w, g_ta, &g_aa, &g_ga, B * T);
    alloc_create(&g_aa);
    alloc_create(&g_ga);
    g_ra = (MPNPreActivations*)calloc(1, sizeof(MPNPreActivations));
    g_net.reg_rollout(g_w, g_ra, &g_ra_alloc, B);
    alloc_create(&g_ra_alloc);
    ulong s = seed;
    g_net.init_weights(g_w, &s, 0);
    cudaDeviceSynchronize();
}

// Flat param / grad buffers exactly as Muon sees them (layout check).
long mpn_param_elems() { return g_pa.total_elems; }
long mpn_grad_elems() { return g_ga.total_elems; }

static void copy_dev(void* dst, const void* src, long n) {
    cudaMemcpy(dst, src, n * sizeof(precision_t), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}
void mpn_get_w(int l, void* dst) { copy_dev(dst, g_w->weights[l].data, 2L * g_H * g_H); }
void mpn_set_w(int l, void* src) { copy_dev(g_w->weights[l].data, src, 2L * g_H * g_H); }
void mpn_get_scalars(int l, void* dst) { copy_dev(dst, g_w->scalars[l].data, MPN_SCALARS); }
void mpn_set_scalars(int l, void* src) { copy_dev(g_w->scalars[l].data, src, MPN_SCALARS); }
void mpn_get_wgrad(int l, void* dst) { copy_dev(dst, g_ta->wgrad_scratch[l].data, 2L * g_H * g_H); }
void mpn_get_sgrad(int l, void* dst) { copy_dev(dst, g_ta->sgrad_scratch[l].data, MPN_SCALARS); }

// x: (B, T, H), state: (L, B, H) (read only), terminals: (B, T) -> out (B, T, H)
void mpn_train_forward(void* out, void* x, void* state, void* terminals) {
    Prec xp = {.data = (precision_t*)x, .shape = {g_B, g_T, g_H}};
    Prec sp = {.data = (precision_t*)state, .shape = {g_L, g_B, g_H}};
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

// One rollout step. x: (B, H), state: (L, B, H) updated in place -> out (B, H).
// Terminal zeroing is the trainer's job (zero_term_state); caller does it.
void mpn_rollout_step(void* out, void* x, void* state) {
    Prec xp = {.data = (precision_t*)x, .shape = {g_B, g_H}};
    Prec sp = {.data = (precision_t*)state, .shape = {g_L, g_B, g_H}};
    Prec r = g_net.forward(g_w, xp, sp, g_ra, 0);
    copy_dev(out, r.data, (long)g_B * g_H);
}

}
