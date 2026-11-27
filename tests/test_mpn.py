"""Check the MPN networks against float64 PyTorch references.

pre  (src/networks/mpn_pre.cu), MPNpre + highway, state m (H) per layer:
    lambda = 0.95 * sigmoid(rho),  c = eta / sqrt(H)
    u_t   = x_t * (1 + m_{t-1})            (m zeroed before step t if terminal[t])
    [a|g] = u_t @ W^T,  s = sigmoid(g)
    out_t = s * tanh(a) + (1 - s) * x_t
    m_t   = lambda * m_{t-1} + c * x_t

full (src/networks/mpn.cu), MPN + highway, state M (H x H) per layer:
    [b|g] = x_t @ W^T,  a = b + ((W_a * M_{t-1}) x_t)   (W_a = W[:H], * elementwise;
                                                        M zeroed if terminal[t])
    out_t = s * tanh(a) + (1 - s) * x_t,  s = sigmoid(g)
    M_t   = lambda * M_{t-1} + c * tanh(a) x_t^T

Checks (per network):
  cuda (float build): param/grad layout, init values, train forward, all
      gradients (W, rho, eta, x) vs autograd, rollout-step vs train parity,
      final state.
  bf16 (bf16 build):  rollout vs train parity and loose agreement with reference.
  cpu:                src/puffercpu.c rnn_forward() step vs reference.

Run: python tests/test_mpn.py [cuda] [bf16] [cpu] [pre] [full]
     (default: all modes, both networks)
"""
import ctypes
import glob
import math
import os
import shutil
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VP = ctypes.c_void_p
LAMBDA_MAX = 0.95
NSCAL = 8  # MPN_SCALARS
NETWORK_ID = {"pre": 1, "full": 2}  # NetworkType in src/algo.cu / src/puffercpu.c


def ref_forward(x, state0, term, Ws, rhos, etas):
    """x (B,T,H), state0 (L,B,H), term (B,T); per-layer W (2H,H), rho, eta scalars."""
    B, T, H = x.shape
    inp, finals = x, []
    for W, rho, eta in zip(Ws, rhos, etas):
        lam = LAMBDA_MAX * torch.sigmoid(rho)
        c = eta / math.sqrt(H)
        m = state0[len(finals)]
        outs = []
        for t in range(T):
            m = torch.where(term[:, t, None] != 0, torch.zeros_like(m), m)
            xt = inp[:, t]
            comb = (xt * (1 + m)) @ W.T
            s = torch.sigmoid(comb[:, H:])
            outs.append(s * torch.tanh(comb[:, :H]) + (1 - s) * xt)
            m = lam * m + c * xt
        inp = torch.stack(outs, 1)
        finals.append(m)
    return inp, torch.stack(finals)


def ref_forward_full(x, state0, term, Ws, rhos, etas):
    """x (B,T,H), state0 (L,B,H*H) row-major M[i][j], term (B,T); per-layer W (2H,H)."""
    B, T, H = x.shape
    inp, finals = x, []
    for W, rho, eta in zip(Ws, rhos, etas):
        lam = LAMBDA_MAX * torch.sigmoid(rho)
        c = eta / math.sqrt(H)
        M = state0[len(finals)].reshape(B, H, H)
        Wa = W[:H]
        outs = []
        for t in range(T):
            M = torch.where(term[:, t, None, None] != 0, torch.zeros_like(M), M)
            xt = inp[:, t]
            comb = xt @ W.T
            a = comb[:, :H] + torch.einsum("ij,bij,bj->bi", Wa, M, xt)
            h = torch.tanh(a)
            s = torch.sigmoid(comb[:, H:])
            outs.append(s * h + (1 - s) * xt)
            M = lam * M + c * h[:, :, None] * xt[:, None, :]
        inp = torch.stack(outs, 1)
        finals.append(M.reshape(B, H * H))
    return inp, torch.stack(finals)


REF = {"pre": ref_forward, "full": ref_forward_full}


def state_size(net, H):
    return H * H if net == "full" else H


def make_case(B, T, H, L, seed, net="pre"):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, H, generator=g, dtype=torch.float64)
    state0 = 0.5 * torch.randn(L, B, state_size(net, H), generator=g, dtype=torch.float64)
    term = (torch.rand(B, T, generator=g) < 0.1).double()
    gout = torch.randn(B, T, H, generator=g, dtype=torch.float64)
    rhos = [torch.tensor(0.5 + 0.7 * l, dtype=torch.float64) for l in range(L)]
    etas = [torch.tensor(0.8 - 1.1 * l, dtype=torch.float64) for l in range(L)]
    return x, state0, term, gout, rhos, etas


def rel_err(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


class Report:
    def __init__(self):
        self.fails = 0

    def check(self, name, err, tol):
        ok = err <= tol
        self.fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<34} err={err:.3e}  tol={tol:.0e}")


# ---------------------------------------------------------------- CUDA harness

def build_cuda(bf16):
    so = os.path.join(HERE, "mpn_test_bf16.so" if bf16 else "mpn_test.so")
    cuda = os.path.dirname(os.path.dirname(shutil.which("nvcc")))
    raylibs = glob.glob(os.path.join(ROOT, "raylib-*_linux_amd64"))
    assert raylibs, "run ./build.sh <env> once to download raylib"
    raylib = raylibs[0]
    cmd = [
        "nvcc", "-shared", "-o", so, os.path.join(HERE, "test_mpn_cuda.cu"),
        "-std=c++17", "-arch=native", "-O2",
        "-I" + ROOT, "-I" + os.path.join(ROOT, "src"),
        "-I" + os.path.join(ROOT, "ocean", "minimal"),
        "-I" + os.path.join(ROOT, "vendor"),
        "-I" + os.path.join(cuda, "include"),
        "-I" + os.path.join(cuda, "include", "cccl"),
        "-I" + os.path.join(raylib, "include"),
        '-DENV_HEADER="ocean/minimal/minimal.h"',
        "-DENV_NAME=minimal", '-DPUFFER_ENV_NAME="minimal"',
        "-Xcompiler=-DPLATFORM_DESKTOP", "-Xcompiler=-fPIC", "-Xcompiler=-fopenmp",
        "-Xcompiler=-Wno-narrowing", "--diag-suppress=2361",
        "--diag-suppress=111", "--diag-suppress=128",
        "-L" + os.path.join(raylib, "lib"), "-lraylib",
        "-Xlinker", "-rpath", "-Xlinker", os.path.join(raylib, "lib"),
        "-lcublas", "-lcublasLt", "-lcusolver", "-lcurand", "-lnvidia-ml", "-lcudart",
    ]
    if not bf16:
        cmd.append("-DPRECISION_FLOAT")
    try:
        import nvidia.nccl
        nccl = nvidia.nccl.__path__[0]
        cmd += ["-I" + os.path.join(nccl, "include"),
                "-L" + os.path.join(nccl, "lib"), "-lnccl"]
    except ImportError:
        cmd += ["-lnccl"]
    print("building:", " ".join(cmd))
    subprocess.check_call(cmd)
    lib = ctypes.CDLL(so)
    lib.mpn_init.argtypes = [ctypes.c_int] * 4 + [ctypes.c_ulong, ctypes.c_int]
    lib.mpn_init.restype = ctypes.c_int
    lib.mpn_param_elems.restype = ctypes.c_long
    lib.mpn_grad_elems.restype = ctypes.c_long
    lib.mpn_lambda_max.restype = ctypes.c_float
    for fn in ("mpn_get_w", "mpn_set_w", "mpn_get_scalars", "mpn_set_scalars",
               "mpn_get_wgrad", "mpn_get_sgrad"):
        getattr(lib, fn).argtypes = [ctypes.c_int, VP]
    lib.mpn_train_forward.argtypes = [VP] * 4
    lib.mpn_train_backward.argtypes = [VP] * 2
    lib.mpn_rollout_step.argtypes = [VP] * 3
    return lib


def ptr(t):
    assert t.is_contiguous()
    return VP(t.data_ptr())


def run_cuda(rep, bf16, net, lib):
    B, T, H, L = 6, 37, 64, 2
    dt = torch.bfloat16 if bf16 else torch.float32
    assert lib.mpn_precision_bytes() == (2 if bf16 else 4)
    assert lib.mpn_num_scalars() == NSCAL
    assert abs(lib.mpn_lambda_max() - LAMBDA_MAX) < 1e-7
    S = lib.mpn_init(B, T, H, L, 1234, NETWORK_ID[net])
    assert S == state_size(net, H), (S, net)
    dev = "cuda"
    tag = f"{'bf16' if bf16 else 'cuda'} {net}"

    print(f"[{tag}] layout + init")
    expect = L * (2 * H * H + NSCAL)
    rep.check("param elems == L*(2H^2+8)", abs(lib.mpn_param_elems() - expect), 0)
    rep.check("grad elems == param elems",
              abs(lib.mpn_grad_elems() - lib.mpn_param_elems()), 0)
    s = torch.zeros(NSCAL, dtype=dt, device=dev)
    lib.mpn_get_scalars(0, ptr(s))
    init = torch.tensor([3.0, 1.0] + [0.0] * (NSCAL - 2))
    rep.check("init scalars [3, 1, 0...]", (s.float().cpu() - init).abs().max().item(), 0)
    w0 = torch.zeros(2 * H, H, dtype=dt, device=dev)
    lib.mpn_get_w(0, ptr(w0))
    bound = 1.0 / math.sqrt(H)
    rep.check("init W within kaiming bound",
              max(0.0, w0.float().abs().max().item() - bound), 1e-2)

    x, state0, term, gout, rhos, etas = make_case(B, T, H, L, seed=0, net=net)
    Ws = []
    for l in range(L):
        w = torch.zeros(2 * H, H, dtype=dt, device=dev)
        lib.mpn_get_w(l, ptr(w))
        Ws.append(w.double().cpu())
        sc = torch.zeros(NSCAL, dtype=torch.float64)
        sc[0], sc[1] = rhos[l], etas[l]
        sc = sc.to(dt).to(dev)
        lib.mpn_set_scalars(l, ptr(sc))
        rhos[l] = sc[0].double().cpu()  # use exactly what the kernel sees
        etas[l] = sc[1].double().cpu()

    # bf16: feed the reference the same rounded inputs the kernel sees
    xd, sd, td, gd = (t.to(dt).to(dev).contiguous() for t in (x, state0, term, gout))
    x, state0, gout = xd.double().cpu(), sd.double().cpu(), gd.double().cpu()

    out = torch.zeros(B, T, H, dtype=dt, device=dev)
    lib.mpn_train_forward(ptr(out), ptr(xd), ptr(sd), ptr(td))

    # Rollout: trainer zeroes state rows on terminal before the step.
    st = sd.clone()
    roll = torch.zeros(B, T, H, dtype=dt, device=dev)
    for t in range(T):
        st[:, td[:, t] != 0] = 0
        xt = xd[:, t].contiguous()
        ot = torch.zeros(B, H, dtype=dt, device=dev)
        lib.mpn_rollout_step(ptr(ot), ptr(xt), ptr(st))
        roll[:, t] = ot

    xr = x.clone().requires_grad_(True)
    Wr = [w.clone().requires_grad_(True) for w in Ws]
    rr = [r.clone().requires_grad_(True) for r in rhos]
    er = [e.clone().requires_grad_(True) for e in etas]
    ref_out, ref_state = REF[net](xr, state0, term, Wr, rr, er)

    print(f"[{tag}] forward")
    tol_f = 3e-2 if bf16 else 1e-5
    rep.check("train forward vs reference", rel_err(out, ref_out.detach()), tol_f)
    rep.check("rollout steps vs train forward", rel_err(roll, out),
              2e-2 if bf16 else 1e-5)
    rep.check("rollout final state vs reference", rel_err(st, ref_state.detach()),
              3e-2 if bf16 else 1e-5)
    if bf16:
        return

    print(f"[{tag}] backward")
    (ref_out * gout).sum().backward()
    dx = torch.zeros(B, T, H, dtype=dt, device=dev)
    lib.mpn_train_backward(ptr(dx), ptr(gd))
    tol_b = 1e-4
    rep.check("dx", rel_err(dx, xr.grad), tol_b)
    for l in range(L):
        dw = torch.zeros(2 * H, H, dtype=dt, device=dev)
        lib.mpn_get_wgrad(l, ptr(dw))
        rep.check(f"dW[{l}]", rel_err(dw, Wr[l].grad), tol_b)
        sg = torch.zeros(NSCAL, dtype=dt, device=dev)
        lib.mpn_get_sgrad(l, ptr(sg))
        sg = sg.double().cpu()
        rep.check(f"drho[{l}]", rel_err(sg[0:1], rr[l].grad.view(1)), tol_b)
        rep.check(f"deta[{l}]", rel_err(sg[1:2], er[l].grad.view(1)), tol_b)
        rep.check(f"scalar grad pad[{l}] == 0", sg[2:].abs().max().item(), 0)


# ---------------------------------------------------------------- CPU mirror

class CWeights(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("size", ctypes.c_int), ("idx", ctypes.c_int)]


class CRNN(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                ("state", ctypes.POINTER(ctypes.c_float)),
                ("output", ctypes.POINTER(ctypes.c_float)),
                ("u", ctypes.POINTER(ctypes.c_float)),
                ("proj", VP), ("scalars", VP),
                ("batch_size", ctypes.c_int), ("hidden_size", ctypes.c_int),
                ("num_layers", ctypes.c_int), ("state_size", ctypes.c_int)]


def build_cpu():
    so = os.path.join(HERE, "puffercpu_test.so")
    cc = shutil.which("clang") or shutil.which("cc")
    cmd = [cc, "-shared", "-fPIC", "-O2", os.path.join(ROOT, "src", "puffercpu.c"),
           "-o", so, "-lm"]
    print("building:", " ".join(cmd))
    subprocess.check_call(cmd)
    lib = ctypes.CDLL(so)
    lib.make_rnn.restype = ctypes.POINTER(CRNN)
    lib.make_rnn.argtypes = [ctypes.POINTER(CWeights)] + [ctypes.c_int] * 4
    lib.rnn_forward.argtypes = [ctypes.POINTER(CRNN), VP]
    lib.rnn_zero_term.argtypes = [ctypes.POINTER(CRNN), VP]
    return lib


def run_cpu(rep, net, lib):
    B, T, H, L = 5, 23, 48, 3
    S = state_size(net, H)
    x, state0, term, _, rhos, etas = make_case(B, T, H, L, seed=1, net=net)
    g = torch.Generator().manual_seed(2)
    Ws = [torch.empty(2 * H, H, dtype=torch.float64).uniform_(-1, 1, generator=g) / math.sqrt(H)
          for _ in range(L)]

    # Flat file layout: per layer W (2H*H, 8-aligned) then [rho, eta, 0...] (8).
    chunks = []
    for W, rho, eta in zip(Ws, rhos, etas):
        w = W.float().flatten()
        pad = (-w.numel()) % 8
        sc = torch.zeros(NSCAL)
        sc[0], sc[1] = rho, eta
        chunks += [w, torch.zeros(pad), sc]
    flat = torch.cat(chunks).float().contiguous()
    weights = CWeights(ctypes.cast(flat.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                       flat.numel(), 0)
    rnn = lib.make_rnn(ctypes.byref(weights), NETWORK_ID[net], B, H, L)
    rep.check(f"cpu {net} weights consumed == file size", abs(weights.idx - flat.numel()), 0)

    st = torch.from_numpy(
        __import__("numpy").ctypeslib.as_array(rnn.contents.state, shape=(L, B, S)))
    st.copy_(state0.float())
    outv = __import__("numpy").ctypeslib.as_array(rnn.contents.output, shape=(B, H))
    xf = x.float().contiguous()
    out = torch.zeros(B, T, H)
    for t in range(T):
        tt = term[:, t].float().contiguous()
        lib.rnn_zero_term(rnn, VP(tt.data_ptr()))
        xt = xf[:, t].contiguous()
        lib.rnn_forward(rnn, VP(xt.data_ptr()))
        out[:, t] = torch.from_numpy(outv.copy())

    ref_out, ref_state = REF[net](x, state0, term, Ws, rhos, etas)
    print(f"[cpu {net}] forward")
    rep.check(f"cpu rnn_forward({net}) vs reference", rel_err(out, ref_out), 1e-5)
    rep.check("cpu final state vs reference", rel_err(st, ref_state), 1e-5)


if __name__ == "__main__":
    args = sys.argv[1:]
    modes = [a for a in args if a in ("cuda", "bf16", "cpu")] or ["cuda", "bf16", "cpu"]
    nets = [a for a in args if a in ("pre", "full")] or ["pre", "full"]
    rep = Report()
    if "cuda" in modes:
        lib = build_cuda(bf16=False)
        for net in nets:
            run_cuda(rep, False, net, lib)
    if "bf16" in modes:
        lib = build_cuda(bf16=True)
        for net in nets:
            run_cuda(rep, True, net, lib)
    if "cpu" in modes:
        lib = build_cpu()
        for net in nets:
            run_cpu(rep, net, lib)
    print(f"\n{'ALL PASS' if rep.fails == 0 else f'{rep.fails} FAILED'}")
    sys.exit(1 if rep.fails else 0)
