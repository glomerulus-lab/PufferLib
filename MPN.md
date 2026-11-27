# MPN: Multi-Plasticity Network layer

This document describes the **MPNpre** recurrent layer: the presynaptic variant
of the multi-plasticity network from Aitken & Mihalas (2023), *eLife* 12:e83035
(Eq. 1 and Eq. 13), with PufferLib's gated highway connection added on top.

## Selecting the recurrent network

The trainer and the CPU inference binaries pick the recurrent core at runtime
from `[policy] network` (default `mpn_pre`, set in `config/default.ini`):

| `network` | Implementation | State per layer per agent |
|---|---|---|
| `mingru` | `src/networks/mingru.cu` (the 5.0 MinGRU) | $H$ |
| `mpn_pre` | `src/networks/mpn_pre.cu` (this document) | $H$ |
| `mpn` | `src/networks/mpn.cu` (see [Full MPN](#full-mpn)) | $H^2$ |

`build_arch` in `src/algo.cu` fills the `Network` vtable via `make_network`,
which also sets `state_size`; the trainer allocates recurrent state as
`(num_layers, agents, state_size)`. On CPU, `src/puffercpu.c` exposes one
`RNN` type (`make_rnn`, `rnn_forward`, `rnn_zero_term`, `rnn_reset`,
`rnn_weight_count`) that every env net uses. Standalone demos that do not read
the ini take the network explicitly: `--network NAME` for the OSRS viewers and
`NH_NETWORK=NAME` for the NetHack demo (both default to `mpn_pre`).

Implementation of MPNpre: `src/networks/mpn_pre.cu` (`mpnpre_*` kernels and
`mpnpre_forward`, `mpnpre_forward_train`, `mpnpre_backward`; shared constants in
`src/networks/mpn_common.cu`), CPU inference mirror in `src/puffercpu.c`
(`mpnpre_layer`), float64 reference in `tests/test_mpn.py`.

## Notation

| Symbol | Shape | Meaning |
|---|---|---|
| $H$ | — | hidden width (input and output width of every layer) |
| $x_t$ | $\mathbb{R}^H$ | layer input at step $t$ |
| $m_t$ | $\mathbb{R}^H$ | plastic state (the recurrent carry) after step $t$ |
| $W =  [W_a;  W_g]$ | $\mathbb{R}^{2H \times H}$ | slow weights (no bias) |
| $\rho, \eta$ | scalars | learned plasticity parameters, one pair per layer |
| $d_t$ | $\{0,1\}$ | terminal flag delivered together with $x_t$ |
| $\sigma$ | — | logistic sigmoid |
| $\odot$ | — | elementwise product |

## Forward pass

Per layer, the two scalar parameters are mapped to a decay and a learning rate:

$$
\lambda = \lambda_{\max} \sigma(\rho), \qquad \lambda_{\max} = 0.95,
\qquad c = \frac{\eta}{\sqrt{H}} .
$$

So $\lambda \in (0, 0.95)$ always, which keeps the state recurrence strictly
contracting.

Given the carried state $m_{t-1}$, each step is:

$$
\begin{aligned}
\tilde m_{t-1} &= (1 - d_t)  m_{t-1} && \text{(reset at episode boundary)} \\
u_t &= x_t \odot (1 + \tilde m_{t-1}) && \text{(presynaptic modulation)} \\
a_t &= W_a u_t, \quad g_t = W_g u_t && \text{(one } 2H \times H \text{ GEMM)} \\
s_t &= \sigma(g_t) \\
y_t &= s_t \odot \tanh(a_t) + (1 - s_t) \odot x_t && \text{(gated highway output)} \\
m_t &= \lambda  \tilde m_{t-1} + c  x_t && \text{(plasticity update)}
\end{aligned}
$$

$y_t$ is the layer output and becomes the next layer's $x_t$. Layers are
stacked $L$ deep, each with its own $W$, $\rho$, $\eta$ and state $m$; the
carried state tensor is $(L, A, H)$ for $A$ agents, the same shape MinGRU used.

### Relation to the MPN of Aitken & Mihalas

The general MPN modulates a weight matrix with a fast, plastic matrix $M_t$:

$$
h_t = \phi\big((W \odot (1 + M_{t-1}))  x_t\big), \qquad
M_t = \lambda M_{t-1} + \eta h_t  x_t^{\top}.
$$

In the **pre**-synaptic variant the Hebbian update depends only on the
presynaptic activity, $M_t = \lambda M_{t-1} + \eta \mathbf{1} x_t^\top$, so (from
a zero or row-constant start) every row of $M$ is the same vector $m_t^\top$.
The modulated product then factors:

$$
(W \odot (1 + \mathbf{1} m^\top))  x = W \big(x \odot (1 + m)\big) = W u .
$$

This is why the layer only stores an $H$-vector of state rather than an
$H \times H$ matrix, and why the forward pass is a single dense GEMM on the
modulated input $u_t$. Here the modulation applies to both halves of $W$, so
it scales the inputs to the candidate $a_t$ and to the gate $g_t$ alike.

### Properties

- **The state is a leaky integral of the input.** Ignoring resets,
  $m_t = \lambda^{t} m_0 + c \sum_{k=0}^{t-1} \lambda^{k} x_{t-k}$.
  Its effective time constant is $1/(1-\lambda)$ steps, and its steady-state
  gain is $c/(1-\lambda)$. For bounded input,
  $\|m\|_\infty \le \max(\|m_0\|_\infty,\ c \|x\|_\infty / (1-\lambda))$.
- **There is no hidden-to-state feedback.** $m$ depends only on the inputs
  $x$, never on $y$ or $W$. The recurrence is a diagonal linear system per
  layer, so training over a horizon is a cheap elementwise scan, and all the
  nonlinear work is in the time-parallel GEMM and the output gate.
- **Recurrence is multiplicative.** Past inputs act on the present only by
  rescaling the current input per feature, $u_{t,j} = x_{t,j}(1 + m_{t-1,j})$.
  When $m = 0$ the layer reduces to a feedforward gated highway block.

### Initialization

| Parameter | Init | Resulting value |
|---|---|---|
| $W$ | Kaiming (gain 1) on the $(2H, H)$ matrix | — |
| $\rho$ | $3.0$ | $\lambda = 0.95 \sigma(3) \approx 0.905$ (time constant ≈ 10.5 steps) |
| $\eta$ | $1.0$ | $c = 1/\sqrt{H}$ |

## Training: truncated BPTT over the horizon

`mpnpre_forward_train` processes a segment of $T$ steps (the horizon) starting
from the stored state $m_0$. Gradients do not flow into $m_0$, and the final
state $m_T$ receives no gradient (truncated BPTT). Let
$\delta_t = \partial \mathcal L / \partial y_t$ and $h_t = \tanh(a_t)$.

**Output gate (time-parallel, `mpnpre_post_backward`):**

$$
\frac{\partial \mathcal L}{\partial a_t} = \delta_t \odot s_t \odot (1 - h_t^2), \qquad
\frac{\partial \mathcal L}{\partial g_t} = \delta_t \odot (h_t - x_t) \odot s_t (1 - s_t), \qquad
\left.\frac{\partial \mathcal L}{\partial x_t}\right|_{\text{highway}} = \delta_t \odot (1 - s_t).
$$

**Weights (time-parallel GEMMs):** with $e_t = [\partial_{a_t}; \partial_{g_t}] \in \mathbb{R}^{2H}$,

$$
\frac{\partial \mathcal L}{\partial u_t} = W^\top e_t, \qquad
\frac{\partial \mathcal L}{\partial W} = \sum_{b,t} e_t  u_t^\top .
$$

**State adjoint (reverse scan, `mpnpre_scan_backward`):** define
$\gamma_t = \partial \mathcal L / \partial m_t$. Then $\gamma_T = 0$ and, for
$t = T, \dots, 1$,

$$
\gamma_{t-1} = (1 - d_t) \big(\lambda  \gamma_t + \tfrac{\partial \mathcal L}{\partial u_t} \odot x_t\big).
$$

The $(1 - d_t)$ factor cuts the gradient at a reset: $m_{t-1}$ was replaced
by zero, so nothing flows to it.

**Input gradient:**

$$
\frac{\partial \mathcal L}{\partial x_t}
= \delta_t \odot (1 - s_t)
+ \frac{\partial \mathcal L}{\partial u_t} \odot (1 + \tilde m_{t-1})
+ c  \gamma_t .
$$

**Plasticity scalars** (summed over batch, time and features, then chained
through the reparameterization in `mpnpre_scalar_grad`):

$$
\frac{\partial \mathcal L}{\partial \lambda} = \sum_{b,t,j} \gamma_{t,j}  \tilde m_{t-1,j}, \qquad
\frac{\partial \mathcal L}{\partial c} = \sum_{b,t,j} \gamma_{t,j}  x_{t,j},
$$

$$
\frac{\partial \mathcal L}{\partial \rho} = \frac{\partial \mathcal L}{\partial \lambda} 
\lambda_{\max} \sigma(\rho) (1 - \sigma(\rho)), \qquad
\frac{\partial \mathcal L}{\partial \eta} = \frac{1}{\sqrt{H}} \frac{\partial \mathcal L}{\partial c}.
$$

The forward scan caches the post-reset $\tilde m_{t-1}$ so the backward scan
can reuse it without recomputing the recurrence.

## Implementation notes

- **Parameters per layer:** $W$ ($2H^2$) followed by an 8-element scalar
  vector $[\rho, \eta, 0, \dots, 0]$, for $2H^2 + 8$ in total. The six padding
  slots keep Muon's flat parameter offsets aligned with the allocator's
  16-byte layout. Their gradients are always zero. The order ($W$, then
  scalars, per layer) is shared by the CUDA trainer, the gradient buffers and
  the CPU weight loader.
- **Rollout vs. training:** during rollout (`mpnpre_forward`) the state is
  updated in place one step at a time (`mpnpre_step_pre`). The trainer zeros the
  state for agents whose terminal flag is set (`zero_term_state`) before the
  post-terminal observation is encoded, which matches the in-scan
  $(1-d_t)$ reset used in training.
- **Precision:** in the training scan, $m$ is rounded to `precision_t` after
  every step, so in bf16 builds the training forward reproduces exactly the
  state sequence the rollout stored.
- **Checkpoints** are network-specific: load a checkpoint with the same
  `[policy] network` it was trained with. MinGRU and MPNpre have different
  parameter counts ($3H^2$ vs. $2H^2 + 8$ per layer), so a mismatch fails the
  weight-count check.

## Full MPN

`[policy] network = mpn` (`src/networks/mpn.cu`, CPU mirror `mpn_layer` in
`src/puffercpu.c`) keeps the full plastic matrix of Aitken & Mihalas (Eq. 1):
one $M \in \mathbb{R}^{H \times H}$ per layer per agent. It has the same
parameters as MPNpre ($W = [W_a; W_g]$ and $[\rho, \eta]$, same init and the
same $\lambda$, $c$ mapping), but only $W_a$ is plastic and the Hebbian update
uses the postsynaptic activity $h_t$:

$$
\begin{aligned}
\tilde M_{t-1} &= (1 - d_t) M_{t-1} \\
a_t &= (W_a \odot (1 + \tilde M_{t-1})) x_t, \quad g_t = W_g x_t \\
h_t &= \tanh(a_t), \quad s_t = \sigma(g_t) \\
y_t &= s_t \odot h_t + (1 - s_t) \odot x_t \\
M_t &= \lambda \tilde M_{t-1} + c\, h_t x_t^{\top}
\end{aligned}
$$

Unlike MPNpre, $M$ depends on $h$, so the recurrence is nonlinear and has to
run step by step. $W x_t$ is still one time-parallel GEMM; the scan adds
$(W_a \odot \tilde M_{t-1}) x_t$ and updates $M$.

**Backward.** With $G_t = \partial \mathcal L / \partial M_t$ ($G_T = 0$) and
$\delta_t = \partial \mathcal L / \partial y_t$, for $t = T, \dots, 1$:

$$
\begin{aligned}
\tfrac{\partial \mathcal L}{\partial h_t} &= \delta_t \odot s_t + c\, G_t x_t, \qquad
\tfrac{\partial \mathcal L}{\partial a_t} = \tfrac{\partial \mathcal L}{\partial h_t} \odot (1 - h_t^2) \\
\tfrac{\partial \mathcal L}{\partial x_t} &= \delta_t \odot (1 - s_t)
  + (W_a \odot (1 + \tilde M_{t-1}))^{\top} \tfrac{\partial \mathcal L}{\partial a_t}
  + W_g^{\top} \tfrac{\partial \mathcal L}{\partial g_t} + c\, G_t^{\top} h_t \\
\tfrac{\partial \mathcal L}{\partial W_a} &\mathrel{+}= \big(\tfrac{\partial \mathcal L}{\partial a_t} x_t^{\top}\big) \odot (1 + \tilde M_{t-1}) \\
G_{t-1} &= (1 - d_t) \big(\lambda G_t + W_a \odot (\tfrac{\partial \mathcal L}{\partial a_t} x_t^{\top})\big) \\
\tfrac{\partial \mathcal L}{\partial \lambda} &\mathrel{+}= \textstyle\sum G_t \odot \tilde M_{t-1}, \qquad
\tfrac{\partial \mathcal L}{\partial c} \mathrel{+}= h_t^{\top} G_t x_t
\end{aligned}
$$

**Kernels.** Row $i$ of $M$ only needs $x_t$ and $h_{t,i}$, so one warp owns one
(agent, row), keeps that row of $M$ (forward) or $G$ (backward) in registers
for the whole horizon, and needs no cross-warp sync. The scans are templated on
$K = H/32$, so `hidden_size` must be one of 32, 64, 128, 256, 512 or 1024. The
unmodulated terms ($\partial a\, x^\top$ for $\partial W$, $W^\top \partial a$
for $\partial x$) go through the usual GEMMs; the plastic terms that sum across
rows ($\partial x$) or agents ($\partial W_a$) use f32 atomics.

**Memory.** Training caches every $\tilde M_{t-1}$:
`num_layers x minibatch_size x H^2` elements (printed at startup; e.g. 1 GB in
bf16 for 4 layers, minibatch 8192, $H = 128$). Recurrent state is
`num_layers x total_agents x H^2` per copy, and the trainer keeps several
copies (also printed). The trainer refuses state tensors over $2^{31}$
elements.

**Not yet optimized.** The cache could be replaced by checkpointing every
$K$ steps and recomputing, and the $\partial x$ atomics by a per-agent block
reduction.

## Reference

Aitken, K. and Mihalas, S. (2023). *Neural population dynamics of computing
with synaptic modulations.* eLife 12:e83035.
