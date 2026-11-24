# MPN: Multi-Plasticity Network layer

This document describes the recurrent layer that replaces MinGRU on the `mpn`
branch. It is the **MPNpre** variant of the multi-plasticity network from
Aitken & Mihalas (2023), *eLife* 12:e83035 (Eq. 1 and Eq. 13), with PufferLib's
gated highway connection added on top.

Implementation: `src/algo.cu` (`mpn_*` kernels and `mpn_forward`,
`mpn_forward_train`, `mpn_backward`), CPU inference mirror in
`src/puffercpu.c` (`mpn`), float64 reference in `tests/test_mpn.py`.

## Notation

| Symbol | Shape | Meaning |
|---|---|---|
| $H$ | — | hidden width (input and output width of every layer) |
| $x_t$ | $\mathbb{R}^H$ | layer input at step $t$ |
| $m_t$ | $\mathbb{R}^H$ | plastic state (the recurrent carry) after step $t$ |
| $W = \begin{bmatrix} W_a \\ W_g \end{bmatrix}$ | $\mathbb{R}^{2H \times H}$ | slow weights (no bias) |
| $\rho, \eta$ | scalars | learned plasticity parameters, one pair per layer |
| $d_t$ | $\{0,1\}$ | terminal flag delivered together with $x_t$ |
| $\sigma$ | — | logistic sigmoid |
| $\odot$ | — | elementwise product |

## Forward pass

Per layer, the two scalar parameters are mapped to a decay and a learning rate:

$$
\lambda = \lambda_{\max}\,\sigma(\rho), \qquad \lambda_{\max} = 0.95,
\qquad c = \frac{\eta}{\sqrt{H}} .
$$

So $\lambda \in (0, 0.95)$ always, which keeps the state recurrence strictly
contracting.

Given the carried state $m_{t-1}$, each step is:

$$
\begin{aligned}
\tilde m_{t-1} &= (1 - d_t)\, m_{t-1} && \text{(reset at episode boundary)} \\
u_t &= x_t \odot (1 + \tilde m_{t-1}) && \text{(presynaptic modulation)} \\
a_t &= W_a u_t, \quad g_t = W_g u_t && \text{(one } 2H \times H \text{ GEMM)} \\
s_t &= \sigma(g_t) \\
y_t &= s_t \odot \tanh(a_t) + (1 - s_t) \odot x_t && \text{(gated highway output)} \\
m_t &= \lambda\, \tilde m_{t-1} + c\, x_t && \text{(plasticity update)}
\end{aligned}
$$

$y_t$ is the layer output and becomes the next layer's $x_t$. Layers are
stacked $L$ deep, each with its own $W$, $\rho$, $\eta$ and state $m$; the
carried state tensor is $(L, A, H)$ for $A$ agents, the same shape MinGRU used.

### Relation to the MPN of Aitken & Mihalas

The general MPN modulates a weight matrix with a fast, plastic matrix $M_t$:

$$
h_t = \phi\big((W \odot (1 + M_{t-1}))\, x_t\big), \qquad
M_t = \lambda M_{t-1} + \eta\, (\text{post}) \, x_t^{\top}.
$$

In the **pre**-synaptic variant the Hebbian update depends only on the
presynaptic activity, $M_t = \lambda M_{t-1} + c\,\mathbf{1} x_t^\top$, so (from
a zero or row-constant start) every row of $M$ is the same vector $m_t^\top$.
The modulated product then factors:

$$
(W \odot (1 + \mathbf{1} m^\top))\, x = W\,\big(x \odot (1 + m)\big) = W u .
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
  $\|m\|_\infty \le \max(\|m_0\|_\infty,\ c\,\|x\|_\infty / (1-\lambda))$.
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
| $\rho$ | $3.0$ | $\lambda = 0.95\,\sigma(3) \approx 0.905$ (time constant ≈ 10.5 steps) |
| $\eta$ | $1.0$ | $c = 1/\sqrt{H}$ |

## Training: truncated BPTT over the horizon

`mpn_forward_train` processes a segment of $T$ steps (the horizon) starting
from the stored state $m_0$. Gradients do not flow into $m_0$, and the final
state $m_T$ receives no gradient (truncated BPTT). Let
$\delta_t = \partial \mathcal L / \partial y_t$ and $h_t = \tanh(a_t)$.

**Output gate (time-parallel, `mpn_post_backward`):**

$$
\frac{\partial \mathcal L}{\partial a_t} = \delta_t \odot s_t \odot (1 - h_t^2), \qquad
\frac{\partial \mathcal L}{\partial g_t} = \delta_t \odot (h_t - x_t) \odot s_t (1 - s_t), \qquad
\left.\frac{\partial \mathcal L}{\partial x_t}\right|_{\text{highway}} = \delta_t \odot (1 - s_t).
$$

**Weights (time-parallel GEMMs):** with $e_t = [\partial_{a_t}; \partial_{g_t}] \in \mathbb{R}^{2H}$,

$$
\frac{\partial \mathcal L}{\partial u_t} = W^\top e_t, \qquad
\frac{\partial \mathcal L}{\partial W} = \sum_{b,t} e_t\, u_t^\top .
$$

**State adjoint (reverse scan, `mpn_scan_backward`):** define
$\gamma_t = \partial \mathcal L / \partial m_t$. Then $\gamma_T = 0$ and, for
$t = T, \dots, 1$,

$$
\gamma_{t-1} = (1 - d_t)\,\big(\lambda\, \gamma_t + \tfrac{\partial \mathcal L}{\partial u_t} \odot x_t\big).
$$

The $(1 - d_t)$ factor cuts the gradient at a reset: $m_{t-1}$ was replaced
by zero, so nothing flows to it.

**Input gradient:**

$$
\frac{\partial \mathcal L}{\partial x_t}
= \delta_t \odot (1 - s_t)
+ \frac{\partial \mathcal L}{\partial u_t} \odot (1 + \tilde m_{t-1})
+ c\, \gamma_t .
$$

**Plasticity scalars** (summed over batch, time and features, then chained
through the reparameterization in `mpn_scalar_grad`):

$$
\frac{\partial \mathcal L}{\partial \lambda} = \sum_{b,t,j} \gamma_{t,j}\, \tilde m_{t-1,j}, \qquad
\frac{\partial \mathcal L}{\partial c} = \sum_{b,t,j} \gamma_{t,j}\, x_{t,j},
$$

$$
\frac{\partial \mathcal L}{\partial \rho} = \frac{\partial \mathcal L}{\partial \lambda}\,
\lambda_{\max}\,\sigma(\rho)\,(1 - \sigma(\rho)), \qquad
\frac{\partial \mathcal L}{\partial \eta} = \frac{1}{\sqrt{H}}\,\frac{\partial \mathcal L}{\partial c}.
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
- **Rollout vs. training:** during rollout (`mpn_forward`) the state is
  updated in place one step at a time (`mpn_step_pre`). The trainer zeros the
  state for agents whose terminal flag is set (`zero_term_state`) before the
  post-terminal observation is encoded, which matches the in-scan
  $(1-d_t)$ reset used in training.
- **Precision:** in the training scan, $m$ is rounded to `precision_t` after
  every step, so in bf16 builds the training forward reproduces exactly the
  state sequence the rollout stored.
- **Checkpoints** from the MinGRU network are not compatible.

## Reference

Aitken, K. and Mihalas, S. (2023). *Neural population dynamics of computing
with synaptic modulations.* eLife 12:e83035.
