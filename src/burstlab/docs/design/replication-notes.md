AI DISCLOSURE: this file was drafted by Claude Opus from a reading of Stuck et al. 2025 and Payeur et al. 2021.

# Stuck 2025 replication notes

These are working notes on where `docs/papers/stuck2025.pdf` (repo root) is ambiguous, inconsistent, or likely to
cause trouble. Each item is tagged:

- **[paper]**: stated in the paper, with a section or equation reference. Check against the PDF before relying on
  it.
- **[derived]**: our own reasoning from the paper's equations. The paper doesn't state it.
- **[hypothesis]**: an untested claim. It needs a test or measurement before anyone treats it as true.

If an item turns out to be wrong, fix or delete it. Don't leave stale claims here.

## Targets [paper]

From S-§4.1–4.2 (10 initializations each, $\pm$ is the standard error of the mean). Burstprop was trained for 100
epochs, BPTT for 200.

| Network | Signed | Unsigned | BPTT |
|---|---|---|---|
| 1 hidden $\times$ 100, symmetric feedback | $97.2 \pm 0.1\%$ | $96.8 \pm 0.1\%$ | $97.7 \pm 0.1\%$ |
| 4 hidden $\times$ 100, symmetric feedback | $95.9 \pm 0.1\%$ | $64 \pm 1\%$ | $84 \pm 1\%$ |
| 4 hidden $\times$ 100, FA | $95.6 \pm 0.1\%$ | $85 \pm 1\%$ | n/a |

## 1. The sign conventions conflict

- **[paper]** Eq. 9: the teaching signal is
  $T_i = \gamma_T \, \partial \varepsilon_T / \partial \bar{E} = -2\gamma_T (\hat{E}_i - \bar{E}_i)$. So the output
  dendrite carries $+\partial L / \partial \bar{E}$.
- **[paper]** Eq. 13 (signed): $dw_{ij}/dt = -\eta \, \tilde{E}_j (B^+_i - B^-_i)$. Eq. 14 (unsigned):
  $dw_{ij}/dt = -\eta \, \tilde{E}_j (B_i - B_P S_i)$.
- **[paper]** The prose under eq. 14 says: for the unsigned model, negative weight changes happen on postsynaptic
  spikes and positive changes on postsynaptic bursts. That's the opposite of eq. 14 as written, and matches
  Payeur's convention (P-eq. 1, no leading minus).
- **[paper]** Fig. 2F shows positive weight changes while $\tilde{V}_d > 0$ (Fig. 2A) and negative ones while
  $\tilde{V}_d < 0$, in both models.
- **[derived]** So Fig. 2 was made with bursts potentiating: consistent with the prose and with our decision below,
  not with eqs. 13–14 as printed.
- **[derived]** Eqs. 9, 13, and 14 are consistent with each other and implement gradient descent. If a rate is below
  target, $T < 0$, so $\tilde{V}_d < 0$, which gives negative bursts (signed) or fewer bursts (unsigned), and the
  weights grow. Only the product of signs along the chain matters.
- **Decision:** we use Payeur's convention (bursts potentiate). $T$, $R$, and the rule each carry the opposite sign
  from Stuck's printed eqs. 9, 10, 13, and 14 (architecture §12). A test pins it: *a positive burst with a positive
  presynaptic trace strengthens the synapse*, and later, *an output rate below target $\Rightarrow$ its incoming
  weights grow*.

## 2. The unsigned model leaks a baseline into hidden dendrites

- **[paper]** The unsigned burst probability is $\sigma(\tilde{V}_d)$, which is 0.5 at $\tilde{V}_d = 0$ (§3.1.4,
  §3.5). Hidden dendritic input is $I_{FB} + R_i$, with $I_{FB} = \sum_j b_{ij} \hat{B}_j$ (eqs. 7, 12). There's no
  baseline subtraction at the dendrite. The plasticity rule subtracts $B_P = 0.5$ (eq. 14), but the dendritic input
  does not.
- **[derived]** With zero output error, output neurons still burst at $p = 0.5$. So hidden dendrites receive about
  $\sum_j b_{ij} \cdot 0.5 \cdot e_j \neq 0$, a signal that depends only on forward activity. The signed model emits
  no bursts at zero error. Payeur's rate code subtracts the no-teacher burst rate explicitly
  (`~/dev/clones/Burstprop/layers.py`, `delta = -(self.b_t - self.b)`).
- **[hypothesis]** This bias contributes to the unsigned model's drop in accuracy with depth (64% at 4 layers).
  Test it by measuring the hidden-layer $\Delta W$ against the true gradient, with and without a dendritic baseline
  subtraction.

## 3. Noisy per-example error signal

- **[paper]** The hidden target rate is 20 Hz (eq. 10). Samples last 100 ms (§3.8).
- **[derived]** At about 20 Hz, a hidden neuron fires about 2 events per sample. That's about 2 Bernoulli draws of
  its burst label per example, and fewer still for the signed model, since $\tanh(|\tilde{V}_d|)$ is small for small
  errors. Actual rates during training are unknown until measured.
- **Action:** diagnostics should compare the spiking $\Delta W$ with its expected value and with the true gradient.

## 4. No temporal credit assignment

- **[paper]** §4.3: the authors don't expect temporal tasks to suit the algorithm, since those typically need BPTT.
  Eligibility traces use $\tau = 10$ ms (§3.1.3).
- Relevant to future temporal tasks (IMU). This is a research question, not something to fix in the replication.

## 5. Discretization is unspecified

- **[paper]** Eq. 6 writes $I_s = \sum_j w_{ij} S_j(t)$ with Dirac spike trains. §3.9 specifies forward Euler at
  $dt = 1$ ms but not how a Dirac spike maps to a per-step voltage change. Also marked "open" in
  `docs/norse/stuck2025-api-map.md`.
- **[paper]** The plasticity rules are written as $dw/dt$ (eqs. 13–14) with $\eta = 10^{-4}$. Whether $\eta$ is
  applied per step or multiplied by $dt$ is not stated.
- **Decision (traces):** a spike counts as a pulse of height $1/dt$, so traces ($\tilde{E}$, $\bar{E}$) are in Hz
  and a steady $r$ Hz neuron's trace averages $r$ (`traces.trace_step`). This matches the paper's targets in Hz
  (§3.4). The $\eta$ question is still open; Fig. 2 bears on it (next items).
- **[paper]** Fig. 2E: the eligibility trace jumps by about 1 per presynaptic event (peaks $\approx$ 1–2.5 for
  $\sim$50 Hz input) and decays with $\tau \approx 10$ ms. Fig. 2F: individual weight changes are
  $\approx (2\text{–}9) \times 10^{-5}$.
- **[derived]** That fits a trace that jumps by 1 per event (our Hz trace $\times\,\tau$) and an update applied once
  per event: $\Delta w \approx \eta \times \text{trace} \times \text{(burst term)}$
  $\approx 10^{-4} \times 1 \times 0.5 = 5 \times 10^{-5}$. Multiplying by $dt$ as well would give $\sim 10^{-8}$.
  With our Hz trace, the same updates need $\eta \approx 10^{-4} \times \tau = 10^{-6}$.
- **[hypothesis]** The MNIST experiments use the same trace scale and $\eta$ convention as Fig. 2. Check against
  training stability once the network runs; the figure may have been made separately.
- **Impact:** both choices set the scale at which weights, initialization, $\gamma_T$, $\gamma_R$, and $\eta$
  compare with the paper. Record each choice where it's made.

## 6. Other specification gaps [paper]

- **Weight init (§3.7):** the text says "standard deviation proportional to the square root of the number of
  presynaptic neurons". Taken literally that's $\propto \sqrt{n}$. The cited convention is probably
  $\propto 1/\sqrt{n}$. Feedback weights are scaled by (number of higher-order neurons / number of lower-order
  neurons). FA/DFA matrices must start within $90^\circ$ of the forward weights.
- **Input normalization (§3.8):** each image is normalized to a constant average intensity, but the constant isn't
  given. Pixels are then mapped linearly to 20–200 Hz Poisson rates.
- **Rate units (§3.4):** targets are in Hz (200 / 20). $\gamma_T = 10^{-3}$ and $\gamma_R = 10^{-5}$ only make
  sense once the units of the rate estimate $\bar{E}$ are fixed (see `docs/norse/gotchas.md` G11).
- **Notation:** eq. 4 defines the eligibility trace as $\bar{E}_j$, but eqs. 13–14 write $\tilde{E}_j$. §3.4.1 says
  the rate estimate for the teaching signal uses the same filter. Treat them as the same quantity unless the paper
  says otherwise.
- **Signed-model dendritic reset (§3.1.4):** resets "to the baseline", presumably $\tilde{V}_d = 0$.
- **Biases:** not mentioned. Assume none.
- **Testing (§3.9.2):** no dendritic input, plasticity off, and the prediction is the output neuron with the most
  activity over the sample.
- **BPTT baseline (§3.10):** a "sigmoid" surrogate and lr $2 \times 10^{-4}$. The optimizer and loss aren't
  specified. Norse's `"super"` surrogate is a fast sigmoid, not a logistic sigmoid (`docs/norse/gotchas.md` G13).
