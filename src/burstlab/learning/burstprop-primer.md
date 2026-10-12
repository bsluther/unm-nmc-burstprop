AI DISCLOSURE: this file was drafted by Claude Opus from a planning discussion with Brian Luther.

> **Note for agents:** this is a learning document for humans. It simplifies and is edited for teaching. Don't treat
> it as a source of facts, specs, or decisions. Use the papers (`docs/papers/`), `src/burstlab/docs/design/` (especially
> `replication-notes.md`), and the reference code instead.

# Burstprop primer: two-compartment neurons and local learning

This primer is for readers who know PyTorch and backprop but are new to spiking networks and neuroscience. It covers
the two ideas Burstprop rests on, then lists the places where replicating Stuck et al. 2025 needs care. Equation
numbers refer to `docs/papers/stuck2025.pdf` ("S-eq.") and `docs/papers/payeur2021.pdf` ("P-eq.", Methods
numbering). Both are at the repo root.

## 1. Two-compartment neurons

### The biology, briefly

A layer-5 pyramidal neuron in cortex has two input regions that sit far apart:

- **Basal and perisomatic dendrites**, near the cell body (soma). They receive **bottom-up** input from lower areas.
- The **apical tuft**, a long way up in cortical layer 1. It receives **top-down** input from higher areas.

The tuft is far enough from the soma to be partly cut off electrically. Apical input on its own barely moves the
soma's voltage. But when the soma fires, the spike also spreads backward from the soma into the dendrites (a
*backpropagating action potential*, unrelated to ML backprop), physically up the apical trunk toward the tuft. If
top-down input arrives at the tuft at about the same time, the two together can trigger a calcium plateau in the
dendrite. The neuron then fires a **burst**, several spikes at about 100 Hz, instead of a single spike (Larkum, Zhu &
Sakmann 1999, "BAC firing": backpropagating-AP-activated Ca²⁺ spike).

So top-down input mostly changes the *pattern* of firing, not its *rate*.

### Events, bursts, and multiplexing

Following Payeur 2021 and Naud & Sprekeler 2018, define an **event** as either a single spike or a whole burst, so a
burst counts as one event. Each event carries a one-bit label: burst or not. A population of neurons then
transmits **two signals at once**:

| Quantity | Symbol | Controlled by | Carries |
|---|---|---|---|
| event rate | e | somatic input | the feedforward / inference signal |
| burst probability | p | dendritic input | the feedback / error signal |
| burst rate | b = p·e | both | what the feedback pathway transmits |

This is called **burst multiplexing**. The receiving end can separate the two signals through **short-term
plasticity (STP)**. A *depressing* synapse responds strongly to the first spike and weakly to the rest, so it passes
roughly the event rate. A *facilitating* synapse responds more strongly to later spikes, so it passes mostly bursts.
Payeur's full spiking model uses exactly this (Methods, "Short-term plasticity"). **Stuck abstracts it away**:
events are routed directly to somas up the hierarchy, and bursts directly to dendrites down the hierarchy (S-§3.3,
Fig. 1A).

### The analogy that helps most

> The dendrite is the neuron's `.grad` buffer, physically stored at the neuron. Bursts are how the neuron sends its
> gradient *down* on the same axon that sends its activation *up*.

Why not use two separate neurons, one forward and one backward? Because the error has to be **at the same cell
whose input synapses need updating**. A synapse can only use information present where it is. Putting the forward
activity (soma) and the error (dendrite) in one cell is what lets the learning rule stay local (§2).

### Stuck's simplified neuron (S-§3.1)

| Part | Dynamics | Notes |
|---|---|---|
| Soma | dimensionless LIF, τ_s = 10 ms, threshold → event, then reset (S-eq. 1–3) | `LIFBox` in Norse terms (`docs/norse/stuck2025-api-map.md`) |
| Dendrite | leaky integrator, τ_d = 10 ms, **no threshold** (S-eq. 5) | |
| Unsigned bursting | on each event, burst with probability σ(Ṽ_d) | p = 0.5 when Ṽ_d = 0 |
| Signed bursting | on each event, burst with probability tanh(\|Ṽ_d\|); sign(Ṽ_d) picks a positive (long) or negative (short) burst; the dendrite resets on a burst | p = 0 when Ṽ_d = 0 |
| Eligibility trace | exponential moving average of the event train, τ = 10 ms (S-eq. 4) | |

**Key point: the coupling is one-way.** The dendrite never changes somatic voltage. It only *labels* events the
soma has already decided to fire, and forward synapses count every event as 1 whether or not it's a burst (S-eq. 6).
So the learning signal **cannot disturb the inference pass**. That's why inference and learning can run in the same
pass ("single-phase").

Payeur's full Auryn neuron is different: there the dendrite feeds current into the soma (g_s·f(V_d)) and the soma's
spike backpropagates into the dendrite (P-eq. 5–6). Stuck drops both directions of coupling.

## 2. Local learning

### What "local" means

- **Spatially local**: a synapse changes using only its presynaptic neuron, its postsynaptic neuron, its own state,
  and possibly a global signal shared by all synapses (S-§1).
- **Temporally local**: it changes online as things happen, without storing history and without separate phases.

Backprop breaks both:

- It needs **Wᵀ** to send errors backward (the *weight-transport problem*).
- It needs error derivatives that no single synapse can see.
- It needs a separate backward pass with the forward activations held fixed.
- BPTT also stores the entire activity history.

### Burstprop as a three-factor rule

Payeur's rule (P-eq. 1, 4):

```
dw_ij/dt = η · [B_i(t) − P_i(t)·E_i(t)] · Ẽ_j(t)
            postsynaptic burst vs. baseline   presynaptic trace
```

- A postsynaptic **burst** potentiates the synapse (by 1 − P).
- A **single spike** depresses it (by −P).
- P is a slow moving average of the neuron's own burst fraction (τ ≈ 1–10 s), so the two balance out when nothing
  has changed.

Averaged over a population, this becomes (P-eq. 2, 9):

```
ΔW = η · (p − p̄) ⊙ e_post · e_preᵀ        compare backprop:  ΔW = −η · δ_post · a_preᵀ
```

So the deviation of the burst rate from its baseline, (p − p̄)·e, plays the role of backprop's δ. The presynaptic
trace plays the role of the input activation.

**Why the third factor matters.** Many biological learning rules have three factors: pre, post, and a modulator.
In reward-modulated STDP, the modulator is a **single global scalar**, like dopamine. With one scalar shared by every
neuron, the gradient estimate gets noisier as the network grows. Node perturbation is the standard example, and it's
implemented as a baseline in `~/dev/clones/Burstprop/layers.py` (`*NP` classes). Burstprop's third factor is
**specific to each neuron**: it's that neuron's own bursting, steered by its own dendrite. That's what lets it scale
to multilayer credit assignment.

### Where backprop's f′ goes

Backprop's δ includes the activation derivative f′(v). In Payeur's rate model, the hidden dendritic potential is
u = h(e) ⊙ (Y·b) with h(e) = f′(v)/e (P-eq. 7). Multiplying by e (since b = p·e) and dividing by e (inside h) leaves
exactly f′. For a sigmoid, h(e) = 1 − e, which is the `b_input*(1 - self.e)` in `layers.py`.

In Stuck's spiking model there's no h. f′ shows up only as **gating by postsynaptic events**: plasticity happens on
postsynaptic events, so it scales with e_post (S-§3.5). That's an approximation, and we plan to measure it.

### Getting the baseline without a second pass

The rule needs to know "what would the burst probability be without the teacher?" Each paper answers differently:

| Paper | Baseline |
|---|---|
| Payeur 2021 | slow moving average P̄. In the XOR task, plasticity is gated (M(t)) so it only acts after P̄ has settled |
| BurstCCN (Greedy et al.) | a second feedback pathway carrying spikes, which presets the baseline |
| Stuck 2025, unsigned | fixed analytically: B_P = σ(0) = 0.5 (S-eq. 14) |
| Stuck 2025, signed | none needed: no error → no bursts → no plasticity (S-eq. 13) |

### Weight transport

How does the feedback weight B relate to the forward weight W?

- **Symmetric**: B = Wᵀ. This is the "cheat": it's what backprop assumes.
- **Feedback alignment (FA)**: B is random and fixed. W learns to align with it.
- **Direct FA (DFA)**: random feedback goes from the output straight to every layer.
- **Kolen–Pollack**: B and W get the same updates plus weight decay, so they converge (Payeur Methods).

Stuck tests the first three (S-§4.2). In `burstlab`, this is the "weight source" of a connection (architecture §1).

## 3. Replication hazards in Stuck 2025

These are places where the paper is ambiguous or where we expect trouble. Each will become a test or a measurement.

1. **The paper's signs conflict.** S-eq. 9 makes the output dendrite carry **+∂L/∂Ē** (the teaching signal is
   −2γ(Ê − Ē)). S-eqs. 13–14 have a leading minus. Those agree with each other: if a rate is below target, the
   dendrite goes negative, the neuron emits negative bursts (signed) or fewer bursts (unsigned), and the weights
   grow. But the prose under S-eq. 14 says bursts cause *positive* weight changes. That's Payeur's convention, and
   it contradicts S-eq. 14 as written. Only the product of signs along the chain matters. We pick one convention
   and pin it with a test: *output rate below target ⇒ its incoming weights grow*.
2. **The unsigned model leaks a baseline.** With no teacher, output neurons still burst at p = 0.5. So hidden
   dendrites receive Σ b_ij · 0.5 · e_j ≠ 0 (S-eq. 7), a fake "gradient" built only from forward activity. Nothing
   subtracts it at the dendrite. The rate code does subtract it (`delta = -(b_t - b)`). The signed model sends
   nothing when the error is zero. **Hypothesis to test:** this explains why unsigned accuracy collapses with depth
   (64% at 4 hidden layers vs. 95.9% signed, S-§4.1).
3. **The gradient is very noisy.** At a hidden rate around 20 Hz (S-§3.4.2) and 100 ms per image, each hidden
   neuron fires about 2 events per image. So it gets about 2 one-bit samples of its error per example, and fewer
   bursts than that. SGD tolerates noise, but our diagnostics should compare the spiking ΔW against its expected
   value and against the true gradient.
4. **Credit assignment is spatial only.** Traces last about 10 ms, so Burstprop has no mechanism for assigning
   credit across time. The paper says so itself (S-§4.3). This is a research question for temporal tasks like IMU,
   not a detail.
5. **How a spike becomes voltage is unspecified.** S-eq. 6 writes the input as Σ w·S(t) with Dirac spikes, but
   doesn't say how that's discretized with dt = 1 ms. This determines how weights, initial scales, and γ values
   compare with the paper. It's already marked "open" in `docs/norse/stuck2025-api-map.md`.

## 4. Questions to check your understanding

- Why can a slow moving average of p (Payeur) stand in for a separate forward pass without the teacher? What does
  it assume about how fast the input changes?
- In the signed model the dendrite resets after each burst. What does that do to how many bursts a sustained error
  produces, and to the gradient estimate?
- With fixed random feedback (FA), why must the output weights learn *before* the hidden layers can learn usefully?
  (Stuck's weight-freezing experiment, S-Fig. 4B, tests exactly this.)
- Burstprop gates plasticity by postsynaptic events instead of f′(v). For an LIF neuron, when would these differ
  most?

## 5. Sources

This document was written from:

- `docs/papers/stuck2025.pdf` and `docs/papers/payeur2021.pdf` (main text and Methods), the source of the
  equations, parameters, and results cited above.
- `~/dev/clones/Burstprop/layers.py` (Payeur's rate model) and `~/dev/clones/spikingburstprop/src/BurstPoissonGroup.cpp`
  (Payeur's simplified spiking neuron).
- `docs/norse/stuck2025-api-map.md`.
- Claude's general background knowledge, for the biology (BAC firing, short-term plasticity) and the ML framing
  (three-factor rules, node perturbation). Larkum et al. 1999 and Naud & Sprekeler 2018 are cited as the papers
  above cite them; they weren't read directly.
- The BurstCCN row in §2 comes from Stuck's description of Greedy et al. 2022 (S-§2.1.3).
