AI DISCLOSURE: this file was drafted by Claude Opus from a planning discussion with Brian Luther.

# burstlab architecture

This document records the design decisions for `burstlab`, why each was made, and what each one defers on purpose.
Paper-specific ambiguities and hazards are in [`replication-notes.md`](replication-notes.md).

**Goals**, in priority order:

1. Replicate Stuck et al. 2025 (`docs/papers/stuck2025.pdf`, repo root): spiking Burstprop on rate-coded MNIST.
   Targets: 1×100 hidden layer, signed 97.2%, unsigned 96.8%, BPTT 97.7%.
2. Make it easy to try, run, and inspect experiments. Expected future directions are other tasks (IMU,
   CIFAR/ImageNet), self-supervised and JEPA-style targets, residual streams, interpretability, and techniques
   from other papers.
3. Help the people working on it build their own understanding. The code should be readable by someone who knows
   PyTorch but is new to spiking networks.

## 1. Explicit wiring: populations and connections are data

**Decision.** A network is a set of **populations** (groups of neurons sharing a model) and **connections**
between them. A connection is a plain data object with:

| Field | Values | Paper reference |
|---|---|---|
| source, target | population names | Fig. 1A |
| target compartment | `soma` \| `dendrite` | §3.3: events → soma, bursts → dendrite |
| signal | `events` \| `bursts` (signed: `bursts+` − `bursts-`) | eqs. 6–8 |
| delay | `0` (feedforward) \| `1` (feedback) | §3.9: feedback is read from the previous step |
| weight source | own \| tied-transpose \| fixed-random \| learned | §2.2, §4.2: symmetric / FA / DFA / Kolen–Pollack |
| plasticity rule | optional function | eqs. 13–14 |

Teaching signals and regularizers are **dendritic drives**: sources that inject a loss gradient into a population's
dendrites (eqs. 9–12). Later self-supervised or prediction targets become new drives, not new simulator code.

**Why.**

- *No autograd, so the wiring has to be explicit.* In ordinary PyTorch you never declare wiring. Autograd records
  the graph while `forward()` runs and derives the backward pass for you. Burstprop's backward pass is made of
  physical feedback connections that carry bursts into dendrites, so something has to state which population
  sends bursts where, through which weights, with what delay. A hand-written loop would hide that in its control
  flow. Data makes it visible and changeable.
- *It matches the paper.* Fig. 1A is this graph: axons that target somas or dendrites and carry events or bursts.
- *It's how local-learning simulators are built.* Simulators centred on plasticity describe networks as groups
  plus connections, with learning rules attached to the connections:
  - NEST: `Create` / `Connect`.
  - Brian2: `NeuronGroup` / `Synapses`, with plasticity written in the synapse's `on_pre` / `on_post` code.
  - PyNN, a cross-simulator API also used for SpiNNaker: `Population` / `Projection`.
  - Auryn, which ran the original spiking Burstprop: see `~/dev/clones/spikingburstprop/src/`, with
    `BurstPoissonGroup`, `TransmitBurstConnection`, and `BCPConnection`.
  - BindsNET, a PyTorch STDP library: `Network.add_layer` / `add_connection`.
  - Intel Lava, for Loihi: processes whose ports are connected explicitly.

  The libraries that use implicit `nn.Module` stacks are the BPTT ones: Norse, snnTorch, SpikingJelly. Autograd
  does their backward pass, so they never need to write down wiring. See
  [`reference-simulators.md`](reference-simulators.md).
- *A route to hardware.* NIR (Neuromorphic Intermediate Representation) is the emerging format for moving models
  between simulators and chips, and it is an explicit graph too. It's already installed as a Norse dependency
  (`nir`; `norse/torch/utils/export_nir.py`). NIR covers inference only, not bursts or plasticity, but an explicit
  graph makes "export the trained forward network" a natural later step.
- *Future ideas become configuration.* DFA means changing the source of a feedback connection. A residual skip is
  one more connection. Lateral interneurons are connections within a layer.

**Costs.** There is one layer of indirection: you read a wiring list instead of a loop. The simulator is roughly
100–200 lines. These costs are kept small by the staged mitigations in §6.

## 2. Performance

**Decision.** The wiring is compiled **once** into a flat execution plan: for each population, in order, its list
of incoming connections. The per-timestep loop walks plain lists.

**Why this should cost about the same as a hand-written loop.** A per-timestep spiking simulator is limited by
Python overhead and kernel launches, not by arithmetic. Both designs run the same matmuls on the same spike
tensors. Iterating over a short static list adds a few dozen Python operations per step. `torch.compile` traces
through loops over static structure, so it unrolls them into the same graph a hand-written loop would produce.

**Rules.** Avoid data-dependent branches in the step loop, such as "skip if no spikes"; they break compilation in
either design. Batch size, device, and whether the loop is compiled matter far more than the wiring.

**Checked, not assumed.** An overhead benchmark against the reference loop is planned (§6) before the first large
sweep.

## 3. Norse's role: test oracle and BPTT engine

**Decision.** Neuron step functions are written in plain torch and **follow Norse's conventions**: a functional
`step(input, state, params, dt) -> (output, state)` with NamedTuple state and params (see
`docs/norse/overview.md`, "Writing a custom neuron in Norse style"). Norse has two jobs:

- **Test oracle.** With the dendrite silenced, our soma must match `lif_box_feed_forward_step`, and our dendrite
  must match `li_box_feed_forward_step`.
- **BPTT baseline engine** (`LIFBox`, surrogate gradients, `SequentialState`). This is what Norse is built for.

**Why not build Burstprop on top of Norse.** Norse provides nothing for two compartments, bursts, burst/event
routing, teaching signals, burst-dependent plasticity, or delays (`docs/norse/stuck2025-api-map.md`). Its
composition tools (`SequentialState`, `Lift`) assume one input stream per neuron, processed one layer at a time
over all timesteps. Burstprop needs a per-timestep loop that runs bottom-up through the layers, with feedback from
the previous step into a second compartment. The pieces Norse does provide are about 3 lines of math each, and
wrapping them brings several gotchas (`docs/norse/gotchas.md`: G2 input scaling, G4 strict threshold, G11 trace
units, G15 shared default tensors). Owning these few lines keeps us free to compile the loop or run on MPS without
working around Norse internals.

## 4. Batching is a simulator property, not a model property

**Decision.** `batch_size` and `update_every` (`step` | `sample`) are config settings.

- **Default for sweeps:** B ≈ 64–128, with ΔW averaged over the batch and applied every timestep.
- **Faithful mode:** B = 1 with per-step updates, which is the paper's protocol. Used to check that batching
  doesn't change results at small scale, and for any experiment where online learning is the point (continual
  learning, non-stationary streams such as IMU, catastrophic forgetting).

**What is and isn't biologically plausible.** The learning rule stays fully local under batching: each synapse
still uses only its own presynaptic trace, postsynaptic events, and bursts. What batching adds is B identical
copies of the network looking at B different images, with their local ΔWs **averaged across copies**. That
cross-copy averaging is the non-biological part. It's the same distinction as minibatch vs. online SGD in deep
learning. Two consequences:

- sample n+1 no longer sees the weights that sample n just changed;
- per-step updates average B noisy estimates, which noticeably lowers gradient noise (replication notes §3).
  This may change learning dynamics, and that is itself measurable.

**Why bother.** The paper's protocol is about 6×10⁸ sequential timesteps for 100 epochs (100 steps × 60,000
images × 100 epochs). Python overhead dominates, so one run takes on the order of a day. A neuromorphic chip runs
batch 1 natively in real time. Batching is how we emulate that cheaply on a CPU or GPU.

## 5. Who writes what

Claude builds the infrastructure (simulator loop, probes, configs, data and encoders) and **writes the tests
first**. The students implement the core pieces against those tests:

- the two-compartment neuron step and the burst generator (phase 1);
- the plasticity rule (phases 1–2).

Each core piece comes with a short explainer and the tests that define it. Claude reviews.

## 6. Mitigations are staged, not built upfront

The costs of explicit wiring (indirection, a newcomer learning the vocabulary) can be offset. But adding every
offsetting measure upfront would bloat the project and hide the core idea. Each is added when its trigger arrives:

| Mitigation | Add when |
|---|---|
| Tiny framework, paper vocabulary (`soma`, `dendrite`, `events`, `bursts`, `delay`); plain dataclasses and functions, no registries or metaclasses | Always. It's a style rule and costs nothing |
| Hand-written reference loop for the paper's 1-hidden-layer network, plus an exact-equivalence test (same seed → same result) against the generic simulator. It's also the entry point for new readers | Phase 2, written *before* the generic simulator, which then generalizes it |
| `net.describe()`: prints the wiring as a table | When the first non-stack topology lands (DFA, skip) |
| Overhead benchmark: generic simulator vs. reference loop | Before the first large sweep (phase 4) |
| `torch.compile`; fusing connections that share a target compartment | Only if the benchmark shows a need |
| Periodic or best-so-far checkpoints, resuming runs | When runs get long enough to need them (phase 2 spiking runs) |
| `.pt` snapshots of internal quantities (burst probabilities, update directions) | When a concrete analysis needs them |
| TensorBoard | When we want to watch runs live |

When you're tempted to add something not on this list, add a row with its trigger instead.

## 7. Testing tools: noted, not yet dependencies

- **Hypothesis (property-based testing).** Welcome where an invariant is natural. Judge each case when it comes
  up; don't go hunting for uses. Candidate invariants:
  - bursts ⊆ events;
  - `bursts+` and `bursts-` are disjoint;
  - burst probability p ∈ [0, 1];
  - low-resolution weights stay on the grid, with |accumulator| < ε/2 (§3.6 of the paper);
  - any generated wiring produces a valid execution order;
  - batched and faithful modes agree on the expected ΔW.
- **External simulators as single-neuron oracles.** Brian2 can express the two-compartment neuron with custom
  equations and `rand()` in its reset code. Gotchas:
  - pass `method='euler'` to match our integrator (Brian2 otherwise picks exact integration for linear equations);
  - compare voltage traces exactly only under identical input spikes;
  - compare burst statistics only in distribution, since the random number streams differ.

  Add as a dev dependency only when a phase needs it.

## 8. Layout

```
src/burstlab/            # this track; sibling of src/unm_nmc_burstprop/ (exploratory notebooks)
  docs/design/           # project documents: this file, replication-notes.md, reference-simulators.md
  learning/              # human learning material; not a project document or source of facts
  ...                    # code subpackages are added per phase
tests/burstlab/          # added in phase 1, mirroring tests/norse/
```

Shared, repo-level resources stay where they are: `docs/papers/`, `docs/norse/`, `tests/norse/`.

## 9. Phases

Each phase gets its own plan and review.

1. **Single two-compartment spiking neuron.** Reproduce Stuck Fig. 2: a sinusoidal dendritic potential → burst
   probabilities, burst trains, eligibility trace, ΔW, for both the signed and unsigned models. Norse oracle for
   the soma and dendrite. Sign-convention test (replication notes §1).
2. **Spiking MNIST.** Reference loop first, then the explicit-wiring simulator and the equivalence test. 1×100
   hidden layer, symmetric feedback. Targets 97.2% / 96.8%.
3. **BPTT baseline** with Norse `LIFBox` on the same architecture. Target 97.7%.
4. **Paper sweeps.** Depth and width (Fig. 3A), weight freezing (3B), FA/DFA (Fig. 4), low-resolution weights
   (Fig. 5), sparsity (Table 1), EMNIST/FMNIST. Run the overhead benchmark first.

A rate-model Burstprop exercise (Payeur 2021) was tried as a first phase and dropped, to keep the focus on one
model: the spiking one. It's preserved in git history at commit `01f8121`.

## 10. Applying updates: the pure rule by default

**Decision.** Plasticity rules apply their update directly: `W += ΔW`, with the learning rate inside the rule.
There's no momentum and no weight decay. This is the starting point for spiking plasticity; phase 2 decides
afresh how spiking updates are applied.

**Considered and set aside: Burstprop as a gradient estimator.** That means storing the update in `.grad` and
applying it with `torch.optim`. Its advantages: an identical optimizer for the backprop comparison, standard
tooling, and direct `.grad` comparisons. Why we didn't take it:

- It presumes what we're testing (that the update approximates a gradient).
- It hides the update rule inside the optimizer.
- It invites optimizers that break locality.

**Momentum changes the learning dynamics.** It's local: each synapse keeps a decaying trace of its own recent
updates. But that trace is an extra per-synapse state variable with its own time constant, and it isn't in Stuck's
rule. It also averages noisy updates over time, so part of any success would be due to it. It's available only as
an explicit, labelled experiment, never as a default.

**Locality rule for optimizers.** Any optimizer used with a local rule must update each synapse using only that
synapse's own history. SGD and momentum qualify; Adam does too, though it's biologically odd. Gradient clipping by
global norm, per-layer normalization (LARS-style), and batch-norm statistics do not.

## 11. Run outputs

Training runs write one directory each:

```
runs/<name>/<timestamp>/
  config.yaml      # the exact config, for reproducing the run
  metrics.jsonl    # one JSON object per line (e.g. per epoch): losses, accuracies, per-layer alignment diagnostics
  final.pt         # state_dict of the model at the end (load with torch.load(..., weights_only=True))
```

Why: JSONL is appended line by line, so a crashed run keeps its history. It's readable in any editor and loads with
`pd.read_json(path, lines=True)`. `.pt` holds tensors and state, not scalar histories. Saving plain `state_dict`s
(not pickled objects) keeps loading safe. More outputs are staged in §6. `runs/` and `data/` (dataset downloads) are
git-ignored. This layout starts with the phase 1–2 infrastructure.

## 12. Sign convention: bursts potentiate

**Decision.** A positive dendritic potential and a positive burst mean *"increase my rate"*. Bursts carry the
negative loss gradient (the error, target − rate), and a burst paired with presynaptic activity strengthens the
synapse:

$$
T_i = +2\gamma_T\,(\hat{E}_i - \bar{E}_i), \qquad R_i = +2\gamma_R\,(\hat{E} - \bar{E}_i), \qquad
\Delta w_{ij} = +\eta\,\tilde{E}_j\,(\hat{B}^+_i - \hat{B}^-_i)
$$

and, for the unsigned model, $\Delta w_{ij} = +\eta\,\tilde{E}_j\,(\hat{B}_i - B_P S_i)$.

**Why.** Stuck's printed equations (9, 10, 13, 14) put the minus sign in the plasticity rule instead: bursts carry
$+\partial L/\partial \bar{E}$ and a positive burst weakens the synapse. Both train identically. We chose
Payeur's convention because it matches the biology (burst-dependent LTP), Stuck's own prose under eq. 14, and the
easier mental model. **Cost:** T, R, and the rule each differ from Stuck's printed equations by one minus sign.
See replication notes §1.

## 13. Ideas we could try (not planned)

These are ideas to remember, not commitments. Each needs its own discussion before anyone builds it.

- **Expected-value mode for the spiking simulator.** The burst generator would emit the burst probability p
  instead of sampling bursts, and inputs could optionally be smooth rates instead of Poisson spikes. It would use
  the same code, made deterministic, which could help separate spiking noise from systematic bias when spiking
  results need explaining.
