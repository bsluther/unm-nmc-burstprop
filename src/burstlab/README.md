AI DISCLOSURE: this file was drafted by Claude Opus from a planning discussion with Brian Luther.

# burstlab

This track builds burst-dependent local learning in spiking neural networks. The first goal is to replicate
Stuck et al. 2025 (`docs/papers/stuck2025.pdf`): spiking Burstprop on rate-coded MNIST. The second is to make
further experiments easy to try, run, and inspect.

It's one track among several in this repo. `src/unm_nmc_burstprop/` holds the exploratory notebooks and is separate
from this package.

## Status

Planning is done, and the design docs are written. There's no code yet. Phase 1 (rate-model Burstprop) is next.

## Reading order

1. [`docs/design/architecture.md`](docs/design/architecture.md): design decisions, their rationale, and the phases.
2. [`docs/design/replication-notes.md`](docs/design/replication-notes.md): ambiguities and hazards in Stuck 2025.
3. [`docs/design/reference-simulators.md`](docs/design/reference-simulators.md): established simulators to read
   alongside the code.
4. `docs/norse/` (repo root): Norse API notes and gotchas, for the test oracle and the BPTT baseline.

## Phases

1. Rate-model Burstprop (Payeur 2021), checked against autograd gradients
2. Single two-compartment spiking neuron (Stuck Fig. 2)
3. Spiking MNIST: hand-written reference loop, then the explicit-wiring simulator
4. BPTT baseline (Norse)
5. Paper sweeps: depth/width, weight freezing, FA/DFA, low-resolution weights, sparsity, EMNIST/FMNIST

Details are in architecture §9. Each phase gets its own plan.

## Learning material

[`learning/`](learning/) holds explanatory notes for people new to the field, such as
[`burstprop-primer.md`](learning/burstprop-primer.md). They simplify for teaching and are not project documents.
For facts, use the papers and `docs/design/`.
