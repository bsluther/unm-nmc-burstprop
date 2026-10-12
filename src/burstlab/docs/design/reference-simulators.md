AI DISCLOSURE: this file was drafted by Claude Opus from a planning discussion with Brian Luther.

# Reference simulators

These are established simulators worth reading while we build `burstlab`. They're design references for explicit
wiring and plasticity (see [`architecture.md`](architecture.md) §1), and some are candidate **test oracles** for
single-neuron behaviour (§7). None of them is a dependency. Clone them locally under `~/dev/clones/` so their
source can be read during implementation, and add a clone to the session's working directories when it's in use.

## Already cloned

| Path | What it is | Read it for |
|---|---|---|
| `~/dev/clones/Burstprop` | Payeur et al. 2021 rate model (PyTorch) | `layers.py`: `HiddenLayer.backward` computes the burst probability with and without the teacher (`p`, `p_t`), and `delta = -(b_t - b)`. The `*NP` classes are a node-perturbation baseline (a global third factor) |
| `~/dev/clones/spikingburstprop` | Payeur et al. 2021 spiking model (Auryn, C++) | `src/BurstPoissonGroup.cpp`: the simplified neuron, a Bernoulli burst per somatic event with p = sigmoid(V_d). `src/*Connection.cpp`: event and burst transmission and burst-dependent plasticity as connection classes |

## To clone

| Repo | Priority | Read it for |
|---|---|---|
| [`brian-team/brian2`](https://github.com/brian-team/brian2) | high | Groups and connections (`NeuronGroup`, `Synapses`). Plasticity in `on_pre`/`on_post` code. Candidate oracle: it can express the two-compartment neuron with custom equations and `rand()` in reset code. Use `method='euler'` to match our integrator |
| [`BindsNET/bindsnet`](https://github.com/BindsNET/bindsnet) | high | The closest structural analog: PyTorch, explicit `Network.add_layer`/`add_connection`, learning rules attached to connections |
| [`apayeur/auryn`](https://github.com/apayeur/auryn) | high | The Auryn fork that spikingburstprop builds on: the base group and connection classes behind the original spiking Burstprop |
| [`lava-nc/lava`](https://github.com/lava-nc/lava) | optional | Intel's Loihi framework: processes connected through ports, and an API for learning rules on hardware |
| [`NeuralEnsemble/PyNN`](https://github.com/NeuralEnsemble/PyNN) | optional | `Population`/`Projection`, the cross-simulator vocabulary standard (NEST, NEURON, SpiNNaker) |
| [`neuromorphs/NIR`](https://github.com/neuromorphs/NIR) | optional | The graph exchange format between simulators and chips. Already installed as a Norse dependency (`nir`), so its source is in `.venv` too |

To clone the high-priority set:

```bash
cd ~/dev/clones && git clone https://github.com/brian-team/brian2 && git clone https://github.com/BindsNET/bindsnet && git clone https://github.com/apayeur/auryn
```

## Ground rules

- Reading a simulator's source for design ideas is free. *Depending* on one (e.g. Brian2 as a dev-group test
  oracle) is a decision for the phase that needs it (architecture §6–7).
- When using one as an oracle, write down which of its defaults differ from ours. Integration method, threshold
  comparison (`>` vs `≥`), and how input is scaled by `dt` are the usual suspects (compare
  `docs/norse/gotchas.md`).
