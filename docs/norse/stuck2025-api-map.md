# Stuck et al. 2025 → Norse API map

Paper: *A burst-dependent algorithm for neuromorphic on-chip learning of spiking neural networks*,
Neuromorph. Comput. Eng. 5 014010 (`docs/papers/stuck2025.pdf`). Equation and section numbers below refer to it.

This file lists which Norse pieces correspond to each part of the paper's model, how their semantics differ, and
what Norse lacks. It records **no** design decisions. Rows marked "open" are choices the project still has to make.

## Summary

Norse covers the **single-compartment dynamics** (LIF soma, leaky-integrator dendrite), **Poisson input**, **trace
arithmetic**, and the **BPTT baseline**. It has **nothing** for two-compartment neurons, burst generation, burst/event
routing, teaching signals, or burst-dependent plasticity. Those are custom code, and Norse's conventions (overview.md) are
the natural way to shape them.

## Component table

| Paper component | Norse building block | Semantics and mismatches |
|---|---|---|
| **Soma**: dimensionless LIF, τ_s = 10 ms, reset (§3.1.2, eq. 1–3) | `functional/lif_box.py::lif_box_feed_forward_step`, `LIFBoxCell`, `LIFBox` (full path, G9) | `LIFBoxParameters()` defaults already equal the dimensionless paper form: `tau_mem_inv=100` (10 ms), `v_leak=0`, `v_th=1`, `v_reset=0`, `reset_value`. Use **Box**, not `LIF` (G3: `LIF` adds a 5 ms synaptic current the paper doesn't have). Input is scaled by `dt*tau_mem_inv` (G2). Threshold is strict `>` vs paper `≥` (G4). |
| Scaling of Σ w S(t) into v (eq. 6) | none | **Open.** The paper doesn't say how its Dirac spike train is discretized. Under Norse's step, a spike with weight w moves v by `0.1·w` at defaults. Decide this before comparing weights, init scales, or γ values with the paper. |
| **Dendrite**: leaky integrator, τ_d = 10 ms, no threshold (§3.1.4, eq. 5) | `functional/leaky_integrator_box.py::li_box_feed_forward_step`, `snn.LIBoxCell`, `LIBox` | Same Euler form and scaling as the soma (G2). `LIBoxParameters` is a plain NamedTuple: tensors required, no `.to()` (G6, G14). `LI`/`LICell` would add a synaptic current (G3). No reset hook. The signed model's "dendrite resets to baseline on burst" must be custom. |
| Two-compartment neuron (soma + dendrite, events + bursts) | none. **Not** `LIFMCRecurrentCell` (G16) | Build it as a custom model following overview.md ("Writing a custom neuron"). Existing parts can be called inside it: `lif_box_feed_forward_step` for the soma and `li_box_feed_forward_step` for the dendrite. |
| **Eligibility trace** Ē_j, τ = 10 ms (§3.1.3, eq. 4) | `functional/stdp.py::STDPState.decay` (the `t_pre` part), or three lines of torch | `STDPState.decay` is exactly the Euler form of eq. 4 with `a_pre=1`, but its units are **rate·dt**, not Hz (G11). It mutates in place, and the current step's spike is included before use. `ExpFilter` is **not** equivalent: it's unnormalized and includes a Linear layer (overview.md). |
| **Rate estimate** Ē_i for teaching and regularization signals (§3.4, eq. 9–10) | same as eligibility trace | Targets are in Hz (200 / 20). A Norse-style trace must be divided by `dt` (or the targets multiplied by `dt`) before forming `Ê − Ē` (G11). |
| **Burst generation**: P_B = σ(Ṽ_d) unsigned; tanh(\|Ṽ_d\|) with sign → burst type (signed) | none | Not in Norse. Izhikevich `tonic_bursting`/`phasic_bursting` presets are deterministic intrinsic dynamics, not this mechanism. Plain `torch.bernoulli(p) * z` is enough. For the record, `snn.logistic_fn(x, 0.5)` samples Bernoulli(σ(x)) with a tanh surrogate backward (verified), but gradients aren't needed for local learning. |
| **Feedforward synapses** w_ij, events → soma (eq. 6) | `torch.nn.Linear(n_pre, n_post, bias=False)` | `weight` has shape `(post, pre)`, the same convention as Norse recurrent cells and `stdp_step_linear`. |
| **Feedback synapses** b_ij, bursts → dendrite, from the *previous* step (eq. 7–8, §3.9) | plain tensors / `nn.Linear` | Norse has no delays (G5), so the caller must keep last step's burst output. Symmetric feedback = using `W.T`. FA/DFA = fixed random `B`. Signed model uses `B⁺ − B⁻` trains through the same `b_ij`. |
| **Teaching signal** T_i (eq. 9) and **hidden regularization** R_i (eq. 10–12) | none | Elementwise torch on rate estimates. `RegularizationCell` is unrelated (it accumulates spike counts for a loss term). |
| **Plasticity**: dw/dt = −η Ẽ_j (B⁺_i − B⁻_i) or −η Ẽ_j (B_i − B_P S_i) (§3.5, eq. 13–14) | `functional/stdp.py::stdp_step_linear` as a **structural reference only** | Same shape pattern (`einsum` of post events × pre traces → `(post, pre)`), but a different rule. If you reuse STDP code: it **sums over batch** and **clamps weights to [0, 1] by default** (G12). Apply updates under `torch.no_grad()` (G8). Paper trains one sample at a time with online updates every step. |
| Low-resolution weights + accumulator (§3.6) | none | Custom. |
| Weight init, feedback scaling, dropout sparsity (§3.7, §3.9.3) | plain torch (`torch.nn.init`, `torch.nn.functional.dropout`) | none |
| **Input encoding**: Poisson, pixel → 20–200 Hz, 100 ms (§3.8) | `snn.poisson_encode` | `poisson_encode(20 + 180*x, 100, f_max=1.0)` gives rates in Hz with shape `(100, *x.shape)`, time first (G10). The default `f_max=100` would give 0–100 Hz. The per-image intensity normalization is custom. |
| **Simulation**: forward Euler, dt = 1 ms, bottom-up within a step, reset everything per sample (§3.9) | Norse default `dt=0.001` and explicit-state stepping | Matches Norse's Euler step. "Reset per sample" = pass fresh states (or `state=None`) for each sample. Time-lifted modules (`LIFBox`, `Lift`) process one layer over all T before the next layer. That's fine for pure feedforward (BPTT), but the within-step bottom-up order plus previous-step feedback needs a manual per-step loop over layers. |
| **Testing**: no dendritic input, plasticity off, argmax of output activity (§3.9.2) | none | Sum spikes over dim 0 (time), then argmax. |
| **BPTT baseline**: same soma dynamics, sigmoid surrogate, lr 2e-4, 100 ms (§3.10) | `LIFBoxCell`/`LIFBox` + `method=...`, `SequentialState`, `Lift`, `torch.optim` | This is what Norse is built for. The paper says "sigmoid function". Norse's `"super"` is the fast-sigmoid derivative `1/(α\|x\|+1)²`. An exact logistic-sigmoid derivative needs a custom autograd Function (G13). `alpha` sets the surrogate width in units of `v - v_th`. Examples: `$NORSE/task/mnist.py`, `$NORSE/torch/module/test/test_training.py`. |

## Related Norse pieces the paper doesn't use

- `functional/tsodyks_makram.py::stp_step`: short-term plasticity. The paper cites target-dependent STP as the biological
  basis for routing bursts vs events, but models the routing directly.
- `LIFRefrac*`: absolute refractory period (in steps). The paper's neurons have none, apart from the signed model's dendritic reset.
- `LIFAdEx` / `LSNN`: adaptation. The paper explicitly uses LIF without adaptation.

## Other papers in `docs/papers/`

`payeur2021.pdf` (original Burstprop), `greedy2026burstccn.pdf` / BurstCCN, `stuck2023.pdf`, `chapman2023/2024`. They
aren't mapped yet. Add a separate `<paper>-api-map.md` if one becomes a target.
