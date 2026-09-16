# Norse gotchas

Each entry was verified against the pinned version. Tests live in `tests/norse/test_norse_conventions.py`
(named `test_gN_*`). Entries marked *(code)* were checked by reading the source only.

### G1. Time is in seconds, and time constants are inverse seconds
`dt` defaults to `0.001` (1 ms), and `tau_mem_inv = 100.` means τ = 10 ms. Several docstrings say "1/ms". They're
wrong for the defaults. One decay step multiplies `v` by `1 - dt*tau_mem_inv` = 0.9. Set τ in seconds:
`tau_mem_inv=torch.tensor(1/10e-3)`. `LIFParameters.bio_default()` uses 1/ms values and does not match `dt=0.001`.

### G2. The feed-forward input is a current, not a voltage jump
`v += dt * tau_mem_inv * (v_leak - v + x)`, so a one-step input `x` moves `v` by `dt*tau_mem_inv*x`
(0.1·x with defaults). To get an instantaneous jump of Δv per presynaptic spike, the synaptic weight must be
`Δv / (dt*tau_mem_inv)`. Keep this in mind when you compare weights or thresholds with a paper's equations. The same
scaling applies to `LIBox`/`LI` (leaky integrators).

### G3. `LIF` / `LIFCell` add a synaptic current filter; `LIFBox` / `LIFBoxCell` do not
In the current-based LIF, input is added to state `i`, which enters `v` on the same step and then decays with
`tau_syn_inv` (default τ_syn = 5 ms). So a single input spike keeps driving `v` over the following steps. To get
"instantaneous synapse" dynamics (τ dV/dt = −V + Σ w S), use the **Box** variants. The same split applies to
`LI` (current-filtered) vs `LIBox` (direct).

### G4. The threshold is strict
`heaviside(x) = x > 0`, so `v == v_th` does **not** spike. Papers often write `V ≥ V_th`. This rarely matters with
float inputs, but it does in hand-built tests with exact values.

### G5. Integrate, threshold and reset all happen in one step
The spike is emitted on the same step as the input that pushed `v` over threshold, and the returned `v` is already
reset. There's no built-in synaptic or axonal delay. Feedback "from the previous timestep" must be arranged by the caller,
for example by keeping last step's outputs in your own state.

### G6. Box / LI parameters must be tensors
`LIFCell`/`LIF` wrap every field in `torch.as_tensor`, but `LIFBoxCell`, `LIFBox`, `LIBoxCell` and `LIBox`
don't, and their `initial_state` calls `.detach()` or `clone_tensor` on `p.v_leak`. If you pass Python floats, you get
`ValueError` / `AttributeError`. Always build parameters with `torch.tensor(...)`.

### G7. The default initial state can be a 0-d tensor
`LIFCell.initial_state` gives `v` shape `()` (a clone of `v_leak`) and `i` the full shape. `LIFBoxCell`/`LIBoxCell`
start with a scalar `v`, which only broadcasts to `(batch, n)` after the first step. If you index or plot `state.v`
before stepping, or write traces keyed on its shape, build the state yourself:
`LIFBoxFeedForwardState(v=torch.zeros(batch, n))`.

### G8. An autograd graph is built even with no trainable parameters
Every `initial_state` sets `state.v.requires_grad = True`, so a 100-step simulation builds a graph over the
whole sequence even if nothing is optimized. That wastes memory and slows local-learning code. Wrap local-learning simulation in `torch.no_grad()` (or `torch.inference_mode()`).
Inside `no_grad`, state tensors come out with `requires_grad=False`. Use autograd only for the BPTT baseline.

### G9. `norse.torch` does not re-export everything
Import these by full path:
- `norse.torch.module.lif_box.LIFBox`
- `norse.torch.module.leaky_integrator_box.LIBox`
- `norse.torch.functional.stdp.{STDPState, STDPParameters, stdp_step_linear, stdp_step_conv2d}`
- `norse.torch.module.encode.PoissonEncoderStep`
- `norse.torch.module.exp_filter.ExpFilter`

`LIFBoxCell`, `LIBoxCell`, `LIF*`, `SequentialState`, `Lift` and the encoders' functional forms *are* in `snn.*`.

### G10. Poisson encoders: rate = `f_max * x`, and the defaults differ
`poisson_encode(x, seq_length, f_max=100, dt=0.001)` draws a Bernoulli per step with p = `dt*f_max*x`
(output shape `(seq_length, *x.shape)`, time first). `poisson_encode_step` defaults to `f_max=1000`, which means
`x=1` spikes on every step. To specify rates directly in Hz, pass `f_max=1.0` and `x` in Hz, e.g.
`poisson_encode(rates_hz, T, f_max=1.0)`. At most one spike per step, so rates saturate at 1/dt.
`signed_poisson_encode` (sequence version) creates its random tensor without `device=` and will fail on CUDA. *(code)*

### G11. STDP traces are "rate × dt", not rate in Hz
`STDPState.decay` does `t += dt*tau_inv*(-t + a*z)`. Each spike bumps the trace by `dt*tau_inv*a` (0.1 for τ = 10 ms),
and for a Poisson train at rate r the trace settles at about `r*dt` (50 Hz → 0.05). Divide by `dt` to get Hz. `decay`
**mutates the state object in place and returns `None`**. `STDPState`/`STDPParameters` are plain classes, not tuples.

### G12. `stdp_step_linear` sums over the batch and clamps weights to [0, 1] by default
- The weight update is `einsum("bi,bj->ij", ...)`, a **sum** over the batch (not a mean), so the effective learning rate
  grows with batch size.
- `w` has shape `(post, pre)`, like `nn.Linear.weight`.
- Traces are decayed and updated **before** the weight update in the same call, so a pre and a post spike on the same
  step both count.
- Defaults are `w_min=0., w_max=1., hardbound=True`, so **signed weights get clipped to 0**. Pass
  explicit bounds or `hardbound=False` for signed networks.
- `stdp_step_linear` doesn't use `torch.no_grad()` internally.
- Separately, in `stdp_sensor.py` both traces decay with `tau_c_inv`, and `tau_ac_inv` is never used. *(code)*

### G13. `method` is a fixed string switch
`threshold()` accepts only `heaviside, super, triangle, tanh, circ, heavi_erfc`. `logistic_fn` and `circ_dist_fn` (stochastic
Bernoulli forward passes) exist in `threshold.py` but **can't** be selected via `method`, and `"sigmoid"` doesn't exist.
`"adjoint"` is intercepted by the LIF/LSNN *modules*, not by `threshold`. Custom surrogates need your own step function.

### G14. Some parameter tuples lack `.to()`
`LIParameters`, `LIBoxParameters`, `LIState`, `LIBoxState`, the `LIFRefrac*` tuples, `IAF*` and `STDPSensor*` are plain
`NamedTuple`s, with no `.to(device)` or `.broadcast()`. The pytree-based ones (`LIF*`, `LIFBox*`) have them. When moving to
GPU, move tensors field by field or use `torch.utils._pytree.tree_map`.

### G15. Default parameter tensors are shared module-level objects
`LIFBoxParameters().v_leak is LIFBoxParameters().v_leak` → `True`. Never modify a parameter tensor in place
(`p.v_th += ...`, `.requires_grad_()`, `.to_()`), because that changes the default for every model. Use `_replace` with a fresh tensor.

### G16. `LIFMCRecurrentCell` is not a soma/dendrite model
"Multi-compartment" here means `v += dt * g_coupling @ v`, with `g_coupling` shaped `(hidden, hidden)` (random, trainable
by default). It couples *units within a layer* on top of a normal current-based LIF (`lif_mc.py` is 68 lines). It has no
separate dendritic state, no compartment-specific inputs, and no burst mechanism. Its `initial_state` also assumes 2-D
`(batch, in)` input. A two-compartment neuron needs a custom model (see overview.md, "Writing a custom neuron").

### Minor
- Importing `norse.torch` emits a `FutureWarning` about `torch.jit.script` (from `functional/filter.py`). It's harmless.
- The time-lifted `SNN` modules loop over time in Python. For long sequences, stepping Cells yourself costs the same and is
  easier to instrument.
- `RegularizationCell` and `SynOpsCounter` keep accumulating across calls. Reset them between samples
  (`cell.state = None` and `counter.reset()` respectively). *(code)*
- The refractory period in `LIFRefrac*` (`rho_reset`) is counted in **steps**, not seconds. *(code)*
