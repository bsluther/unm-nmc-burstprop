# Norse overview

Norse is plain PyTorch. Neurons are ordinary functions and `nn.Module`s that carry explicit state.
There is no simulator, scheduler, or network graph object. You write the time loop (or let a
wrapper write it), and weights are ordinary `nn.Linear`/`nn.Conv2d` layers. That makes it easy to
bend for non-gradient (local) learning rules. You just stop using autograd.

## Package layout (`$NORSE/`)

```
norse/torch/
  functional/   stateless step functions + State/Parameters tuples  (the real math)
  module/       nn.Module wrappers around functional steps           (Cell / time-lifted / Recurrent)
  utils/        pytree (state tuples), plot, tensorboard hooks, NIR import/export
  models/       ready-made conv nets (VGG, MobileNet) - not relevant here
norse/task/     example training scripts (mnist.py, cartpole.py, ...) - BPTT examples
norse/dataset/  speech / memory datasets - not relevant here
```

`norse.torch` re-exports *most* things. `norse.torch.functional` and `norse.torch.module` have empty
`__init__`s. Some useful objects are **not** re-exported (see gotcha G9), so prefer full-path imports:

```python
import norse.torch as snn                                   # convenience namespace
from norse.torch.functional.lif_box import LIFBoxParameters, LIFBoxFeedForwardState, lif_box_feed_forward_step
from norse.torch.module.lif_box import LIFBox                # not in snn.*
```

## Three API layers for each neuron model

Take the LIF family as the example (`functional/lif.py`, `module/lif.py`):

| Layer | Example | Input shape | Returns |
|---|---|---|---|
| Functional step | `lif_feed_forward_step(x, state, p, dt)` | `(batch, ...)` one timestep | `(z, new_state)` |
| Cell module (one step) | `LIFCell(p, dt=...)` | `(batch, ...)` | `(z, state)` |
| Time-lifted module | `LIF(p, dt=..., record_states=False)` | `(time, batch, ...)` | `(z_all_steps, final_state)` |
| Recurrent variants | `LIFRecurrentCell(in, hidden)`, `LIFRecurrent(...)` | as above | own input + recurrent weights |

- "Feed-forward" means the step takes an *already-weighted input* (for example the output of an `nn.Linear`).
  Recurrent variants take raw spikes and hold their own `input_weights` / `recurrent_weights` (shape `(out, in)`).
- **Time is dimension 0** for every time-lifted module and encoder. Cells never see time.
- If `state=None` is passed, the module builds a default state via `initial_state(input)`.
- `record_states=True` on a time-lifted module returns the state with each field stacked over time
  (`states.v.shape == (T, batch, ...)`). Otherwise you get only the last state.
- All the base classes are in `module/snn.py` (`SNNCell`, `SNN`, `SNNRecurrentCell`, `SNNRecurrent`) and are
  about 50 lines each. Each class stores an `activation` (the functional step), a `state_fallback`, `p` and `dt`,
  and loops over time in Python.

## State and parameters are NamedTuples

- `XParameters` and `XState` are NamedTuples. Most are built with
  `pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta` (`utils/pytree.py`). That makes them
  torch pytrees (so `tree_map` works) and adds `.to(device)`, `.cuda()`, `.float()` and `.broadcast(template)`.
  Some older ones, such as `LIParameters`, `LIBoxParameters` and `LIFRefracState`, are plain `NamedTuple`s without those helpers (G14).
- They are immutable. Use `p._replace(v_th=torch.tensor(0.5))`.
- Composite models nest tuples, e.g. `LIFRefracState(lif=LIFState(...), rho=...)`.
- Parameter fields should be **tensors** (G6). Defaults are module-level tensors shared between instances (G15).

## Common parameter fields

| Field | Meaning | Default (LIF / LIFBox) |
|---|---|---|
| `tau_mem_inv` | 1/τ_mem, in **1/s** (G1) | `100.` (τ = 10 ms) |
| `tau_syn_inv` | 1/τ_syn, 1/s (current-based models only) | `200.` (τ = 5 ms) |
| `v_leak`, `v_th`, `v_reset` | dimensionless by default | `0., 1., 0.` |
| `method` | spike nonlinearity / surrogate gradient: `"super"`, `"heaviside"`, `"triangle"`, `"tanh"`, `"circ"`, `"heavi_erfc"` (+ `"adjoint"` at module level for LIF/LSNN) | `"super"` |
| `alpha` | surrogate sharpness | `100.` |
| `reset_method` (LIFBox only) | `reset_value` or `reset_subtract` (`functional/reset.py`) | `reset_value` |

`LIFParameters.bio_default()` switches to mV-scale values (`v_leak=-70`, `v_th=-55`) *and*
`tau_*_inv` values in 1/ms (`1/10`). That's inconsistent with the default `dt=0.001`. Avoid it unless you also change `dt`.

## Update order in every step function

Each step does forward Euler, then threshold, then reset:

```
v_decayed = v + dt * tau_mem_inv * (v_leak - v + input)     # input already "current"-scaled (G2)
z         = threshold(v_decayed - v_th, method, alpha)      # Heaviside forward, surrogate backward; strict > (G4)
v_new     = (1 - z) * v_decayed + z * v_reset               # reset in the same step (G5)
```

The spike appears on the same step as the input that caused it, so there's no built-in one-step delay.
Any delay (such as feedback from the previous timestep) has to be arranged by the caller.

## Composing networks

- `snn.SequentialState(*modules)` is like `nn.Sequential`, but it threads state. It detects stateful layers by
  checking whether `forward` has a parameter named `state`. It returns `(output, [state_per_layer])`, with
  `None` for stateless layers. `return_hidden=True` returns every layer's output instead.
  `register_forward_state_hooks(fn)` attaches a hook to the stateful layers only.
- `snn.Lift(module)` applies a Cell (or any module) over dimension 0 (time). `snn.lift(fn, p)` does the same for a
  functional step. `Lift(LIFBoxCell())`, `LIFBox()` and `lift(lif_box_feed_forward_step)` give identical output.
- Typical stepwise pattern (what the local-learning work will need):

```python
lin = torch.nn.Linear(n_in, n_out, bias=False)
cell = snn.LIFBoxCell(p)
state = None
with torch.no_grad():                       # local rules: no autograd (G8)
    for t in range(T):
        z, state = cell(lin(x[t]), state)
        ...                                 # traces / plasticity on lin.weight (shape (n_out, n_in))
```

## Writing a custom neuron in Norse style

Norse has no plug-in registry. A custom model is just the same three pieces. The skeleton below was checked
in `tests/norse/test_norse_conventions.py::test_custom_state_tuple_pattern`:

```python
from typing import Tuple
import torch
from norse.torch.utils import pytree
from norse.torch.functional.threshold import threshold
from norse.torch.module.snn import SNNCell

class MyParameters(pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta):
    tau_mem_inv: torch.Tensor = torch.as_tensor(100.0)
    v_th: torch.Tensor = torch.as_tensor(1.0)
    method: str = "super"
    alpha: torch.Tensor = torch.as_tensor(100.0)

class MyState(pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta):
    v: torch.Tensor
    a: torch.Tensor            # add as many state variables as the model needs

def my_step(x, state: MyState, p: MyParameters, dt: float = 0.001) -> Tuple[torch.Tensor, MyState]:
    v = state.v + dt * p.tau_mem_inv * (x - state.v)
    z = threshold(v - p.v_th, p.method, p.alpha)
    return z, MyState(v=(1 - z) * v, a=state.a)

class MyCell(SNNCell):
    def __init__(self, p=MyParameters(), **kw):
        super().__init__(activation=my_step, state_fallback=self.initial_state, p=p, **kw)
    def initial_state(self, x):
        return MyState(v=torch.zeros_like(x), a=torch.zeros_like(x))
```

- The `activation` signature must be `(input, state, p, dt) -> (output, state)`. `input` can be any object (for example a
  tuple of two tensors, one per input stream), but then `SNN`/`Lift` time-lifting and `SequentialState` won't
  work unchanged, since they assume a single tensor with time on dim 0.
- Look at `functional/lif_box.py` + `module/lif_box.py` (~190 lines total) as the minimal real example, and
  `functional/lif_refrac.py` for wrapping one model's state inside another's.

## Surrogate gradients (BPTT baseline)

- `functional/threshold.py::threshold` dispatches on `method`. Forward is always a Heaviside on `v - v_th`.
  Backward depends on the method:
  - `super` (SuperSpike): `1/(alpha*|x|+1)^2`. This is the fast-sigmoid derivative, `functional/superspike.py`.
  - `triangle`: `alpha*relu(1-|x|)`
  - `tanh`: `1 - tanh(alpha x)^2`
  - `circ`, `heavi_erfc`
- To use a different surrogate (e.g. the derivative of an exact logistic sigmoid), write a
  `torch.autograd.Function` like `SuperSpike` and call it inside your own step function. `method` is a fixed
  string switch (G13).
- `lif_step` (recurrent) detaches the spike in the reset term, but `lif_feed_forward_step` and
  `lif_box_feed_forward_step` do **not**, so gradients flow through the reset. *(code)*

## Other pieces that exist

- **Encoders** (`functional/encode.py`, `module/encode.py`): `poisson_encode`, `poisson_encode_step`,
  `signed_poisson_encode`, `constant_current_lif_encode`, `spike_latency_encode`, `population_encode`. See G10 for rate units.
- **Plasticity**: `functional/stdp.py` (pair-based STDP on traces, linear and conv2d; G11, G12),
  `functional/stdp_sensor.py`, `functional/correlation_sensor.py` + `lif_correlation.py` (older, event-driven),
  `functional/tsodyks_makram.py` (short-term plasticity, `stp_step`).
- **Other neurons**: IAF, LI / LIBox (leaky integrators, no spikes), LIF, LIFBox, LIFRefrac (refractory
  period counted in **steps**, `rho_reset=5`), LIFAdEx, LIFEx, CobaLIF, LSNN (adaptive threshold), Izhikevich
  (with preset behaviours including `tonic_bursting`, `phasic_bursting`), LIFMC (G16).
- **Readout/misc**: `snn.LI` as a non-spiking readout. `module/exp_filter.py::ExpFilter` is a Linear layer plus an
  **unnormalized** filter `s_t = x_t + exp(-dt/τ) s_{t-1}`. `RegularizationCell` accumulates spike counts across
  calls, so you must reset its `.state` yourself. `SynOpsCounter` counts synaptic operations.
- **Utils**: `snn.plot_spikes_2d`, `snn.plot_heatmap_2d`, `snn.plot_neuron_states(states, "v")` (use with
  `record_states=True`), tensorboard hooks, `to_nir` / `from_nir`.
- **Adjoint** (`functional/adjoint/`): EventProp-style exact gradients for LIF/LSNN (`method="adjoint"`).
  Not relevant to local learning.
