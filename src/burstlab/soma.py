"""Somatic compartment: dimensionless LIF soma (Stuck et al. 2025, §3.1.2, eqs. 1-3).

AI DISCLOSURE: this skeleton (module docstring, signatures, docstrings, and hint comments) was written by Claude
Opus. The function bodies are for the students to implement.
Summary of the discussion so far: the Fig. 2 pieces are done and working: the signed and unsigned plasticity rules
(plasticity.py), the signed and unsigned burst probabilities and burst sampling (bursts.py), and the exponential
trace (traces.py). The next step is the single two-compartment neuron, starting with the soma. Daniel asked Claude
for skeleton code, in the style of plasticity.py and traces.py, for the equations in §3.1.2. Claude read §3.1.2 of
docs/papers/stuck2025.pdf, docs/norse/stuck2025-api-map.md, docs/norse/gotchas.md, and the burstlab design notes,
and wrote this file. It holds no implementations.

The soma decides *whether* a neuron emits an event. The dendrite (§3.1.4, bursts.py) then decides whether that
event is a single spike or a burst. The paper writes the soma in physical units (eq. 1), then rescales to
dimensionless variables (eq. 3), which is the form we simulate:

    eq. 1   tau_s * dV_s/dt = -(V_s - E_L) + I_s / g_L                  physical units (volts, amps, siemens)
    eq. 2   if V_s >= V_threshold:  V_s -> V_reset                       fire an event, then reset
    eq. 3   tau_s * dV~_s/dt = -V~_s + I~_s                              dimensionless

    with    V~_s = (V_s - E_L) / (V_threshold - E_L)
            I~_s = I_s / (g_L * (V_threshold - E_L))

In dimensionless units rest is 0 and threshold is 1, so v_th = 1 and v_reset = 0 (assuming V_reset = E_L, which
the paper doesn't state). tau_s = 10 ms. The paper prints V~_s = V_s - E_L / (V_threshold - E_L) without the
parentheses (docs/notation.md copies it). The parenthesized form above is the one that makes rest 0 and
threshold 1.

All functions work elementwise on tensors of any shape (typically (batch, n_neurons)). Following architecture §3,
the step is plain torch in Norse's style: a functional step(input, state, params, dt) -> (output, state) with
NamedTuple state and parameters. With the dendrite silenced, soma_step should match Norse's
lif_box_feed_forward_step (docs/norse/stuck2025-api-map.md), apart from two known differences:
  - Norse's threshold is strict (v > v_th), the paper's is v >= v_th (gotchas G4).
  - How a Dirac input spike becomes a per-step current is still open (eq. 6, replication notes §5, gotchas G2).
    This file takes I~_s as given. Whoever builds I~_s from the presynaptic events decides the scaling.
"""

from typing import NamedTuple

import torch  # noqa: F401  (for the implementations: torch.where, torch.zeros_like, ...)
from torch import Tensor


# --- Parameters and state (Norse-style NamedTuples) ------------------------------------------------------------
# Plain floats keep the defaults immutable, so the shared-default-tensor trap in gotchas G15 can't happen here.
class SomaParameters(NamedTuple):
    """Parameters of the dimensionless LIF soma (§3.1.2).

    tau_s:    somatic membrane time constant, seconds (paper: 10 ms).
    v_th:     dimensionless threshold. 1 by construction of the rescaling in eq. 3.
    v_reset:  dimensionless reset potential. 0 if V_reset = E_L (assumed; the paper doesn't say).
    """

    tau_s: float = 10e-3
    v_th: float = 1.0
    v_reset: float = 0.0


class SomaState(NamedTuple):
    """State of the soma between timesteps.

    v: dimensionless somatic membrane potential V~_s, same shape as the input (e.g. (batch, n_neurons)).
    """

    v: Tensor


DEFAULT_SOMA_PARAMETERS = SomaParameters()


# --- Initial state -------------------------------------------------------------------------------------------
# §3.9: every state is reset at the start of each sample, so this is called once per sample.
def initial_soma_state(like: Tensor) -> SomaState:
    """A soma at rest: V~_s = 0 everywhere, shaped like `like`.

    Hints:
      - Rest is E_L, which is 0 in dimensionless units.
      - torch.zeros_like keeps the shape, dtype, and device of `like`.
    """

    # Here I'm initializing the resting potential
    # like is used here to copy the shape (so batch and num of neurons), 
    # dtype (so float32, float16,... etc), and device (so if you're using cpu or gpu)
    return SomaState(v=torch.zeros_like(like))

# --- Rescaling to dimensionless variables (definitions under eq. 2) ----------------------------------------------
# These two convert physical quantities into the dimensionless ones of eq. 3. The simulator never needs them,
# since it works in dimensionless units throughout. They're here to make the rescaling concrete, and for checking
# that eq. 1 and eq. 3 agree (simulate both and compare after rescaling).
def dimensionless_voltage(v_s: Tensor, e_l: float, v_threshold: float) -> Tensor:
    """V~_s = (V_s - E_L) / (V_threshold - E_L).

    v_s: somatic potential in volts. e_l: resting potential, volts. v_threshold: threshold, volts.
    Returns the dimensionless potential: 0 at rest, 1 at threshold.

    Hints:
      - Mind the parentheses (see the module docstring on the paper's typo).
      - Sanity checks: v_s = e_l gives 0, and v_s = v_threshold gives 1.
    """

    # if the somatic potential voltage is above our threshold we fire and rest the soma state
    # else we change the voltage base of our current soma volts

    return (v_s - e_l) / (v_threshold - e_l)

    #down blow, this is my first implemenation that was wrong, but I think it is a good way to
    #think about it and how it relates back to the paper
    
    # if v_s >= v_threshold:
    #     initial_soma_state(v_s)  here this is not really right but the idea is the potential would be reset
    #     return 1
    # else:
    #     v_s = (v_s - e_l) / (v_threshold - e_l)
    #     return 0
    


def dimensionless_current(i_s: Tensor, g_l: float, e_l: float, v_threshold: float) -> Tensor:
    """I~_s = I_s / (g_L * (V_threshold - E_L)).

    i_s: input current, amps. g_l: leak conductance, siemens. e_l, v_threshold: volts.
    Returns the dimensionless input current.

    Hints:
      - I_s / g_L is a voltage (Ohm's law), so dividing by (V_threshold - E_L) leaves a pure number.
      - Sanity check: a constant I~_s = 1 drives V~_s towards exactly threshold (eq. 3 at steady state), so it is
        the smallest constant input that can make the neuron fire.
    """
    #Ok so here the current input current is set to the old input current div by the leak conductance
    #times the volt threshold minus the resting potenial (alot of currents here)
    
    return i_s / (g_l * (v_threshold - e_l))


# --- Eq. 3: subthreshold integration --------------------------------------------------------------------------
def soma_integrate(v: Tensor, i_tilde: Tensor, dt: float, tau_s: float) -> Tensor:
    """One forward-Euler step of  tau_s * dV~_s/dt = -V~_s + I~_s  (eq. 3), before thresholding.

    v:       dimensionless potential at the start of the step.
    i_tilde: dimensionless input I~_s during this step, same shape as v.
    dt, tau_s: seconds (e.g. 1e-3 and 10e-3).
    Returns the new potential, which may now be at or above threshold.

    Hints:
      - Same pattern as traces.trace_step: new = old + (dt / tau) * (right-hand side).
      - With no input, v decays towards 0 by a factor (1 - dt/tau_s) per step (0.9 at the paper's values).
      - With a constant input I, v approaches I. So I > 1 is needed to ever reach threshold.
      - Norse writes this as v + dt * tau_mem_inv * (v_leak - v + x), with tau_mem_inv = 1/tau_s and v_leak = 0.
    """

    #da math
    #dV~_s/dt = (-v + i_tilde)/tau_s
    #dV~_s = (-v + i_tilde) * (1/tau_s) * dt
    #Next step I got a little confused on
    #dV~_s ≈ V~_s - v
    #so for last step
    #V~_s = (-v + i_tilde) * (1/tau_s) * dt + v
    
    v_new: Tensor = (-v + i_tilde) * (1/tau_s) * dt + v
    return v_new


# --- Eq. 2, first half: emit an event --------------------------------------------------------------------------
def soma_fire(v: Tensor, v_th: float) -> Tensor:
    """Event train for this step: 1 where V~_s >= v_th (eq. 2), else 0.

    v: dimensionless potential after integration. v_th: dimensionless threshold (1).
    Returns a 0/1 tensor shaped like v, in v's dtype (so it can be multiplied into weights and traces).

    Hints:
      - A comparison gives a bool tensor. Convert it with .to(v.dtype), as bursts.py does.
      - The paper uses >=, Norse uses >. Keep the paper's >= and note the difference in the tests.
      - No surrogate gradient is needed: Burstprop learns locally, without autograd. The BPTT baseline uses Norse.
    """

    #This just giving a 0 or 1 based of if the voltage surpassed the threshold

    return (v >= v_th).to(v.dtype)

    #Again this was my first attempt but I think it is easier to visuals this way
    # if v >= v_th.to(v.dtype):
    #     return 1
    # else:
    #     return 0


# --- Eq. 2, second half: reset ---------------------------------------------------------------------------------
def soma_reset(v: Tensor, events: Tensor, v_reset: float) -> Tensor:
    """Set V~_s to v_reset wherever an event fired (eq. 2). Leave the other neurons unchanged.

    v: potential after integration. events: 0/1 tensor from soma_fire. v_reset: dimensionless reset (0).
    Returns the potential after the reset.

    Hints:
      - Avoid an if-statement on tensor values: architecture §2 asks for no data-dependent branches in the step.
      - Either torch.where(events > 0, v_reset, v), or the arithmetic form (1 - events) * v + events * v_reset.
      - There's no refractory period: a neuron can fire again on the very next step (api-map, "LIFRefrac").
    """
    #Here we want to reset the potential if we fired
    #torch.where works by checking the condition and returning v_reset or v based off if it's true or false
    return torch.where(events > 0, v_reset, v)


# --- Full soma step: eqs. 2-3 together, Norse-style signature ---------------------------------------------------
def soma_step(
    i_tilde: Tensor,  # (batch, n_neurons) dimensionless somatic input I~_s for this step
    state: SomaState,
    p: SomaParameters = DEFAULT_SOMA_PARAMETERS,
    dt: float = 1e-3,
) -> tuple[Tensor, SomaState]:
    """Advance the soma by one timestep: integrate (eq. 3), fire, and reset (eq. 2).

    Returns (events, new_state). events is the 0/1 event train S(t) for this step, which feeds
      - the dendrite's burst decision (bursts.sample_unsigned_bursts / sample_signed_bursts),
      - the eligibility trace (traces.trace_step), and
      - the next layer's somatic input (eq. 6).

    Hints:
      - Compose the three functions above in order: soma_integrate -> soma_fire -> soma_reset.
      - Order matters. Like Norse (gotchas G5), the event is emitted on the same step as the input that pushed v
        over threshold, and the returned state is already reset.
      - Return a fresh SomaState(v=...) rather than modifying state.v in place.
      - The dendrite and bursts come later, in the two-compartment step. Keep this function soma-only so it can be
        tested against Norse's lif_box_feed_forward_step on its own.
    """

    #here we are just calling the function from above in the right order

    
    v = soma_integrate(state.v, i_tilde, dt, p.tau_s) 
    events = soma_fire(v, p.v_th)
    v_after_reset = soma_reset(v, events, p.v_reset)
    
    return events, SomaState(v=v_after_reset)


# --- Somatic input from presynaptic events (§3.3, eq. 6) --------------------------------------------------------
# Not part of §3.1.2, but the soma can't be driven without it. Included as a stub because the scaling is an open
# decision (replication notes §5) that has to be made before comparing weights with the paper.
def somatic_input(events_pre: Tensor, weights: Tensor, dt: float) -> Tensor:
    """I~_s,i = sum_j w_ij * S_j(t)  (eq. 6), for one timestep.

    events_pre: (batch, n_pre) 0/1 presynaptic event train for this step.
    weights:    (n_post, n_pre) feedforward weights, same layout as nn.Linear.weight.
    Returns (batch, n_post) somatic input.

    Hints:
      - The matrix part is events_pre @ weights.T, the same shape pattern as the plasticity rules.
      - Open question (replication notes §5): S_j(t) is a Dirac train, so how far does one spike of weight w move
        v in one Euler step of eq. 3 (a change of (dt / tau_s) * I~_s)? Three candidate conventions:
          (a) use the 0/1 events as they are:     dv = w * dt / tau_s   (Norse's choice, 0.1 * w at defaults, G2)
          (b) pulse of height 1/dt, as in traces:  dv = w / tau_s        (w then carries units of seconds)
          (c) scale so a spike is a jump of w:    multiply by tau_s / dt, giving dv = w
        Pick one, write it down here and in the replication notes, and test it.
    """
    #This was the most confusing one for me gonna have to come back and look at this one
    #Claude told me that events_pre @ weights.T is the sum of the weights of the inputs that spiked
    #So this is ∑(w_ij * S_j(t)) for one timestep
    
    #NOTE: here I choice option b
    
    #w_ij represents the is the feedforward synaptic weight from lower-order neuron j to higher-order neuron i
    i_tilde = (events_pre @ weights.T) / dt
    return i_tilde
    
