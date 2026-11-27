"""Teaching signal and hidden target rates (Stuck et al. 2025, §3.4.1 and §3.4.2, eqs. 9-10).

AI DISCLOSURE: this skeleton (module docstring, signatures, docstrings, and hint comments) was written by Claude
Opus. The function bodies are for the students to implement.
Summary of the discussion so far: the Fig. 2 pieces are done and working: the signed and unsigned plasticity rules
(plasticity.py), the signed and unsigned burst probabilities and burst sampling (bursts.py), and the exponential
trace (traces.py). The single two-compartment neuron is being built one compartment at a time. In earlier threads
Dan asked Claude for skeleton code for the somatic compartment (§3.1.2, eqs. 1-3), which became soma.py (with
Dan's hand checks turned into experiments/somaTest.py), and then for the dendritic compartment (§3.1.4, eq. 5,
plus the eqs. 7-8 and 11-12 wiring), which became dendrite.py. Dan then asked Claude for skeleton code in the same
style covering §3.4.1 (teaching signals) and §3.4.2 (hidden target rates). Claude read soma.py, dendrite.py,
bursts.py, traces.py, §§3.4-3.4.3 of docs/papers/stuck2025.pdf, the burstlab replication notes (§§1, 2, 5, 6) and
architecture §12, and docs/norse/gotchas.md G11, and wrote this file. It holds no implementations.

These two signals are what turn the dendrite into an error channel. The teaching signal T sits at the top of the
hierarchy, on the output units' dendrites. The regulariser R sits on every hidden unit's dendrite, as a
self-connection. Both compare a unit's smoothed firing rate Ē with a target rate Ê. The paper's equations:

    rate    tau * dĒ/dt = -Ē + S(t)                                     same filter as the eligibility trace (eq. 4)
    eq. 9   T_i(t) = gamma_T * d eps_T / d Ē = -2 gamma_T (Ê_i - Ē_i(t))  output units, eps_T = (Ê_i - Ē_i)^2
            Ê_i    = 200 Hz for the correct class, 20 Hz for the others
    eq. 10  R_i(t) = gamma_R * d eps_R / d Ē = -2 gamma_R (Ê - Ē_i(t))    hidden units, eps_R = (Ê - Ē_i)^2
            Ê      = 20 Hz for every hidden unit

    and where they go (§3.4.3, in dendrite.py):
    eq. 11  I~_d,i = T_i(t)                                              output units
    eq. 12  I~_d,i = I_FB,i(t) + R_i(t)                                  hidden units

tau = 10 ms (§3.1.3), gamma_T = 1e-3, gamma_R = 1e-5 (so gamma_R << gamma_T, and the regulariser doesn't drown
out the error). Input units get neither signal: they have no dendrites. Output units get T only (the paper's
"teaching signal and the regularizing signal" before eq. 11 is contradicted by eq. 11 itself and by §3.4.2,
which excludes output neurons from R). At test time (§3.9.2) there's no dendritic input at all.

Sign convention (architecture §12, replication notes §1): we use "bursts potentiate", so V~_d > 0 means
"increase my rate". T and R therefore carry the OPPOSITE sign from the printed eqs. 9-10:

    ours:   T_i = +2 gamma_T (Ê_i - Ē_i)        R_i = +2 gamma_R (Ê - Ē_i)

A unit firing below its target gets a positive dendritic input, so it bursts more (unsigned) or emits positive
bursts (signed), and its incoming weights grow. Only the product of signs along the chain matters, and plasticity.py
already uses the matching sign.

Units (replication notes §5-6, gotchas G11): Ē is in Hz, because traces.trace_step divides the 0/1 event train by
dt (a spike is a pulse of height 1/dt). That's the convention that makes the targets in Hz (200 / 20) meaningful.
With it, T is dimensionless and at most 2 * 1e-3 * 180 = 0.36 in size for a silent correct-class unit, which is
the right order of magnitude for V~_d (sigmoid / tanh are sensitive around |V~_d| < 2). If Ē were in "rate * dt"
units, gamma_T and gamma_R would be off by a factor of 1/dt = 1000.

All functions work elementwise on tensors of any shape (typically (batch, n_neurons)). Following architecture §3,
the step functions are plain torch in Norse's style: step(input, state, params, dt) -> (output, state) with
NamedTuple state and parameters.

Where things go in the network step (built later, not here). The paper doesn't spell out the order, so this is the
order the skeleton assumes. Check it against gotchas G5 and dendrite.py's module docstring:
    1. T and R for this step are computed from Ē *before* this step's events are added, i.e. from the rate
       estimate at the end of the previous step, the same one-step delay as the feedback bursts (§3.9, G5).
    2. dendrite_step integrates I~_d (= T for outputs, I_FB + R for hidden units).
    3. soma_step fires events, bursts.py samples bursts.
    4. rate_estimate_step adds this step's events into Ē, ready for the next step.
"""

from typing import NamedTuple

import torch  # noqa: F401  (for the implementations: torch.where, torch.full_like, torch.nn.functional.one_hot ...)
from burstlab import traces
from torch import Tensor


# --- Parameters and state (Norse-style NamedTuples) ------------------------------------------------------------
# Plain floats keep the defaults immutable, so the shared-default-tensor trap in gotchas G15 can't happen here.
class TeachingParameters(NamedTuple):
    """Parameters of the teaching signal (§3.4.1) and the hidden regulariser (§3.4.2).

    tau:          time constant of the rate estimate Ē, seconds. Same as the eligibility trace (§3.1.3): 10 ms.
    gamma_t:      teaching signal strength gamma_T (paper: 1e-3).
    gamma_r:      regularisation strength gamma_R (paper: 1e-5). A fixed, non-plastic "self-connection weight".
    rate_high:    target rate Ê for the output unit of the correct class, Hz (paper: 200).
    rate_low:     target rate Ê for every other output unit, Hz (paper: 20).
    rate_hidden:  target rate Ê for every hidden unit, Hz (paper: 20).
    """

    tau: float = 10e-3
    gamma_t: float = 1e-3
    gamma_r: float = 1e-5
    rate_high: float = 200.0
    rate_low: float = 20.0
    rate_hidden: float = 20.0


class RateState(NamedTuple):
    """State of a layer's rate estimate between timesteps.

    rate: smoothed firing rate Ē in Hz, same shape as the layer's events (e.g. (batch, n_neurons)).
    """

    rate: Tensor


DEFAULT_TEACHING_PARAMETERS = TeachingParameters()


# --- Initial state -------------------------------------------------------------------------------------------
# §3.9: every state is reset at the start of each sample, so this is called once per sample, next to
# soma.initial_soma_state and dendrite.initial_dendrite_state.
def initial_rate_state(like: Tensor) -> RateState:
    """A rate estimate of Ē = 0 Hz everywhere, shaped like `like`.

    Hints:
      - torch.zeros_like keeps the shape, dtype, and device of `like`.
      - Starting from 0 Hz means every output unit begins "below target", so at t = 0 the teaching signal is
        positive for all outputs (+2 * gamma_T * Ê_i). That transient is expected. Look for it in plots.
    """
    return RateState(rate = torch.zeros_like(like))


# --- Rate estimate Ē (§3.4.1: "the same time constant as for the eligibility trace") ---------------------------
def rate_estimate_step(
    events: Tensor,  # (batch, n_neurons) 0/1 event train of this layer for this step, from soma_step
    state: RateState,
    p: TeachingParameters = DEFAULT_TEACHING_PARAMETERS,
    dt: float = 1e-3,
) -> tuple[Tensor, RateState]:
    """Advance the rate estimate by one step:  tau * dĒ/dt = -Ē + S(t).

    Returns (rate, new_state), with rate the updated Ē in Hz.

    Hints:
      - This is exactly traces.trace_step(state.rate, events, dt, p.tau). Reuse it rather than rewriting the
        divide-by-dt step, so the rate estimate and the eligibility trace can't drift apart.
      - Replication notes §6: eq. 4 calls the trace Ē, eqs. 13-14 call it Ẽ. They're the same quantity, so a
        layer's outgoing eligibility trace and its rate estimate can be one tensor. Keeping a separate state here
        is fine for testing; merge them later if it saves memory.
      - Sanity check: a neuron firing at a steady r Hz should have Ē averaging r (e.g. a Poisson 50 Hz train
        gives Ē around 50, with a lot of noise because tau is only 10 ms).
      - Return a fresh RateState(rate=...) rather than modifying state.rate in place.
    """
    #Here im calling trace_step from traces.py for the new rate
    rate = traces.trace_step(state.rate, events, dt, p.tau)
    return (rate, RateState(rate = rate))


# --- Target rates Ê (§3.4.1 and §3.4.2) ------------------------------------------------------------------------
def output_target_rates(labels: Tensor, n_classes: int, rate_high: float, rate_low: float) -> Tensor:
    """Ê_i for the output layer: rate_high for the correct class, rate_low for every other class.

    labels:    (batch,) integer class labels (0 .. n_classes - 1).
    n_classes: number of output units (10 for MNIST-like datasets).
    rate_high, rate_low: Hz (200 and 20).
    Returns (batch, n_classes) target rates in Hz, as a float tensor.

    Hints:
      - torch.nn.functional.one_hot(labels, n_classes) gives a 0/1 mask. Then
        rate_low + (rate_high - rate_low) * mask, after converting the mask to float.
      - Compute this once per sample, not every step: the label doesn't change within a sample.
      - Sanity check: each row sums to rate_high + (n_classes - 1) * rate_low (200 + 9 * 20 = 380 Hz for MNIST).
    """
    
    mask = torch.nn.functional.one_hot(labels, n_classes)
    otr = rate_low + (rate_high - rate_low) * mask
    return otr
    


def hidden_target_rates(like: Tensor, rate_hidden: float) -> Tensor:
    """Ê for a hidden layer: the same low target rate_hidden (20 Hz) for every unit, shaped like `like`.

    Hints:
      - torch.full_like(like, rate_hidden).
      - It's a constant, so you could pass the float straight into regularisation_signal and let broadcasting do
        the work. A tensor-returning helper just keeps the output and hidden code paths looking the same.
    """
    return torch.full_like(like, rate_hidden)


# --- Squared-error losses (the eps_T and eps_R that eqs. 9-10 differentiate) -----------------------------------
# The simulator never needs these: T and R are written in closed form below. They're here to make the
# derivative in eqs. 9-10 concrete, and for checks: autograd of these should give back -T / gamma_T and
# -R / gamma_R (minus, because of our sign convention), and their per-sample mean is a useful training curve.
def squared_error(rate: Tensor, target: Tensor) -> Tensor:
    """eps(Ē, Ê) = (Ê - Ē)^2, elementwise.

    rate: Ē in Hz. target: Ê in Hz, same shape (or broadcastable).
    Returns the elementwise squared error, in Hz^2.

    Hints:
      - d eps / d Ē = -2 (Ê - Ē), which is the paper's eq. 9 / eq. 10 without gamma.
      - Check with torch.autograd.grad on a small tensor with requires_grad=True. That's the one place autograd
        appears near this code: Burstprop itself learns locally, without it.
    """
    
    sqError = (target - rate) ** 2
    return sqError

# --- Eq. 9: teaching signal for the output layer ---------------------------------------------------------------
def teaching_signal(rate: Tensor, target: Tensor, gamma_t: float) -> Tensor:
    """T_i(t) = +2 * gamma_T * (Ê_i - Ē_i(t))  (eq. 9, our sign convention: see the module docstring).

    rate:    (batch, n_out) output rate estimate Ē in Hz (from the end of the previous step, see module docstring).
    target:  (batch, n_out) target rates Ê in Hz, from output_target_rates.
    gamma_t: teaching strength (1e-3).
    Returns (batch, n_out) dimensionless teaching signal, passed straight to dendrite.dendrite_step (eq. 11).

    Hints:
      - This is the printed eq. 9 with the sign flipped (replication notes §1, architecture §12). Write the sign
        choice in a comment where you implement it, and test it: a unit below target must get T > 0.
      - Size check with the Hz convention: a correct-class unit at 0 Hz gets T = 0.36, at 200 Hz gets 0. A wrong
        class unit at 100 Hz gets T = 2e-3 * (20 - 100) = -0.16.
      - The paper applies T for the whole 100 ms sample, with no delay (end of §3.4.1). Unlike Payeur's model,
        there's no "teacher off, then teacher on" phase to build here.
      - At test time (§3.9.2), don't call this: the output dendrites get zeros.
    """
    #Here just recreating eq 9 from the Stuck paper
    t_i = 2 * gamma_t * (target - rate)
    return t_i


# --- Eq. 10: regularisation signal for hidden layers -----------------------------------------------------------
def regularisation_signal(rate: Tensor, target: Tensor | float, gamma_r: float) -> Tensor:
    """R_i(t) = +2 * gamma_R * (Ê - Ē_i(t))  (eq. 10, our sign convention: see the module docstring).

    rate:    (batch, n_hidden) hidden rate estimate Ē in Hz.
    target:  Ê in Hz, a tensor from hidden_target_rates or the float rate_hidden (20).
    gamma_r: regularisation strength (1e-5).
    Returns (batch, n_hidden) dimensionless regularisation signal, added to the feedback input in
    dendrite.dendritic_input_hidden (eq. 12).

    Hints:
      - Same formula as teaching_signal with gamma_R. You could call teaching_signal(rate, target, gamma_r), but a
        separate name makes the eq. 10 wiring easy to find, and keeps the two gammas from being swapped.
      - The paper calls this a self-connection: R depends only on the unit's own rate, never on other units.
      - Size check: a hidden unit at 100 Hz gets R = 2e-5 * (20 - 100) = -1.6e-3, two orders of magnitude below
        typical T. That's the intent (gamma_R << gamma_T). If R ever dominates I_FB, check the units of Ē first.
      - Not applied to input units (no dendrites) or output units (they get T, eq. 11).
    """
    r_i = 2 * gamma_r * (target - rate)
    return r_i


# --- Output-layer step: rate estimate + eq. 9, Norse-style signature -------------------------------------------
def teacher_step(
    events_out: Tensor,  # (batch, n_out) 0/1 event train of the output layer for this step
    target: Tensor,  # (batch, n_out) Ê from output_target_rates, computed once per sample
    state: RateState,
    p: TeachingParameters = DEFAULT_TEACHING_PARAMETERS,
    dt: float = 1e-3,
) -> tuple[Tensor, RateState]:
    """Compute this step's teaching signal and update the output rate estimate.

    Returns (t_signal, new_state). t_signal is T(t), computed from state.rate (the Ē from the previous step),
    for the output dendrites at the *next* step's dendrite_step. new_state holds Ē with this step's events added.

    Hints:
      - Compose: teaching_signal(state.rate, target, p.gamma_t), then rate_estimate_step(events_out, state, p, dt).
      - Order matters (module docstring, item 1). Using the old Ē gives the same one-step delay as the feedback
        bursts (G5). If you choose the other order, write it down: it changes nothing at 100 ms timescales but
        will show up in exact-match tests.
      - Return a fresh RateState rather than modifying state.rate in place.
    """
    #Calling the teaching signal to get the teaching sig
    #rate estimate step gets the new rate
    
    t_i = teaching_signal(state.rate, target, p.gamma_t)
    _, new_state = rate_estimate_step(events_out, state, p, dt)
    return (t_i, new_state)

# --- Hidden-layer step: rate estimate + eq. 10, Norse-style signature ------------------------------------------
def regulariser_step(
    events_hidden: Tensor,  # (batch, n_hidden) 0/1 event train of one hidden layer for this step
    state: RateState,
    p: TeachingParameters = DEFAULT_TEACHING_PARAMETERS,
    dt: float = 1e-3,
) -> tuple[Tensor, RateState]:
    """Compute this step's regularisation signal and update the hidden rate estimate.

    Returns (r_signal, new_state), with the same timing as teacher_step: R(t) comes from the previous Ē.

    Hints:
      - Compose: regularisation_signal(state.rate, p.rate_hidden, p.gamma_r), then rate_estimate_step.
      - One RateState per hidden layer. With 4 hidden layers you'll have 4 of these plus one for the output.
      - Same one-step-delay choice as teacher_step: keep the two consistent.
    """
    #calling the regularisation signal to get new regSig
    #Again rate estimate step for the new state
    regSig = regularisation_signal(state.rate, p.rate_hidden, p.gamma_r)
    _, new_state = rate_estimate_step(events_hidden, state, p, dt)
    return (regSig, new_state)
