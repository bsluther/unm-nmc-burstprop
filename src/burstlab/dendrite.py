"""Dendritic compartment: dimensionless leaky-integrator dendrite (Stuck et al. 2025, §3.1.4, eq. 5).

AI DISCLOSURE: this skeleton (module docstring, signatures, docstrings, and hint comments) was written by Claude
Opus. The function bodies are for the students to implement.
Summary of the discussion so far: the Fig. 2 pieces are done and working: the signed and unsigned plasticity rules
(plasticity.py), the signed and unsigned burst probabilities and burst sampling (bursts.py), and the exponential
trace (traces.py). The single two-compartment neuron is being built one compartment at a time. In an earlier
thread Dan asked Claude for skeleton code for the somatic compartment (§3.1.2, eqs. 1-3), which became soma.py,
and Claude then turned Dan's hand checks for it into experiments/somaTest.py. Dan then asked Claude for skeleton
code in the same style for the dendritic compartment in §3.1.4. Claude read soma.py, bursts.py, traces.py,
§3.1.4 and §§3.3-3.4.3 of docs/papers/stuck2025.pdf, docs/norse/stuck2025-api-map.md, docs/norse/gotchas.md, and
the burstlab design notes (architecture, replication notes), and wrote this file. It holds no implementations.

The dendrite is the second compartment of each hidden and output neuron (input units have none, §3.4.3). It
integrates feedback but never fires on its own. The soma (soma.py) decides *whether* an event happens. When it
does, the dendritic potential sets the chance that the event is a burst (bursts.py). The paper's equations:

    eq. 5   tau_d * dV~_d/dt = -V~_d + I~_d                             leaky integrator, no threshold
            P_B(V~_d | event)   = sigmoid(V~_d)                          unsigned model (bursts.py)
            P_B(V~_d | event)   = tanh(|V~_d|), type = sign(V~_d)         signed model (bursts.py)
            signed model only:  on a burst, V~_d -> baseline              reset (assumed baseline = 0)

    and the dendritic input I~_d comes from (§3.3, §3.4.3):
    eq. 7   I_FB,i = sum_j b_ij * B_j(t)                                 unsigned feedback bursts
    eq. 8   I_FB,i = sum_j b_ij * (B+_j(t) - B-_j(t))                    signed feedback bursts
    eq. 11  I_d,i  = T_i(t)                                              output units (teaching signal)
    eq. 12  I_d,i  = I_FB,i(t) + R_i(t)                                  hidden units (feedback + regulariser)

tau_d = 10 ms. Eq. 5 is already written in dimensionless form, so unlike soma.py there's no rescaling step. Unlike
the soma there's no threshold either, so the dendrite's state is just V~_d, and in the unsigned model nothing
ever resets it.

All functions work elementwise on tensors of any shape (typically (batch, n_neurons)). Following architecture §3,
the step is plain torch in Norse's style: step(input, state, params, dt) -> (output, state) with NamedTuple state
and parameters. Our dendrite must match Norse's li_box_feed_forward_step (architecture §3, api-map "Dendrite"),
which writes the same Euler step as v + dt * tau_mem_inv * (v_leak - v + x). Norse has no reset hook, so the
signed model's dendritic reset is ours alone and gets tested by hand.

Where things go in the two-compartment step (built later, not here). The paper doesn't spell out the order, so
this is the order the skeleton assumes. Check it against Fig. 2 and gotchas G5:
    1. dendrite_step: integrate I~_d (built from *last* step's bursts, §3.9, G5) into V~_d.
    2. soma_step: integrate I~_s, fire events, reset the soma.
    3. bursts.sample_*_bursts(events, V~_d, u): decide which events are bursts, using the V~_d from step 1.
    4. Signed model only: dendrite_burst_reset(V~_d, bursts) sets V~_d back to baseline where a burst fired.

Sign convention (architecture §12, replication notes §1): V~_d > 0 means "increase my rate", and bursts
potentiate. So the teaching signal T and regulariser R used here carry the opposite sign from Stuck's printed
eqs. 9-10. They're computed elsewhere (§3.4) and arrive here as tensors.
"""

from typing import NamedTuple

import torch  # noqa: F401  (for the implementations: torch.where, torch.zeros_like, ...)
from torch import Tensor


# --- Parameters and state (Norse-style NamedTuples) ------------------------------------------------------------
# Plain floats keep the defaults immutable, so the shared-default-tensor trap in gotchas G15 can't happen here.
# If you call Norse's li_box_feed_forward_step in a test, build LIBoxParameters from tensors instead (gotchas G6).
class DendriteParameters(NamedTuple):
    """Parameters of the dimensionless leaky-integrator dendrite (§3.1.4).

    tau_d:    dendritic membrane time constant, seconds (paper: 10 ms).
    v_reset:  baseline the signed model resets V~_d to after a burst. 0 is assumed: the paper says "reset ... to
              the baseline" without a value (replication notes, "Signed-model dendritic reset").
    """

    tau_d: float = 10e-3
    v_reset: float = 0.0


class DendriteState(NamedTuple):
    """State of the dendrite between timesteps.

    v: dimensionless dendritic membrane potential V~_d, same shape as the soma's v (e.g. (batch, n_neurons)).
    """

    v: Tensor


DEFAULT_DENDRITE_PARAMETERS = DendriteParameters()


# --- Initial state -------------------------------------------------------------------------------------------
# §3.9: every state is reset at the start of each sample, so this is called once per sample, next to
# soma.initial_soma_state.
def initial_dendrite_state(like: Tensor) -> DendriteState:
    """A dendrite at rest: V~_d = 0 everywhere, shaped like `like`.

    Hints:
      - Rest is 0 in dimensionless units, the same as the soma.
      - torch.zeros_like keeps the shape, dtype, and device of `like`.
      - Build the full shape now rather than a 0-d tensor (gotchas G7), so V~_d can be plotted before the first step.
      - At V~_d = 0 the unsigned burst probability is 0.5 (the baseline B_P) and the signed one is 0 (§3.5).
    """
    #Similar to how I initialized the soma State
    #Here again like is used the insure the the shape of the tensor says the same
    return DendriteState(v=torch.zeros_like(like))

# --- Eq. 5: leaky integration ----------------------------------------------------------------------------------
def dendrite_integrate(v: Tensor, i_tilde: Tensor, dt: float, tau_d: float) -> Tensor:
    """One forward-Euler step of  tau_d * dV~_d/dt = -V~_d + I~_d  (eq. 5).

    v:       dimensionless dendritic potential at the start of the step.
    i_tilde: dimensionless dendritic input I~_d during this step, same shape as v.
    dt, tau_d: seconds (e.g. 1e-3 and 10e-3).
    Returns the new dendritic potential. It can be any real number, positive or negative: there's no threshold.

    Hints:
      - Same equation as soma.soma_integrate with tau_s replaced by tau_d. You could even call it, but a separate
        function keeps the dendrite testable on its own against li_box_feed_forward_step.
      - With no input, v decays towards 0 by a factor (1 - dt/tau_d) per step (0.9 at the paper's values).
      - With a constant input I, v approaches I. A slowly varying input is followed with a lag of about tau_d.
    """
    #Da MATH
    #tau_d * dnew_v/dt = -v + i_tilde
    #dnew_v/dt = (-v + i_tilde) * (1/tau_d)
    #dnew_v = (-v + i_tilde) * (1/tau_d) * dt
    #dnew_v ≈ new_v - v
    #new_v = (-v + i_tilde) * (1/tau_d) * dt + v

    new_v = (-v + i_tilde) * (1/tau_d) * dt + v
    return new_v

# --- Signed model: dendritic reset after a burst (§3.1.4, last paragraph) ---------------------------------------
def dendrite_burst_reset(v: Tensor, bursts: Tensor, v_reset: float) -> Tensor:
    """Signed model only: set V~_d to v_reset wherever a burst (of either sign) fired. Leave the rest unchanged.

    v:       dendritic potential used to sample this step's bursts.
    bursts:  0/1 tensor shaped like v, 1 where any burst fired. For the signed model that's b_plus + b_minus from
             bursts.sample_signed_bursts (at most one of the two is 1 for a given neuron and step).
    v_reset: baseline, 0 by assumption.
    Returns the potential after the reset.

    Hints:
      - Same pattern as soma.soma_reset: torch.where(bursts > 0, v_reset, v), or (1 - bursts) * v + bursts * v_reset.
      - No if-statement on tensor values (architecture §2).
      - The paper describes the reset for the signed model only, as a refractoriness that helps approximate
        backprop. The unsigned model has no reset. Don't call this function there.
      - A single spike (an event that isn't a burst) does not reset the dendrite.
    """
    return torch.where(bursts > 0, v_reset, v)


# --- Full dendrite step: eq. 5, Norse-style signature -----------------------------------------------------------
def dendrite_step(
    i_tilde: Tensor,  # (batch, n_neurons) dimensionless dendritic input I~_d for this step
    state: DendriteState,
    p: DendriteParameters = DEFAULT_DENDRITE_PARAMETERS,
    dt: float = 1e-3,
) -> tuple[Tensor, DendriteState]:
    """Advance the dendrite by one timestep (eq. 5).

    Returns (v_d, new_state). v_d is the dendritic potential V~_d after integration, which
      - goes to bursts.sample_unsigned_bursts / sample_signed_bursts with the soma's events, and
      - is what Fig. 2A plots.

    Hints:
      - Only dendrite_integrate happens here. The signed reset needs this step's bursts, which need the soma's
        events, so the two-compartment step calls dendrite_burst_reset afterwards (order in the module docstring).
      - Return a fresh DendriteState(v=...) rather than modifying state.v in place.
      - With the reset left out, this should match Norse's li_box_feed_forward_step(i_tilde, state, p, dt) with
        tau_mem_inv = 1/tau_d and v_leak = 0. Norse returns (v, state) too.
    """
    #gets our new dendritic potential after integration
    #gets a fresh state using our new potential
    v_new= dendrite_integrate(state.v, i_tilde, dt, p.tau_d)
    new_state = DendriteState(v= v_new)
    return (v_new, new_state)


# --- Dendritic input from feedback bursts (§3.3, eqs. 7-8) ------------------------------------------------------
# Not part of §3.1.4, but the dendrite can't be driven in a network without them. Included as stubs, like
# soma.somatic_input, because the same Dirac-scaling question (gotchas G2) applies to bursts as to events.
def feedback_input_unsigned(bursts_higher: Tensor, b: Tensor, dt: float) -> Tensor:
    """I_FB,i = sum_j b_ij * B_j(t)  (eq. 7), for one timestep.

    bursts_higher: (batch, n_higher) 0/1 burst train of the higher-order layer, from the *previous* step (§3.9, G5).
    b:             (n_lower, n_higher) feedback weights onto this layer's dendrites.
    Returns (batch, n_lower) feedback input.

    Hints:
      - bursts_higher @ b.T, the same shape pattern as soma.somatic_input.
      - Symmetric feedback means b = W.T, where W is the (n_higher, n_lower) forward weight matrix of the
        layer above. Feedback alignment means b is a fixed random matrix.
      - Unsigned bursts are 0.5 * events on average even with no error, so this input has a baseline that the
        signed model doesn't (replication notes §2). Don't subtract it here. That's a separate experiment.
      - Same open scaling question as soma.somatic_input: pick a convention for how one burst moves V~_d.
    """
    #Im again choosing 1/dt as the scaler for how much the burst moves 
    i_fb_u = (bursts_higher @ b.T) / dt
    return i_fb_u
    


def feedback_input_signed(bursts_plus_higher: Tensor, bursts_minus_higher: Tensor, b: Tensor, dt: float) -> Tensor:
    """I_FB,i = sum_j b_ij * (B+_j(t) - B-_j(t))  (eq. 8), for one timestep.

    bursts_plus_higher, bursts_minus_higher: (batch, n_higher) 0/1 positive and negative burst trains of the
        higher-order layer, from the previous step.
    b: (n_lower, n_higher) feedback weights.
    Returns (batch, n_lower) feedback input.

    Hints:
      - Form the signed train (B+ - B-) first, which is -1, 0, or +1 per neuron, then reuse the eq. 7 pattern.
      - The same b_ij acts on both burst types, which is why the paper says this model needs weight transport.
    """
    bursts_higher= (bursts_plus_higher - bursts_minus_higher)
    i_fb_s = (bursts_higher @ b.T) / dt
    return i_fb_s
    


# --- Total dendritic input (§3.4.3, eqs. 11-12) -----------------------------------------------------------------
# T (eq. 9) and R (eq. 10) are computed from rate estimates elsewhere (§3.4) and arrive here as tensors.
def dendritic_input_hidden(i_fb: Tensor, r: Tensor) -> Tensor:
    """I~_d = I_FB(t) + R(t)  (eq. 12), the dendritic input of a hidden unit.

    i_fb: (batch, n) feedback input from feedback_input_unsigned or feedback_input_signed.
    r:    (batch, n) hidden regularisation signal R_i(t) (eq. 10), our sign convention (module docstring).
    Returns (batch, n) dendritic input for dendrite_step.

    Hints:
      - Just a sum, but keeping it as a named function makes the eq. 12 wiring easy to find and test.
      - For the Fig. 2 single-neuron check, skip this and feed dendrite_step the sinusoid directly.
      - Output units use eq. 11, I~_d = T(t), so they don't need a helper: pass T straight to dendrite_step.
      - At test time (§3.9.2) there's no dendritic input at all: pass zeros.
    """
    #again that the hidden target rates and teaching signal is going to be implemented elsewhere
    #Here we are just calculating the dendritic input here assuming those eq are correctly implemented
    i_d = i_fb + r
    return i_d