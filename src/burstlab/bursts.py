"""Burst generation: which somatic events become bursts (Stuck et al. 2025, §3.1.4; Fig. 2B, D).

AI DISCLOSURE: this skeleton (docstrings and signatures) was written by Claude Opus. The function bodies are
for the students to implement.

The soma decides *whether* an event happens. The dendrite decides whether that event is a burst. There's no
threshold on the dendrite: each event independently becomes a burst with a probability set by the dendritic
potential v_d. All functions work elementwise on tensors of any shape (typically (batch, n_neurons)).

Sign convention (architecture §12): v_d > 0 means "increase my rate", so positive bursts strengthen synapses.

Sampling takes the uniform random numbers `u` as an argument instead of drawing them inside. That keeps these
functions pure and exactly testable: the caller draws `u = torch.rand_like(v_d)` each timestep.

The signed model also resets the dendrite after a burst. That needs the dendrite's state, so it belongs to the
two-compartment neuron step (phase 1, after Fig. 2), not here.
"""

from torch import Tensor


def unsigned_burst_prob(v_d: Tensor) -> Tensor:
    """Unsigned model: P(burst | event) = sigmoid(v_d). It is 0.5 when the dendrite is at rest (v_d = 0)."""
    raise NotImplementedError


def signed_burst_probs(v_d: Tensor) -> tuple[Tensor, Tensor]:
    """Signed model: probabilities of a positive and a negative burst, given an event.

    P(burst | event) = tanh(|v_d|), and the sign of v_d picks the type (Stuck: long bursts for v_d > 0, short for
    v_d < 0). Split by type, as in Fig. 2B:

        p_plus  = max(0,  tanh(v_d))
        p_minus = max(0, -tanh(v_d))

    At most one of them is nonzero, and both are 0 when v_d = 0: no error, no bursts.

    Returns (p_plus, p_minus), each shaped like v_d.
    """
    raise NotImplementedError


def sample_unsigned_bursts(events: Tensor, v_d: Tensor, u: Tensor) -> Tensor:
    """Burst train for the unsigned model.

    events: 0/1 tensor, the soma's events this timestep.
    u:      uniform [0, 1) random numbers, same shape.

    An event becomes a burst when u < unsigned_burst_prob(v_d). No event means no burst. Returns a 0/1 tensor with
    the same shape and dtype as `events`.
    """
    raise NotImplementedError


def sample_signed_bursts(
    events: Tensor, v_d: Tensor, u: Tensor
) -> tuple[Tensor, Tensor]:
    """Positive and negative burst trains for the signed model.

    An event becomes a burst when u < tanh(|v_d|); its type follows the sign of v_d. Returns (bursts_plus,
    bursts_minus): 0/1 tensors shaped and typed like `events`. They are never both 1 for the same neuron and step.
    """
    raise NotImplementedError
