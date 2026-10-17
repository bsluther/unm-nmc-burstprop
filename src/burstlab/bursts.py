"""Burst generation: which somatic events become bursts (Stuck et al. 2025, §3.1.4; Fig. 2B, D).

AI DISCLOSURE: this header and function signautres were written by Claude Opus. The function bodies
and docstrings were human authored.

The soma decides *whether* an event happens. The dendrite decides whether that event is a burst.
There's no threshold on the dendrite: each event independently becomes a burst with a probability
set by the dendritic potential v_d. All functions work elementwise on tensors of any shape
(typically (batch, n_neurons)).

Sign convention (architecture §12): v_d > 0 means "increase my rate", so positive bursts strengthen
synapses.

Sampling takes the uniform random numbers `u` as an argument instead of drawing them inside. That
keeps these functions pure and exactly testable: the caller draws `u = torch.rand_like(v_d)` each
timestep.

The signed model also resets the dendrite after a burst. That needs the dendrite's state, so it
belongs to the two-compartment neuron step (phase 1, after Fig. 2), not here.
"""

import torch
from torch import Tensor


def unsigned_burst_prob(v_d: Tensor) -> Tensor:
    """Probability of an event becoming a burst given the dendritic membrane potential, in the
    unsigned model.

    Args:
        v_d: Dimensionless dendritic membrane potential.

    Returns:
        Unsigned burst probability.
    """
    return torch.sigmoid(v_d)


def signed_burst_probs(v_d: Tensor) -> tuple[Tensor, Tensor]:
    """Probabilities of an event becoming a positive or negative burst given the dendritic
    membrane potential, in the signed model. At least one of the two (positive or negative) is
    always zero.

    Args:
        v_d: Dimensionless dendritic membrane potentials.

    Returns:
        Positive and negative burst probabilities.
    """
    t = torch.tanh(v_d)
    return (
        torch.relu(t),  # max(0, tanh(v_d))
        torch.relu(-t),  # max(0, -tanh(v_d))
    )


def sample_unsigned_bursts(events: Tensor, v_d: Tensor, u: Tensor) -> Tensor:
    """Generate the burst train corresponding to the provided event train and dendritic membrane
    potentials.

    Args:
        events: The event train.
        v_d: Dimensionless dendritic membrane potentials.
        u: Samples from a Uniform(0, 1) distribution used to choose if an event is a burst, given
            the burst probability. Provided by the caller for determinism.

    Returns:
        Tensor: The burst train, a 0/1 tensor shaped like `events`.
    """
    return (u < unsigned_burst_prob(v_d)).to(events.dtype) * events


def sample_signed_bursts(events: Tensor, v_d: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
    """Generate the postive and negative burst trains corresponding to the provided event train and
    dendritic membrane potentials.

    Args:
        events: The event train.
        v_d: Dimensionless dendritic membrane potentials.
        u: Samples from a Uniform(0, 1) distribution used to choose if an event is a burst, given
            the signed burst probability. The sign of v_d chooses whether bursts are positive or
            negative. Provided by the caller for determinism.

    Returns:
        A tuple containing the positive burst train and negative burst train.
    """
    p_plus, p_minus = signed_burst_probs(v_d)
    return (u < p_plus).to(events.dtype) * events, (u < p_minus.to(events.dtype)) * events
