"""Burst-dependent plasticity rules (Stuck et al. 2025, eqs. 13-14; Fig. 2F).

AI DISCLOSURE: this skeleton (docstrings and signatures) was written by Claude Opus. The function bodies are
for the students to implement.

Sign convention (architecture §12): bursts potentiate. These rules have a leading +eta, where Stuck's printed
equations have -eta; the teacher and regularizer flip sign to match.

Each rule returns the weight change for one timestep, to *add* to the weights: W += dW. Shapes:

    pre_trace     (batch, n_pre)    presynaptic eligibility trace, from traces.trace_step
    post signals  (batch, n_post)   0/1 burst and event trains of the postsynaptic neurons this step
    dW            (n_post, n_pre)   same layout as nn.Linear.weight

The batch dimension is *averaged* (architecture §4: batch copies share weights and average their local updates).
Each entry is still local: dW[i, j] uses only postsynaptic neuron i's bursts/events and presynaptic neuron j's
trace.

How eta relates to the paper's eta = 1e-4 (per step or per second, and with traces in Hz) is still open
(replication notes §5). These functions just multiply by whatever eta they're given.
"""

import torch
from torch import Tensor


def signed_rule(pre_trace: Tensor, bursts_plus: Tensor, bursts_minus: Tensor, eta: float) -> Tensor:
    """Signed model:  dW[i, j] = eta * mean over batch of (B+_i - B-_i) * E~_j.

    A positive burst strengthens every synapse onto that neuron in proportion to how active its presynaptic
    neuron has been. A negative burst weakens them. No burst, no change.
    """
    a = bursts_plus - bursts_minus
    b = eta * pre_trace
    batch_size = pre_trace.shape[0]
    return (a.T @ b) / batch_size


def unsigned_rule(
    pre_trace: Tensor,  # (batch, n_pre) presynaptic eligibility trace
    bursts: Tensor,  # (batch, n_post) burst train
    events: Tensor,  # (batch, n_post) event train
    eta: float,
    p_baseline: float = 0.5,
) -> Tensor:
    """Unsigned model:  dW[i, j] = eta * mean over batch of (B_i - p_baseline * S_i) * E~_j.

    S_i is the postsynaptic event train and B_i its bursts (a burst is also an event). Per event:
      - a burst changes the weight by +eta * (1 - p_baseline) * E~_j  (potentiation)
      - a single spike changes it by -eta * p_baseline * E~_j         (depression)
    When events burst with probability p_baseline, these cancel on average. p_baseline = 0.5 = sigmoid(0) matches
    the burst probability of a dendrite at rest.
    """
    a = bursts - p_baseline * events  # (batch, n_post)
    b = eta * pre_trace  # (batch, n_pre)

    # For each batch we want the outer product. It's mind-warpy to think about the batched version,
    # but apparently it works out as:
    # a.t @ b -> (n_post, batch) * (batch, n_pre) = (n_post, n_pre) which matches the weight matrix.
    # That gives the sum over the batch so divide by the batch size to get the average over batches.
    batch_size = pre_trace.shape[0]
    return (a.T @ b) / batch_size
