"""Rate-model Burstprop: a learning exercise (Payeur et al. 2021, Methods, "Deep network model").

AI DISCLOSURE: this skeleton (docstrings and signatures) was written by Claude Opus. The function bodies are
the exercise.

Each unit stands for an *ensemble* of two-compartment neurons, described by three numbers:

    e  event rate         the feedforward signal (somatic input)
    p  burst probability  the feedback signal (dendritic input)
    b  burst rate,  b = p * e

Learning compares two feedback passes: one **without** the teacher (p, b) and one **with** it (p_t, b_t). The
difference in burst rate, b_t - b, plays the role of backprop's error. Read `README.md` in this directory first.

Conventions (all tensors are batch-first):

    rates[l]             e_l, shape (batch, n_l). rates[0] is the input x; rates[-1] is the output e_L.
    weights[l-1]         W_l, shape (n_l, n_{l-1}), the same layout as nn.Linear.weight.
    biases[l-1]          shape (n_l,).
    feedback_weights[l-1]  Y_l, the feedback matrix carrying bursts from layer l+1 to the dendrites of hidden
                         layer l. It has the same shape as W_{l+1}, i.e. (n_{l+1}, n_l). Symmetric feedback means
                         feedback_weights = weights[1:]; feedback alignment means fixed random matrices.

The reference implementation is `~/dev/clones/Burstprop/layers.py` (OutputLayer, HiddenLayer, without the
recurrent inputs). Try writing yours before reading it.
"""

import torch  # noqa: F401  (you will need it)
from torch import Tensor


def forward(x: Tensor, weights: list[Tensor], biases: list[Tensor]) -> list[Tensor]:
    """Feedforward pass: event rates of every layer.

    e_0 = x,   e_l = sigmoid(W_l e_{l-1} + bias_l)   for l = 1..L      (Payeur Methods: v_l = W_l e_{l-1}, e_l = f(v_l))

    Returns [e_0, e_1, ..., e_L] (length L + 1).
    """
    raise NotImplementedError


def output_burst_probs(
    e_out: Tensor, target: Tensor, p_baseline: float
) -> tuple[Tensor, Tensor]:
    """Output-layer burst probabilities without and with the teacher.

    Without the teacher, every output unit bursts at the baseline:   p = p_baseline.

    With the teacher, the burst probability moves by the error, scaled by h(e) = f'(v) / e:

        p_t = p_baseline * (1 - h(e_out) * dL/de_out)

    with the loss L = 1/2 * sum (e_out - target)^2, so dL/de_out = e_out - target, and for a sigmoid h(e) = 1 - e.
    (Payeur's Methods describe a "squashed version of p_baseline - h(e) * grad L"; this is the unsquashed form
    used in layers.py, OutputLayer.backward. For targets in [0, 1] and p_baseline <= 0.5 it stays in (0, 1).)

    Returns (p, p_t), each shaped like e_out.
    """
    raise NotImplementedError


def hidden_burst_probs(e: Tensor, fb: Tensor, fb_t: Tensor) -> tuple[Tensor, Tensor]:
    """Hidden-layer burst probabilities from the dendritic input (Payeur Methods eq. 7, with alpha = 0, beta = 1).

    fb and fb_t are the feedback arriving at this layer's dendrites in the no-teacher and teacher passes:
    fb = b_{l+1} @ Y_l (and fb_t likewise with b_t). The dendritic potential is

        u = h(e) * fb,      h(e) = f'(v) / e = 1 - e   (sigmoid)

    and the burst probability is p = sigmoid(u). Same for the teacher pass with fb_t.

    Why h(e)? The burst rate multiplies by e (b = p * e), and h divides by e. What is left over is exactly f'(v):
    the activation derivative that backprop needs.

    Returns (p, p_t), each shaped like e.
    """
    raise NotImplementedError


def burst_deltas(
    rates: list[Tensor],
    target: Tensor,
    feedback_weights: list[Tensor],
    p_baseline: float,
) -> list[Tensor]:
    """Top-down feedback passes, returning each layer's error signal.

    1. Output layer: (p, p_t) from `output_burst_probs`; burst rates b = p * e, b_t = p_t * e.
    2. For each hidden layer, from the top down: send the layer above's b and b_t through Y_l to get fb and fb_t,
       then compute (p, p_t) with `hidden_burst_probs` and the burst rates b, b_t.
    3. Every layer's error signal is delta_l = -(b_t,l - b_l).

    The minus sign makes delta follow backprop's convention (delta ~ dL/dv). With symmetric feedback and small
    dendritic inputs, delta_l is approximately proportional to backprop's delta. With no error (target equal to the
    output), every delta is exactly zero.

    Returns [delta_1, ..., delta_L] (aligned with `weights`), delta_l of shape (batch, n_l).
    """
    raise NotImplementedError


def weight_updates(
    deltas: list[Tensor], rates: list[Tensor], lr: float
) -> list[tuple[Tensor, Tensor]]:
    """The plasticity rule: weight and bias changes to *add* to each layer (Payeur Methods eq. 9, no decay).

    In the rule's natural form, a burst above baseline potentiates:

        Delta W_l    = +lr * (b_t,l - b_l)^T @ e_{l-1}  =  -lr * delta_l^T @ e_{l-1}
        Delta bias_l = -lr * sum over the batch of delta_l

    Sum over the batch (do not average), as layers.py does. Usage: W_l += Delta W_l. This is the pure rule: no
    momentum, no weight decay.

    Returns [(Delta W_1, Delta bias_1), ..., (Delta W_L, Delta bias_L)].
    """
    raise NotImplementedError
