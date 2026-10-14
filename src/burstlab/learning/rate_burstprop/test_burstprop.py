"""Tests that define the rate-model Burstprop exercise. See README.md for the order to tackle them.

AI DISCLOSURE: this file was written by Claude Opus.

Run from the repo root:  uv run pytest src/burstlab/learning/rate_burstprop -q
"""

from itertools import pairwise

import burstprop as bp
import pytest
import torch
import torch.nn.functional as F

P_BASELINE = 0.2
ARCHITECTURES = [(20, 30, 10), (20, 30, 25, 10), (784, 100, 10)]


def make_net(sizes, seed):
    """Xavier-initialized weights (as in Payeur's code) and a batch of inputs in [0, 1] with one-hot targets."""
    g = torch.Generator().manual_seed(seed)
    weights, biases = [], []
    for n_in, n_out in pairwise(sizes):
        std = (2.0 / (n_in + n_out)) ** 0.5
        weights.append(torch.randn(n_out, n_in, generator=g, dtype=torch.float64) * std)
        biases.append(torch.zeros(n_out, dtype=torch.float64))
    batch = 16
    x = torch.rand(batch, sizes[0], generator=g, dtype=torch.float64)
    labels = torch.randint(0, sizes[-1], (batch,), generator=g)
    target = F.one_hot(labels, sizes[-1]).to(torch.float64)
    return x, target, weights, biases


def autograd_grads(x, target, weights, biases):
    """Backprop gradients of L = 1/2 * sum over batch and units of (e_L - target)^2."""
    ws = [w.clone().requires_grad_() for w in weights]
    bs = [b.clone().requires_grad_() for b in biases]
    e = x
    for w, b in zip(ws, bs):
        e = torch.sigmoid(F.linear(e, w, b))
    loss = 0.5 * ((e - target) ** 2).sum()
    loss.backward()
    return [w.grad for w in ws], [b.grad for b in bs]


def cosine(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


# --- Step 1: forward pass -------------------------------------------------------------------------------------


@pytest.mark.parametrize("sizes", ARCHITECTURES)
def test_forward_returns_all_layer_rates(sizes):
    x, _, weights, biases = make_net(sizes, seed=0)
    rates = bp.forward(x, weights, biases)
    assert len(rates) == len(sizes)
    assert torch.equal(rates[0], x)
    e = x
    for l, (w, b) in enumerate(zip(weights, biases), start=1):
        e = torch.sigmoid(F.linear(e, w, b))
        torch.testing.assert_close(rates[l], e)


# --- Step 2: output layer -------------------------------------------------------------------------------------


def test_output_without_error_stays_at_baseline():
    e_out = torch.rand(8, 10, dtype=torch.float64)
    p, p_t = bp.output_burst_probs(e_out, target=e_out.clone(), p_baseline=P_BASELINE)
    torch.testing.assert_close(p, torch.full_like(e_out, P_BASELINE))
    torch.testing.assert_close(p_t, torch.full_like(e_out, P_BASELINE))


def test_output_teacher_formula_and_range():
    g = torch.Generator().manual_seed(1)
    e_out = torch.rand(8, 10, generator=g, dtype=torch.float64)
    target = (torch.rand(8, 10, generator=g) > 0.5).to(torch.float64)
    _, p_t = bp.output_burst_probs(e_out, target, P_BASELINE)
    expected = P_BASELINE * (1 - (1 - e_out) * (e_out - target))
    torch.testing.assert_close(p_t, expected)
    assert ((p_t > 0) & (p_t < 1)).all()


def test_output_below_target_bursts_more():
    e_out = torch.full((1, 3), 0.3, dtype=torch.float64)
    target = torch.tensor([[1.0, 0.3, 0.0]], dtype=torch.float64)
    p, p_t = bp.output_burst_probs(e_out, target, P_BASELINE)
    assert p_t[0, 0] > p[0, 0]  # below target: burst more -> potentiate
    assert p_t[0, 1] == p[0, 1]  # on target: no change
    assert p_t[0, 2] < p[0, 2]  # above target: burst less -> depress


# --- Step 3: hidden layers ------------------------------------------------------------------------------------


def test_hidden_zero_feedback_gives_half():
    e = torch.rand(8, 30, dtype=torch.float64)
    zero = torch.zeros_like(e)
    p, p_t = bp.hidden_burst_probs(e, zero, zero)
    torch.testing.assert_close(p, torch.full_like(e, 0.5))
    torch.testing.assert_close(p_t, torch.full_like(e, 0.5))


def test_hidden_formula():
    g = torch.Generator().manual_seed(2)
    e = torch.rand(8, 30, generator=g, dtype=torch.float64)
    fb = torch.randn(8, 30, generator=g, dtype=torch.float64)
    fb_t = torch.randn(8, 30, generator=g, dtype=torch.float64)
    p, p_t = bp.hidden_burst_probs(e, fb, fb_t)
    torch.testing.assert_close(p, torch.sigmoid((1 - e) * fb))
    torch.testing.assert_close(p_t, torch.sigmoid((1 - e) * fb_t))


# --- Step 4: the feedback passes ------------------------------------------------------------------------------


@pytest.mark.parametrize("sizes", ARCHITECTURES)
def test_deltas_have_layer_shapes(sizes):
    x, target, weights, biases = make_net(sizes, seed=3)
    rates = bp.forward(x, weights, biases)
    deltas = bp.burst_deltas(rates, target, weights[1:], P_BASELINE)
    assert [d.shape for d in deltas] == [r.shape for r in rates[1:]]


@pytest.mark.parametrize("sizes", ARCHITECTURES)
def test_no_error_means_no_deltas(sizes):
    """With target == output, the teacher pass equals the no-teacher pass, so the baseline cancels exactly."""
    x, _, weights, biases = make_net(sizes, seed=4)
    rates = bp.forward(x, weights, biases)
    deltas = bp.burst_deltas(rates, rates[-1].clone(), weights[1:], P_BASELINE)
    for d in deltas:
        torch.testing.assert_close(d, torch.zeros_like(d))


# --- Step 5: the plasticity rule, compared with backprop ------------------------------------------------------


@pytest.mark.parametrize("sizes", ARCHITECTURES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_output_layer_update_is_exactly_scaled_backprop(sizes, seed):
    """At the output layer, Burstprop's update is backprop's update times p_baseline: no approximation."""
    x, target, weights, biases = make_net(sizes, seed)
    lr = 0.1
    rates = bp.forward(x, weights, biases)
    deltas = bp.burst_deltas(rates, target, weights[1:], P_BASELINE)
    dW, db = bp.weight_updates(deltas, rates, lr)[-1]
    gW, gb = autograd_grads(x, target, weights, biases)
    torch.testing.assert_close(dW, -lr * P_BASELINE * gW[-1])
    torch.testing.assert_close(db, -lr * P_BASELINE * gb[-1])


@pytest.mark.parametrize("sizes", ARCHITECTURES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hidden_layer_updates_align_with_backprop(sizes, seed):
    """With symmetric feedback, hidden-layer updates point along backprop's (cosine > 0.99).

    Their size shrinks by about sigmoid'(0) = 1/4 per layer below the output: the linear regime of the dendritic
    sigmoid. Work out why on paper; it's the heart of the exercise.
    """
    x, target, weights, biases = make_net(sizes, seed)
    lr = 0.1
    rates = bp.forward(x, weights, biases)
    deltas = bp.burst_deltas(rates, target, weights[1:], P_BASELINE)
    updates = bp.weight_updates(deltas, rates, lr)
    gW, _ = autograd_grads(x, target, weights, biases)
    n_layers = len(weights)
    for l in range(n_layers - 1):
        dW, _ = updates[l]
        assert cosine(dW, -gW[l]) > 0.99, f"layer {l + 1}"
        layers_below_output = n_layers - 1 - l
        predicted = lr * P_BASELINE * 0.25**layers_below_output
        ratio = (dW.norm() / gW[l].norm()).item() / predicted
        assert 0.8 < ratio < 1.05, (
            f"layer {l + 1}: size ratio {ratio:.3f} of the 1/4-per-layer prediction"
        )


def test_updates_sum_over_batch():
    x, target, weights, biases = make_net((20, 30, 10), seed=5)
    rates = bp.forward(x, weights, biases)
    deltas = bp.burst_deltas(rates, target, weights[1:], P_BASELINE)
    once = bp.weight_updates(deltas, rates, lr=0.1)
    doubled = bp.weight_updates(
        [torch.cat([d, d]) for d in deltas], [torch.cat([r, r]) for r in rates], lr=0.1
    )
    for (dW1, db1), (dW2, db2) in zip(once, doubled):
        torch.testing.assert_close(dW2, 2 * dW1)
        torch.testing.assert_close(db2, 2 * db1)


@pytest.mark.parametrize("sizes", ARCHITECTURES)
def test_one_small_step_reduces_the_loss(sizes):
    """The sign convention, end to end: applying W += Delta W must move the outputs toward the target."""
    x, target, weights, biases = make_net(sizes, seed=6)

    def loss(ws, bs):
        return 0.5 * ((bp.forward(x, ws, bs)[-1] - target) ** 2).sum().item()

    rates = bp.forward(x, weights, biases)
    deltas = bp.burst_deltas(rates, target, weights[1:], P_BASELINE)
    updates = bp.weight_updates(deltas, rates, lr=0.01)
    new_weights = [w + dW for w, (dW, _) in zip(weights, updates)]
    new_biases = [b + db for b, (_, db) in zip(biases, updates)]
    assert loss(new_weights, new_biases) < loss(weights, biases)
