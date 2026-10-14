"""Tests defining burst generation (burstlab/bursts.py). AI DISCLOSURE: written by Claude Opus."""

import pytest
import torch

from burstlab import bursts


def grid():
    return torch.linspace(-4, 4, 81)


# --- Probabilities ---------------------------------------------------------------------------------------------


def test_unsigned_prob_is_sigmoid():
    v = grid()
    torch.testing.assert_close(bursts.unsigned_burst_prob(v), torch.sigmoid(v))
    assert bursts.unsigned_burst_prob(torch.zeros(1)).item() == 0.5


def test_signed_probs_split_tanh_by_sign():
    v = grid()
    p_plus, p_minus = bursts.signed_burst_probs(v)
    torch.testing.assert_close(p_plus, torch.relu(torch.tanh(v)))
    torch.testing.assert_close(p_minus, torch.relu(-torch.tanh(v)))
    assert (p_plus * p_minus == 0).all()  # at most one type is possible
    torch.testing.assert_close(p_plus + p_minus, torch.tanh(v.abs()))


def test_signed_rest_means_no_bursts():
    p_plus, p_minus = bursts.signed_burst_probs(torch.zeros(5))
    assert (p_plus == 0).all() and (p_minus == 0).all()


# --- Sampling --------------------------------------------------------------------------------------------------


def random_inputs(seed, shape=(64, 50), event_rate=0.3):
    g = torch.Generator().manual_seed(seed)
    events = (torch.rand(shape, generator=g) < event_rate).float()
    v_d = 3 * torch.randn(shape, generator=g)
    u = torch.rand(shape, generator=g)
    return events, v_d, u


@pytest.mark.parametrize("seed", range(5))
def test_unsigned_bursts_exactly_where_events_and_u_below_p(seed):
    events, v_d, u = random_inputs(seed)
    b = bursts.sample_unsigned_bursts(events, v_d, u)
    expected = events * (u < torch.sigmoid(v_d)).float()
    assert torch.equal(b, expected)
    assert b.dtype == events.dtype and b.shape == events.shape


@pytest.mark.parametrize("seed", range(5))
def test_signed_bursts_exactly_where_events_and_u_below_p(seed):
    events, v_d, u = random_inputs(seed)
    b_plus, b_minus = bursts.sample_signed_bursts(events, v_d, u)
    burst = events * (u < torch.tanh(v_d.abs())).float()
    assert torch.equal(b_plus, burst * (v_d > 0).float())
    assert torch.equal(b_minus, burst * (v_d < 0).float())
    assert b_plus.dtype == events.dtype and b_minus.shape == events.shape


@pytest.mark.parametrize("seed", range(5))
def test_bursts_only_on_events(seed):
    events, v_d, u = random_inputs(seed)
    assert (bursts.sample_unsigned_bursts(events, v_d, u) <= events).all()
    b_plus, b_minus = bursts.sample_signed_bursts(events, v_d, u)
    assert (b_plus + b_minus <= events).all()  # disjoint, and only on events


def test_burst_fraction_matches_probability():
    """Over many events, the fraction that burst approaches P(burst | event)."""
    g = torch.Generator().manual_seed(0)
    n = 200_000
    events = torch.ones(n)
    v_d = torch.full((n,), 0.8)
    b = bursts.sample_unsigned_bursts(events, v_d, torch.rand(n, generator=g))
    assert abs(b.mean().item() - torch.sigmoid(torch.tensor(0.8)).item()) < 0.005
    b_plus, b_minus = bursts.sample_signed_bursts(
        events, -v_d, torch.rand(n, generator=g)
    )
    assert b_plus.sum() == 0
    assert abs(b_minus.mean().item() - torch.tanh(torch.tensor(0.8)).item()) < 0.005
