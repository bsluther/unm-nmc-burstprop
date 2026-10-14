"""Tests defining the burst-dependent plasticity rules (burstlab/plasticity.py). AI DISCLOSURE: written by Claude Opus.

Sign convention (architecture §12): bursts potentiate.
"""

import torch

from burstlab.plasticity import signed_rule, unsigned_rule

ETA = 0.01


def test_signed_shape_and_outer_product():
    g = torch.Generator().manual_seed(0)
    pre = 100 * torch.rand(1, 6, generator=g)
    b_plus = torch.tensor([[1.0, 0.0, 0.0]])
    b_minus = torch.tensor([[0.0, 1.0, 0.0]])
    dW = signed_rule(pre, b_plus, b_minus, ETA)
    assert dW.shape == (3, 6)
    torch.testing.assert_close(dW[0], ETA * pre[0])  # positive burst: potentiate
    torch.testing.assert_close(dW[1], -ETA * pre[0])  # negative burst: depress
    torch.testing.assert_close(dW[2], torch.zeros(6))  # no burst: no change


def test_signed_positive_burst_strengthens():
    """The sign-convention test: positive burst + active presynaptic neuron => stronger synapse."""
    dW = signed_rule(
        torch.tensor([[20.0]]), torch.tensor([[1.0]]), torch.tensor([[0.0]]), ETA
    )
    assert dW.item() > 0


def test_unsigned_burst_potentiates_single_spike_depresses():
    pre = torch.tensor([[40.0]])
    burst = unsigned_rule(
        pre, bursts=torch.tensor([[1.0]]), events=torch.tensor([[1.0]]), eta=ETA
    )
    single = unsigned_rule(
        pre, bursts=torch.tensor([[0.0]]), events=torch.tensor([[1.0]]), eta=ETA
    )
    silent = unsigned_rule(
        pre, bursts=torch.tensor([[0.0]]), events=torch.tensor([[0.0]]), eta=ETA
    )
    torch.testing.assert_close(burst, torch.tensor([[ETA * 0.5 * 40.0]]))
    torch.testing.assert_close(single, torch.tensor([[-ETA * 0.5 * 40.0]]))
    torch.testing.assert_close(silent, torch.zeros(1, 1))


def test_unsigned_p_baseline_sets_the_balance():
    pre = torch.tensor([[10.0]])
    burst = unsigned_rule(
        pre, torch.tensor([[1.0]]), torch.tensor([[1.0]]), ETA, p_baseline=0.2
    )
    single = unsigned_rule(
        pre, torch.tensor([[0.0]]), torch.tensor([[1.0]]), ETA, p_baseline=0.2
    )
    torch.testing.assert_close(burst, torch.tensor([[ETA * 0.8 * 10.0]]))
    torch.testing.assert_close(single, torch.tensor([[-ETA * 0.2 * 10.0]]))


def test_unsigned_balances_on_average_at_baseline_burst_probability():
    """Events that burst with probability p_baseline give zero weight change on average."""
    g = torch.Generator().manual_seed(0)
    n = 100_000
    events = torch.ones(n, 1)
    bursts = (torch.rand(n, 1, generator=g) < 0.5).float()
    pre = torch.full((n, 1), 30.0)
    dW_mean = unsigned_rule(pre, bursts, events, ETA)  # batch-averaged over n samples
    assert abs(dW_mean.item()) < 0.01 * ETA * 30.0 * 0.5


def test_batch_is_averaged():
    g = torch.Generator().manual_seed(1)
    pre = 50 * torch.rand(4, 5, generator=g)
    b_plus = (torch.rand(4, 3, generator=g) < 0.5).float()
    b_minus = (1 - b_plus) * (torch.rand(4, 3, generator=g) < 0.5).float()
    once = signed_rule(pre, b_plus, b_minus, ETA)
    twice = signed_rule(
        torch.cat([pre, pre]),
        torch.cat([b_plus, b_plus]),
        torch.cat([b_minus, b_minus]),
        ETA,
    )
    torch.testing.assert_close(once, twice)
    expected = ETA * torch.einsum("bi,bj->ij", b_plus - b_minus, pre) / 4
    torch.testing.assert_close(once, expected)
