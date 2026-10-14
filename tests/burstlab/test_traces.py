"""Tests defining the exponential trace (burstlab/traces.py). AI DISCLOSURE: written by Claude Opus."""

import torch

from burstlab.traces import trace_step

DT, TAU = 1e-3, 10e-3


def test_single_spike_jumps_by_one_over_tau_then_decays():
    trace = torch.zeros(1)
    trace = trace_step(trace, torch.ones(1), DT, TAU)
    torch.testing.assert_close(trace, torch.tensor([1 / TAU]))  # 100 Hz for tau = 10 ms
    for _ in range(5):
        prev = trace
        trace = trace_step(trace, torch.zeros(1), DT, TAU)
        torch.testing.assert_close(trace, prev * (1 - DT / TAU))


def test_euler_step_formula():
    g = torch.Generator().manual_seed(0)
    trace = 50 * torch.rand(4, 7, generator=g)
    spikes = (torch.rand(4, 7, generator=g) < 0.5).float()
    expected = trace + (DT / TAU) * (-trace + spikes / DT)
    torch.testing.assert_close(trace_step(trace, spikes, DT, TAU), expected)


def test_steady_poisson_rate_gives_trace_in_hz():
    """A neuron firing at r Hz has a trace that averages r Hz."""
    g = torch.Generator().manual_seed(0)
    rate, n_neurons, n_steps = 50.0, 1000, 2000
    trace = torch.zeros(n_neurons)
    total = torch.zeros(n_neurons)
    for t in range(n_steps):
        spikes = (torch.rand(n_neurons, generator=g) < rate * DT).float()
        trace = trace_step(trace, spikes, DT, TAU)
        if t >= 200:  # skip the initial rise
            total += trace
    mean = (total / (n_steps - 200)).mean().item()
    assert abs(mean - rate) < 1.0
