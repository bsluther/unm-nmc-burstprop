"""How the eligibility trace works out to be an exponential moving average (Stuck et al. 2025,
eq. 4).

AI DISCLOSURE: this file was written by Claude Opus.

traces.trace_step takes one forward-Euler step of tau * dE/dt = -E + S(t). Rearranged, that step
is an exponential moving average (EMA) of the spike train:

    E_new = (1 - a) * E + a * (S / dt),    a = dt / tau

Each step keeps a fraction 1 - a of the trace and mixes in a fraction a of the newest input, where
a spike counts as a pulse of height 1/dt.

The plot drives a Poisson neuron whose rate steps from 20 to 80 Hz, with tau = 100 ms. The trace
(from traces.trace_step) follows the EMA of the rate, lagging the step by about tau. The step is
linear, so feeding it the expected spike count per step, rate * dt, gives that expected trace.

The paper uses tau = 10 ms (§3.1.3). This plot defaults to 100 ms because at 10 ms the trace is a
series of single-spike bumps (rate * tau < 1) and the averaging is hard to see. Try --tau 0.01.

Run:

    uv run python -m burstlab.experiments.trace_demo
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from burstlab.traces import trace_step

DT = 1e-3  # seconds


def run_trace(train: Tensor, tau: float) -> Tensor:
    """The trace at every step of a 1-D train, stepped with traces.trace_step from 0."""
    trace = torch.zeros(1, 1)  # shapes are (batch=1, neurons=1)
    out = torch.zeros(len(train))
    for k, s in enumerate(train):
        trace = trace_step(trace, s.view(1, 1), DT, tau)
        out[k] = trace.item()
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tau", type=float, default=100e-3, help="trace time constant, seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/trace_demo.png"))
    args = parser.parse_args()

    n = 2000
    t = torch.arange(n) * DT
    rate = torch.where(t < 1.0, 20.0, 80.0)
    rng = torch.Generator().manual_seed(args.seed)
    spikes = (torch.rand(n, generator=rng) < rate * DT).float()

    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    trace = run_trace(spikes, args.tau)
    (trace_line,) = ax.plot(t, trace, color="C1", lw=1.2, label="trace (Poisson spikes)")
    (ema_line,) = ax.plot(
        t, run_trace(rate * DT, args.tau), color="k", lw=1.4, label="EMA of the rate"
    )
    (rate_line,) = ax.plot(t, rate, color="0.5", ls="--", lw=1, label="firing rate")

    # Spikes as ticks in a strip below 0.
    top = 1.05 * max(trace.max().item(), rate.max().item())
    strip = 0.12 * top
    ax.vlines(t[spikes > 0], -0.95 * strip, -0.2 * strip, color="k", lw=1)
    ax.set_ylim(-strip, top)
    ax.set_yticks([y for y in ax.get_yticks() if 0 <= y <= top])
    ax.text(
        1.005, -0.55 * strip, "spikes", transform=ax.get_yaxis_transform(), fontsize=8, va="center"
    )
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)

    ax.set_title(
        "The eligibility trace as an exponential moving average of the firing rate "
        f"(tau = {args.tau * 1e3:g} ms)"
    )
    ax.set_ylabel("trace (Hz)")
    ax.set_xlabel("time (s)")
    # Legend in reading order: the input rate, its EMA, then the trace that estimates it.
    ax.legend(handles=[rate_line, ema_line, trace_line], loc="upper left", fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
