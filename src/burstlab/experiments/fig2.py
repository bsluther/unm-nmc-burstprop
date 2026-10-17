"""Reproduce Stuck et al. 2025, Fig. 2: the dendritic potential steers the direction of plasticity.

AI DISCLOSURE: this file was written by Claude Opus.

One postsynaptic neuron and one synapse, with no network:

    A  a sinusoidal dendritic potential v_d(t)
    B  burst probabilities: unsigned sigmoid(v_d); signed max(0, ±tanh(v_d))
    C  a 50 Hz Poisson event train (standing in for the soma)
    D  burst trains generated from C and B
    E  the eligibility trace of a presynaptic Poisson neuron
    F  the weight change at each step (with --cumulative, the accumulated weight), for both models

The defaults match what the paper's figure shows: amplitude 1, period 500 ms, 1 s, 50 Hz events.
The paper doesn't state the presynaptic rate; 50 Hz is our choice. Sign convention: bursts
potentiate (architecture §12), so the weight grows while v_d > 0. The paper's Fig. 2F shows the
same.

Scale: our trace is in Hz (a jump of 1/tau = 100 per event). The paper's Fig. 2E trace jumps by
about 1 per event, so panels E and F here are about 1/tau = 100x larger than the paper's
(replication notes §5).

Run (after implementing bursts.py, traces.py, plasticity.py):

    uv run python -m burstlab.experiments.fig2
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from burstlab import bursts, plasticity
from burstlab.traces import trace_step


def simulate(duration, dt, amplitude, period, post_rate, pre_rate, tau_trace, eta, seed):
    g = torch.Generator().manual_seed(seed)
    n = round(duration / dt)
    t = torch.arange(n) * dt
    v_d = amplitude * torch.sin(2 * math.pi * t / period)
    events = (torch.rand(n, generator=g) < post_rate * dt).float()
    pre_spikes = (torch.rand(n, generator=g) < pre_rate * dt).float()
    u = torch.rand(n, generator=g)  # shared by both models, so their bursts line up where they can

    rec = {
        k: torch.zeros(n)
        for k in (
            "pre_trace",
            "b_unsigned",
            "b_plus",
            "b_minus",
            "dw_unsigned",
            "dw_signed",
        )
    }
    pre_trace = torch.zeros(1, 1)  # shapes are (batch=1, neurons=1)
    for k in range(n):
        pre_trace = trace_step(pre_trace, pre_spikes[k].view(1, 1), dt, tau_trace)
        ev, v, uk = events[k].view(1, 1), v_d[k].view(1, 1), u[k].view(1, 1)
        b = bursts.sample_unsigned_bursts(ev, v, uk)
        b_plus, b_minus = bursts.sample_signed_bursts(ev, v, uk)
        rec["dw_unsigned"][k] = plasticity.unsigned_rule(pre_trace, b, ev, eta).item()
        rec["dw_signed"][k] = plasticity.signed_rule(pre_trace, b_plus, b_minus, eta).item()
        rec["pre_trace"][k] = pre_trace.item()
        rec["b_unsigned"][k] = b.item()
        rec["b_plus"][k] = b_plus.item()
        rec["b_minus"][k] = b_minus.item()

    p_unsigned = bursts.unsigned_burst_prob(v_d)
    p_plus, p_minus = bursts.signed_burst_probs(v_d)
    return t, v_d, events, p_unsigned, p_plus, p_minus, rec


def plot(t, v_d, events, p_unsigned, p_plus, p_minus, rec, out, cumulative=False):
    fig, axes = plt.subplots(6, 2, figsize=(10, 11), sharex=True)
    for col, title in enumerate(("Unsigned Burstprop", "Signed Burstprop")):
        ax = axes[0, col]
        ax.plot(t, v_d, color="k")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("A  $v_d$")

        ax = axes[1, col]
        if col == 0:
            ax.plot(t, p_unsigned, color="C0", label="$P_B$")
        else:
            ax.plot(t, p_plus, color="C2", label="$P_{B+}$")
            ax.plot(t, p_minus, color="C3", label="$P_{B-}$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("B  burst prob.")
        ax.legend(loc="upper right", fontsize=8)

        ax = axes[2, col]
        ax.vlines(t[events > 0], 0, 1, color="k", lw=0.8)
        ax.set_yticks([])
        ax.set_ylabel("C  events")

        ax = axes[3, col]
        if col == 0:
            ax.vlines(t[rec["b_unsigned"] > 0], 0, 1, color="C0", lw=0.8)
        else:
            ax.vlines(t[rec["b_plus"] > 0], 0, 1, color="C2", lw=0.8, label="positive")
            ax.vlines(t[rec["b_minus"] > 0], -1, 0, color="C3", lw=0.8, label="negative")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_yticks([])
        ax.set_ylabel("D  bursts")

        ax = axes[4, col]
        ax.plot(t, rec["pre_trace"], color="C1")
        ax.set_ylabel("E  pre trace (Hz)")

        ax = axes[5, col]
        dw = rec["dw_unsigned" if col == 0 else "dw_signed"]
        if cumulative:
            ax.plot(t, dw.cumsum(0), color="C4")
            ax.set_ylabel("F  $\\sum \\Delta w$")
        else:
            nonzero = dw != 0
            ax.vlines(t[nonzero], 0, dw[nonzero], color="C4", lw=1.2)
            ax.set_ylabel("F  $\\Delta w$ per step")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duration", type=float, default=1.0, help="seconds")
    parser.add_argument("--dt", type=float, default=1e-3, help="seconds")
    parser.add_argument("--amplitude", type=float, default=1.0, help="peak dendritic potential")
    parser.add_argument("--period", type=float, default=0.5, help="seconds")
    parser.add_argument("--post-rate", type=float, default=50.0, help="postsynaptic event rate, Hz")
    parser.add_argument("--pre-rate", type=float, default=50.0, help="presynaptic spike rate, Hz")
    parser.add_argument(
        "--tau-trace",
        type=float,
        default=10e-3,
        help="eligibility trace time constant, seconds",
    )
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cumulative", action="store_true", help="plot the accumulated weight in panel F"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/fig2.png"))
    args = parser.parse_args()
    results = simulate(
        args.duration,
        args.dt,
        args.amplitude,
        args.period,
        args.post_rate,
        args.pre_rate,
        args.tau_trace,
        args.eta,
        args.seed,
    )
    plot(*results, args.out, cumulative=args.cumulative)


if __name__ == "__main__":
    main()
