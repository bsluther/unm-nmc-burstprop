"""A live, scrolling Stuck et al. 2025, Fig. 2, with sliders for the pre- and postsynaptic rates.

AI DISCLOSURE: this file was written by Claude Opus.

The setup and panels are the same as fig2.py: one postsynaptic neuron, one synapse, and a
sinusoidal dendritic potential v_d(t). Here time keeps running: v_d cycles indefinitely, the window
scrolls left, and a slider change takes effect from that moment on, so the window shows the before
and the after.

    A  the dendritic potential v_d(t)
    B  burst probabilities: unsigned sigmoid(v_d); signed max(0, ±tanh(v_d))
    C  postsynaptic events (Poisson, post_rate)
    D  burst trains generated from C and B
    E  the eligibility trace of a presynaptic Poisson neuron (pre_rate)
    F  the weight change at each step, as stems (the paper's Fig. 2F)
    G  unsigned only: the smoothed rate of weight change (solid) and its expected value (dashed)

Why G is unsigned only. In the signed model v_d sets the sign of every update (only positive bursts
while v_d > 0), so the stems in F already show the direction. In the unsigned model every event
updates the weight: a burst potentiates and a single spike depresses, whatever v_d is. v_d only
tips the balance, through P_B, so the direction shows up on average and not in single updates.

G smooths dw/dt with an exponential filter (time constant --tau-smooth). The pre and post trains
are independent and the trace averages pre_rate (traces.py), so the expected rate is [derived]

    E[dw/dt] = eta * pre_rate * post_rate * (P_B(v_d) - p_baseline)

The dashed line is that expected rate passed through the same filter, so both curves lag v_d by
about tau_smooth. The sign of v_d sets the direction; the rates set the size.

The scale is the same as in fig2.py: our trace is in Hz, so E, F and G are about 1/tau = 100x
larger than the paper's (replication notes §5).

Each step holds at most one event, so keep rate * dt well below 1. The sliders stop at --max-rate
(default 200 Hz, which is 0.2 per step at dt = 1 ms).

The speed slider sets simulated seconds per wall-clock second, on a log scale. At full speed the
simulation runs in real time only if drawing keeps up. The Pause/Play button (or the space bar)
stops and restarts it.

The tau_trace slider (log scale, 2 dt to 500 ms) sets how long the trace remembers a presynaptic
spike. Each spike lifts the trace by 1/tau_trace, and the trace still averages pre_rate, so the
expected curve in G doesn't depend on it. The trace smooths into a rate estimate only when
pre_rate * tau_trace >> 1; at the paper's 10 ms and 50 Hz it's a series of single-spike bumps.

Run:

    uv run python -m burstlab.experiments.fig2_live
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from burstlab import bursts, plasticity
from burstlab.traces import trace_step

# unsigned_rule's default baseline: the burst probability sigmoid(0) of a dendrite at rest.
P_BASELINE = 0.5

# Fixed top of the trace axis (panel E), in Hz, so trace heights compare across time and rates.
# Peaks above it clip, which happens at high pre_rate.
TRACE_YMAX = 500.0

KEYS = (
    "v_d",
    "p_unsigned",
    "p_plus",
    "p_minus",
    "events",
    "b_unsigned",
    "b_plus",
    "b_minus",
    "pre_trace",
    "dw_unsigned",
    "dw_signed",
    "rate_unsigned",
    "rate_expected_unsigned",
)


class Fig2Stepper:
    """One presynaptic Poisson neuron, one synapse, one postsynaptic neuron, advanced one dt at a
    time.

    pre_rate and post_rate are plain attributes. Change them between steps and the next step uses
    them.
    """

    def __init__(
        self,
        dt: float,
        amplitude: float,
        period: float,
        pre_rate: float,
        post_rate: float,
        tau_trace: float,
        eta: float,
        tau_smooth: float,
        seed: int,
    ):
        """
        Args:
            dt: Size of a timestep in seconds.
            amplitude: Peak dendritic potential.
            period: Sine wave period in seconds.
            pre_rate: Presynaptic spike rate in Hz.
            post_rate: Postsynaptic event rate in Hz.
            tau_trace: Time constant of the eligibility trace in seconds.
            eta: Learning rate.
            tau_smooth: Time constant of the filter on the unsigned dw/dt, in seconds.
            seed: Random number generator seed.
        """
        self.dt = dt
        self.amplitude = amplitude
        self.period = period
        self.pre_rate = pre_rate
        self.post_rate = post_rate
        self.tau_trace = tau_trace
        self.eta = eta
        self.tau_smooth = tau_smooth
        self.rng = torch.Generator().manual_seed(seed)
        self.k = 0  # steps taken so far
        self.pre_trace = torch.zeros(1, 1)  # shapes are (batch=1, neurons=1)
        # Smoothed dw/dt of the unsigned model: sampled, and expected.
        self.rate = 0.0
        self.rate_expected = 0.0

    @property
    def t(self) -> float:
        """Simulated time in seconds."""
        return self.k * self.dt

    def step(self) -> dict[str, float]:
        """Advance one timestep and return this step's values, keyed as in KEYS."""
        v_d = torch.tensor([[self.amplitude * math.sin(2 * math.pi * self.t / self.period)]])
        # Always three draws per step, whatever the rates, so a step's random numbers depend only
        # on the seed and the step. Moving a slider then adds or removes events at the margin and
        # leaves the rest in place.
        draws = torch.rand(3, generator=self.rng).view(3, 1, 1)
        events = (draws[0] < self.post_rate * self.dt).float()
        pre_spikes = (draws[1] < self.pre_rate * self.dt).float()
        u = draws[2]

        self.pre_trace = trace_step(self.pre_trace, pre_spikes, self.dt, self.tau_trace)
        b = bursts.sample_unsigned_bursts(events, v_d, u)
        b_plus, b_minus = bursts.sample_signed_bursts(events, v_d, u)
        dw_unsigned = plasticity.unsigned_rule(self.pre_trace, b, events, self.eta, P_BASELINE)
        dw_signed = plasticity.signed_rule(self.pre_trace, b_plus, b_minus, self.eta)

        p_unsigned = bursts.unsigned_burst_prob(v_d).item()
        p_plus, p_minus = (p.item() for p in bursts.signed_burst_probs(v_d))
        # Exponential filters: tau_smooth * dr/dt = -r + input, one Euler step each.
        expected = self.eta * self.pre_rate * self.post_rate * (p_unsigned - P_BASELINE)
        decay = self.dt / self.tau_smooth
        self.rate += decay * (-self.rate + dw_unsigned.item() / self.dt)
        self.rate_expected += decay * (-self.rate_expected + expected)

        self.k += 1
        return {
            "v_d": v_d.item(),
            "p_unsigned": p_unsigned,
            "p_plus": p_plus,
            "p_minus": p_minus,
            "events": events.item(),
            "b_unsigned": b.item(),
            "b_plus": b_plus.item(),
            "b_minus": b_minus.item(),
            "pre_trace": self.pre_trace.item(),
            "dw_unsigned": dw_unsigned.item(),
            "dw_signed": dw_signed.item(),
            "rate_unsigned": self.rate,
            "rate_expected_unsigned": self.rate_expected,
        }


class History:
    """The last n steps of every key, oldest first."""

    def __init__(self, n: int):
        self.buf = {key: np.full(n, np.nan) for key in KEYS}

    def push(self, rows: list[dict[str, float]]):
        m = len(rows)
        for key, buf in self.buf.items():
            buf[:-m] = buf[m:]
            buf[-m:] = [row[key] for row in rows]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.buf[key]


def raster(train: np.ndarray, y: float) -> np.ndarray:
    """y where the 0/1 train has an event, NaN elsewhere (NaN points aren't drawn)."""
    return np.where(train > 0, y, np.nan)


def stems(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points for one line that draws a vertical stem from 0 to y at every nonzero y.

    Each stem is (x, 0) -> (x, y) followed by a NaN, which breaks the line before the next stem.
    """
    i = np.flatnonzero(np.nan_to_num(y))
    xs = np.repeat(x[i], 3)
    ys = np.zeros(3 * len(i))
    ys[1::3] = y[i]
    xs[2::3] = ys[2::3] = np.nan
    return xs, ys


def make_live_figure(
    sim: Fig2Stepper, window: float, interval_ms: int, max_rate: float, speed: float
):
    """Build the figure and its controls.

    Returns (fig, update, widgets). Each call update(frame) advances the simulation by
    speed * interval_ms of simulated time and redraws, unless paused. Keep the widgets referenced:
    matplotlib widgets stop responding once they're garbage-collected.
    """
    n = round(window / sim.dt)
    # paused: whether update() steps. steps_owed: the fraction of a step carried to the next frame,
    # so slow speeds (under one step per frame) still advance.
    state = {"paused": False, "speed": speed, "steps_owed": 0.0}
    x = (np.arange(n) - (n - 1)) * sim.dt  # seconds relative to now
    hist = History(n)
    # Fill the window before the first frame.
    hist.push([sim.step() for _ in range(n)])

    fig, axes = plt.subplots(7, 2, figsize=(13, 10), sharex=True)
    # Plots on the left, controls in a column on the right.
    fig.subplots_adjust(left=0.07, right=0.78, bottom=0.06, top=0.94, hspace=0.25, wspace=0.3)
    lines = {}
    for col, (model, title) in enumerate(
        (("unsigned", "Unsigned Burstprop"), ("signed", "Signed Burstprop"))
    ):
        ax = axes[0, col]
        (lines["v_d", col],) = ax.plot(x, hist["v_d"], color="k")
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylim(-1.1 * sim.amplitude, 1.1 * sim.amplitude)
        ax.set_title(title)
        ax.set_ylabel("A  $v_d$")

        ax = axes[1, col]
        if model == "unsigned":
            (lines["p_unsigned"],) = ax.plot(x, hist["p_unsigned"], color="C0", label="$P_B$")
        else:
            (lines["p_plus"],) = ax.plot(x, hist["p_plus"], color="C2", label="$P_{B+}$")
            (lines["p_minus"],) = ax.plot(x, hist["p_minus"], color="C3", label="$P_{B-}$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("B  burst prob.")
        ax.legend(loc="upper left", fontsize=8)

        raster_style = {"linestyle": "none", "marker": "|", "markersize": 10}
        ax = axes[2, col]
        (lines["events", col],) = ax.plot(x, raster(hist["events"], 0), color="k", **raster_style)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_ylabel("C  events")

        ax = axes[3, col]
        if model == "unsigned":
            (lines["b_unsigned"],) = ax.plot(
                x, raster(hist["b_unsigned"], 0), color="C0", **raster_style
            )
            ax.set_yticks([])
        else:
            (lines["b_plus"],) = ax.plot(x, raster(hist["b_plus"], 0.5), color="C2", **raster_style)
            (lines["b_minus"],) = ax.plot(
                x, raster(hist["b_minus"], -0.5), color="C3", **raster_style
            )
            ax.set_yticks([0.5, -0.5], ["+", "−"])
        ax.set_ylim(-1, 1)
        ax.set_ylabel("D  bursts")

        ax = axes[4, col]
        (lines["pre_trace", col],) = ax.plot(x, hist["pre_trace"], color="C1")
        ax.set_ylim(0, TRACE_YMAX)
        ax.set_ylabel("E  pre trace (Hz)")

        ax = axes[5, col]
        (lines["dw", model],) = ax.plot(*stems(x, hist[f"dw_{model}"]), color="C4", lw=1.2)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.set_ylabel("F  $\\Delta w$ per step")
        # Fixed limits: the largest step a trace at TRACE_YMAX can produce, so F clips exactly when
        # E does. Each step is eta * trace * (burst term); the burst term peaks at
        # max(1 - p_baseline, p_baseline) = 0.5 for unsigned and at 1 for signed.
        burst_term_max = max(1 - P_BASELINE, P_BASELINE) if model == "unsigned" else 1.0
        dw_max = sim.eta * TRACE_YMAX * burst_term_max
        ax.set_ylim(-dw_max, dw_max)

    ax = axes[6, 0]
    (lines["rate"],) = ax.plot(
        x,
        hist["rate_unsigned"],
        color="C4",
        label=f"sample, smoothed ({sim.tau_smooth * 1e3:g} ms)",
    )
    (lines["rate_expected"],) = ax.plot(
        x, hist["rate_expected_unsigned"], color="C4", ls="--", lw=1, label="expected"
    )
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_ylabel("G  $dw/dt$ (1/s)")
    # Fixed limits: the peak expected rate with both rate sliders at max_rate. The expected curve
    # always fits; the noisy sample clips at high rates.
    rate_max = sim.eta * max_rate**2 * (1 / (1 + math.exp(-sim.amplitude)) - P_BASELINE)
    ax.set_ylim(-rate_max, rate_max)
    ax.set_xlabel("time relative to now (s)")
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    # The signed column has no G, so its F carries the time axis.
    axes[6, 1].set_visible(False)
    axes[5, 1].tick_params(labelbottom=True)
    axes[5, 1].set_xlabel("time relative to now (s)")
    axes[0, 0].set_xlim(x[0], x[-1])
    suptitle = fig.suptitle("")

    # Vertical sliders side by side, with the Pause/Play button under them.
    def slider_axes(i):
        return fig.add_axes((0.815 + 0.052 * i, 0.4, 0.015, 0.45))

    vertical = {"orientation": "vertical"}
    # The speed and tau_trace sliders hold log10 of their value; their labels show the value.
    speed_slider = Slider(
        slider_axes(0), "speed", valmin=-2.0, valmax=0.0, valinit=math.log10(speed), **vertical
    )
    rate_style = {"valmin": 0.0, "valmax": max_rate, "valstep": 1.0, "valfmt": "%g Hz", **vertical}
    pre_slider = Slider(slider_axes(1), "pre_rate", valinit=sim.pre_rate, **rate_style)
    post_slider = Slider(slider_axes(2), "post_rate", valinit=sim.post_rate, **rate_style)
    # trace_step is a forward-Euler step, so keep tau_trace at least 2 dt.
    tau_slider = Slider(
        slider_axes(3),
        "tau_trace",
        valmin=math.log10(2 * sim.dt),
        valmax=math.log10(0.5),
        valinit=math.log10(sim.tau_trace),
        **vertical,
    )
    for slider in (speed_slider, pre_slider, post_slider, tau_slider):
        slider.label.set_fontsize(8)
        slider.valtext.set_fontsize(8)
    pause_button = Button(fig.add_axes((0.83, 0.28, 0.14, 0.05)), "Pause")

    def set_speed(log_speed):
        state["speed"] = 10**log_speed
        speed_slider.valtext.set_text(f"{state['speed']:.2g}x")

    def set_tau(log_tau):
        sim.tau_trace = 10**log_tau
        tau_slider.valtext.set_text(f"{sim.tau_trace * 1e3:.3g} ms")

    def toggle_pause(_event=None):
        state["paused"] = not state["paused"]
        pause_button.label.set_text("Play" if state["paused"] else "Pause")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            toggle_pause()

    set_speed(speed_slider.val)
    set_tau(tau_slider.val)
    speed_slider.on_changed(set_speed)
    tau_slider.on_changed(set_tau)
    pre_slider.on_changed(lambda v: setattr(sim, "pre_rate", v))
    post_slider.on_changed(lambda v: setattr(sim, "post_rate", v))
    pause_button.on_clicked(toggle_pause)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_frame):
        if state["paused"]:
            return
        state["steps_owed"] += state["speed"] * interval_ms / 1000 / sim.dt
        steps = int(state["steps_owed"])
        if steps == 0:
            return
        state["steps_owed"] -= steps
        hist.push([sim.step() for _ in range(steps)])
        for col in (0, 1):
            lines["v_d", col].set_ydata(hist["v_d"])
            lines["events", col].set_ydata(raster(hist["events"], 0))
            lines["pre_trace", col].set_ydata(hist["pre_trace"])
        for key in ("p_unsigned", "p_plus", "p_minus"):
            lines[key].set_ydata(hist[key])
        lines["b_unsigned"].set_ydata(raster(hist["b_unsigned"], 0))
        lines["b_plus"].set_ydata(raster(hist["b_plus"], 0.5))
        lines["b_minus"].set_ydata(raster(hist["b_minus"], -0.5))
        for col, model in enumerate(("unsigned", "signed")):
            lines["dw", model].set_data(*stems(x, hist[f"dw_{model}"]))
        lines["rate"].set_ydata(hist["rate_unsigned"])
        lines["rate_expected"].set_ydata(hist["rate_expected_unsigned"])
        suptitle.set_text(
            f"t = {sim.t:.1f} s    pre_rate = {sim.pre_rate:.0f} Hz    "
            f"post_rate = {sim.post_rate:.0f} Hz    tau_trace = {sim.tau_trace * 1e3:.3g} ms"
        )

    update(0)
    return fig, update, (speed_slider, pre_slider, post_slider, tau_slider, pause_button)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--window", type=float, default=2.0, help="visible time, seconds")
    parser.add_argument("--dt", type=float, default=1e-3, help="seconds")
    parser.add_argument("--amplitude", type=float, default=1.0, help="peak dendritic potential")
    parser.add_argument("--period", type=float, default=0.5, help="seconds")
    parser.add_argument(
        "--post-rate", type=float, default=50.0, help="initial postsynaptic event rate, Hz"
    )
    parser.add_argument(
        "--pre-rate", type=float, default=50.0, help="initial presynaptic spike rate, Hz"
    )
    parser.add_argument("--max-rate", type=float, default=200.0, help="slider maximum, Hz")
    parser.add_argument(
        "--tau-trace", type=float, default=10e-3, help="eligibility trace time constant, seconds"
    )
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument(
        "--tau-smooth", type=float, default=50e-3, help="smoothing time constant for G, seconds"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speed", type=float, default=0.1, help="initial simulated seconds per second, 0.01 to 1"
    )
    parser.add_argument("--interval", type=int, default=40, help="milliseconds between frames")
    args = parser.parse_args()

    sim = Fig2Stepper(
        args.dt,
        args.amplitude,
        args.period,
        args.pre_rate,
        args.post_rate,
        args.tau_trace,
        args.eta,
        args.tau_smooth,
        args.seed,
    )
    fig, update, _widgets = make_live_figure(
        sim, args.window, args.interval, args.max_rate, args.speed
    )
    _ani = FuncAnimation(fig, update, interval=args.interval, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
