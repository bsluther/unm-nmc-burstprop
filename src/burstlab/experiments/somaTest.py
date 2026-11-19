"""Checks for the somatic compartment in soma.py (Stuck et al. 2025, §3.1.2, eqs. 1-3, and eq. 6).

AI DISCLOSURE: this script was written by Claude Opus at Dan's request. It turns the hand checks from
Dan's review session on soma.py into runnable checks, and adds a comparison with Norse.

Part 1 runs the hand checks: one small worked example per function in soma.py, with the answer
worked out by hand.

Part 2 compares soma_step with Norse's lif_box_feed_forward_step, which is the test oracle for the
soma (architecture §3). Both get the same random input currents for many steps, and their events
and voltages must match. Norse writes the Euler step as v + dt * tau_mem_inv * (v_leak - v + x),
which is eq. 3 with tau_mem_inv = 1/tau_s, v_leak = 0 and x = I~_s, so they should agree to
floating-point precision. The one expected difference is the threshold: the paper fires at
v >= v_th, Norse only at v > v_th (gotchas G4). Part 2 shows that case separately.

Run:

    uv run python -m burstlab.experiments.somaTest

It prints PASS or FAIL for each check, and exits with status 1 if any check fails.
"""

import sys

import torch
from norse.torch.functional.lif_box import (
    LIFBoxFeedForwardState,
    LIFBoxParameters,
    lif_box_feed_forward_step,
)

from burstlab.soma import (
    SomaParameters,
    SomaState,
    dimensionless_current,
    dimensionless_voltage,
    initial_soma_state,
    soma_fire,
    soma_integrate,
    soma_reset,
    soma_step,
    somatic_input,
)

DT = 1e-3  # seconds
TAU_S = 10e-3  # seconds

failures: list[str] = []


def check(name: str, got, expected, atol: float = 1e-6) -> None:
    """Print PASS/FAIL for one check, comparing tensors or numbers within atol."""
    got_t = torch.as_tensor(got, dtype=torch.float64)
    exp_t = torch.as_tensor(expected, dtype=torch.float64)
    ok = got_t.shape == exp_t.shape and torch.allclose(got_t, exp_t, atol=atol, rtol=0)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"          expected {exp_t.tolist()}, got {got_t.tolist()}")
        failures.append(name)


def run_check(name: str, fn) -> None:
    """Run fn(); report a crash as a FAIL instead of stopping the whole script."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001  (show every failure, not just the first)
        print(f"  FAIL  {name}: raised {type(e).__name__}: {e}")
        failures.append(name)


# --- Part 1: hand checks ------------------------------------------------------------------------


def hand_checks() -> None:
    print("Part 1: hand checks")

    def initial_state():
        like = torch.ones(3, 4)
        s = initial_soma_state(like)
        check(
            "initial_soma_state: a SomaState of zeros shaped like the input", s.v, torch.zeros(3, 4)
        )
        assert isinstance(s, SomaState), "initial_soma_state should return a SomaState"

    def voltage():
        # E_L = -70 mV, V_threshold = -50 mV
        v = torch.tensor([-70e-3, -50e-3, -60e-3])
        check(
            "dimensionless_voltage: -70, -50, -60 mV -> 0, 1, 0.5",
            dimensionless_voltage(v, -70e-3, -50e-3),
            [0.0, 1.0, 0.5],
        )

    def current():
        # g_L = 10 nS, E_L = -70 mV, V_threshold = -50 mV: 200 pA / (10 nS * 20 mV) = 1
        i = torch.tensor([200e-12, 100e-12])
        check(
            "dimensionless_current: 200, 100 pA -> 1, 0.5",
            dimensionless_current(i, 10e-9, -70e-3, -50e-3),
            [1.0, 0.5],
        )

    def integrate_decay():
        v = soma_integrate(torch.tensor([1.0]), torch.tensor([0.0]), DT, TAU_S)
        check("soma_integrate: v = 1, no input, one step -> 0.9", v, [0.9])

    def integrate_steady_state():
        # With constant input I, v approaches I. No firing here: this is integration only.
        v = torch.tensor([0.0])
        for _ in range(500):
            v = soma_integrate(v, torch.tensor([2.0]), DT, TAU_S)
        check("soma_integrate: constant input 2 for 500 steps -> v close to 2", v, [2.0], atol=1e-4)

    def fire():
        events = soma_fire(torch.tensor([0.5, 1.0, 1.3]), 1.0)
        check(
            "soma_fire: v = 0.5, 1.0, 1.3 -> 0, 1, 1 (fires AT threshold)", events, [0.0, 1.0, 1.0]
        )
        assert events.dtype == torch.float32, f"events should be float32, got {events.dtype}"

    def reset():
        v = soma_reset(torch.tensor([0.5, 1.2]), torch.tensor([0.0, 1.0]), 0.0)
        check("soma_reset: only the neuron that fired goes to v_reset", v, [0.5, 0.0])

    def step_fires():
        # 0.95 + 0.1 * (-0.95 + 2) = 1.055 >= 1, so it fires and resets.
        events, s = soma_step(torch.tensor([2.0]), SomaState(v=torch.tensor([0.95])))
        check("soma_step: v = 0.95, input 2 -> event", events, [1.0])
        check("soma_step: v = 0.95, input 2 -> v reset to 0", s.v, [0.0])
        assert isinstance(s, SomaState), "soma_step should return (events, SomaState)"

    def step_quiet():
        events, s = soma_step(torch.tensor([0.0]), SomaState(v=torch.tensor([0.5])))
        check("soma_step: v = 0.5, no input -> no event", events, [0.0])
        check("soma_step: v = 0.5, no input -> v decays to 0.45", s.v, [0.45])

    def input_convention_b():
        # One postsynaptic neuron, two presynaptic neurons that both spike, weights 0.002 and 0.003.
        # Convention (b): (0.002 + 0.003) / dt = 5.
        events_pre = torch.tensor([[1.0, 1.0]])  # (batch=1, n_pre=2)
        weights = torch.tensor([[0.002, 0.003]])  # (n_post=1, n_pre=2)
        i_tilde = somatic_input(events_pre, weights, DT)
        check("somatic_input: weights 0.002 + 0.003, both spike -> 5.0", i_tilde, [[5.0]])
        # One step from rest then moves v by (dt / tau_s) * 5 = 0.5 = (0.002 + 0.003) / tau_s.
        v = soma_integrate(torch.zeros(1, 1), i_tilde, DT, TAU_S)
        check("somatic_input -> soma_integrate: v jumps by sum(w) / tau_s = 0.5", v, [[0.5]])

    def input_shape():
        events_pre = (torch.rand(8, 5) < 0.3).float()  # (batch=8, n_pre=5)
        weights = torch.randn(3, 5)  # (n_post=3, n_pre=5)
        i_tilde = somatic_input(events_pre, weights, DT)
        check("somatic_input: shape is (batch, n_post)", torch.tensor(i_tilde.shape), [8, 3])

    for name, fn in [
        ("initial_soma_state", initial_state),
        ("dimensionless_voltage", voltage),
        ("dimensionless_current", current),
        ("soma_integrate decay", integrate_decay),
        ("soma_integrate steady state", integrate_steady_state),
        ("soma_fire", fire),
        ("soma_reset", reset),
        ("soma_step fires", step_fires),
        ("soma_step quiet", step_quiet),
        ("somatic_input convention (b)", input_convention_b),
        ("somatic_input shape", input_shape),
    ]:
        run_check(name, fn)


# --- Part 2: comparison with Norse --------------------------------------------------------------


def norse_comparison(steps: int = 1000, batch: int = 16, n: int = 50, seed: int = 0) -> None:
    print(f"\nPart 2: soma_step vs Norse lif_box_feed_forward_step ({steps} steps, {batch}x{n})")
    torch.manual_seed(seed)
    p_norse = LIFBoxParameters()  # tau_mem_inv = 100 (10 ms), v_leak = 0, v_th = 1, v_reset = 0
    p_ours = SomaParameters()

    def random_current():
        # Mean 1.5 with spread 1: a mix of neurons that fire often, rarely, and never.
        i_tilde = 1.5 + torch.randn(steps, batch, n)

        v_ours = torch.zeros(batch, n)
        s_ours = SomaState(v=v_ours)
        s_norse = LIFBoxFeedForwardState(v=v_ours.clone())
        max_dv = 0.0
        mismatched_events = 0
        total_events = 0
        with torch.no_grad():  # gotchas G8: Norse builds an autograd graph otherwise
            for t in range(steps):
                z_ours, s_ours = soma_step(i_tilde[t], s_ours, p_ours, DT)
                z_norse, s_norse = lif_box_feed_forward_step(i_tilde[t], s_norse, p_norse, DT)
                mismatched_events += int((z_ours != z_norse).sum())
                total_events += int(z_ours.sum())
                max_dv = max(max_dv, float((s_ours.v - s_norse.v).abs().max()))
        print(
            f"        {total_events} events in total, mean rate {total_events / (batch * n * steps * DT):.1f} Hz"
        )
        check("random input current: same events every step", mismatched_events, 0)
        check("random input current: same voltages every step (max |diff|)", max_dv, 0.0, atol=1e-5)

    def spike_input():
        # Drive both through somatic_input with Poisson spikes, so the whole path from presynaptic
        # events to postsynaptic events is compared.
        n_pre = 100
        rate = 50.0  # Hz
        weights = 0.0005 * torch.rand(n, n_pre)  # small positive weights so the neurons fire
        s_ours = SomaState(v=torch.zeros(batch, n))
        s_norse = LIFBoxFeedForwardState(v=torch.zeros(batch, n))
        mismatched_events = 0
        total_events = 0
        max_dv = 0.0
        with torch.no_grad():
            for _ in range(steps):
                events_pre = (torch.rand(batch, n_pre) < rate * DT).float()
                i_tilde = somatic_input(events_pre, weights, DT)
                z_ours, s_ours = soma_step(i_tilde, s_ours, p_ours, DT)
                z_norse, s_norse = lif_box_feed_forward_step(i_tilde, s_norse, p_norse, DT)
                mismatched_events += int((z_ours != z_norse).sum())
                total_events += int(z_ours.sum())
                max_dv = max(max_dv, float((s_ours.v - s_norse.v).abs().max()))
        print(
            f"        {total_events} events in total, mean rate {total_events / (batch * n * steps * DT):.1f} Hz"
        )
        check("Poisson spikes through somatic_input: same events every step", mismatched_events, 0)
        check(
            "Poisson spikes through somatic_input: same voltages (max |diff|)",
            max_dv,
            0.0,
            atol=1e-5,
        )

    def threshold_difference():
        # Freeze the dynamics so v stays exactly at threshold: Norse via tau_mem_inv = 0, ours via
        # an infinite tau_s (dt / tau_s = 0). Expected: ours fires (>=), Norse doesn't (>).
        v = torch.ones(1)
        z_ours, _ = soma_step(
            torch.zeros(1), SomaState(v=v), SomaParameters(tau_s=float("inf")), DT
        )
        z_norse, _ = lif_box_feed_forward_step(
            torch.zeros(1),
            LIFBoxFeedForwardState(v=v.clone()),
            LIFBoxParameters(tau_mem_inv=torch.tensor(0.0)),
            DT,
        )
        print(
            "        expected difference (gotchas G4): at exactly v = v_th the paper fires, Norse doesn't"
        )
        check("v exactly at threshold: ours fires (paper's >=)", z_ours, [1.0])
        check("v exactly at threshold: Norse doesn't (strict >)", z_norse, [0.0])

    for name, fn in [
        ("Norse: random input current", random_current),
        ("Norse: Poisson spike input", spike_input),
        ("Norse: threshold difference", threshold_difference),
    ]:
        run_check(name, fn)


def main() -> None:
    hand_checks()
    norse_comparison()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: " + "; ".join(failures))
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
