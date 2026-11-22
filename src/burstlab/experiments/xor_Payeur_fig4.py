"""Recreate Payeur et al. 2021, Fig. 4 (spiking XOR) with our Stuck et al. 2025 neurons and rules.

AI DISCLOSURE: this file was written by Claude Opus at Dan's request. It uses the human-written
functions in bursts.py, plasticity.py, soma.py and traces.py unchanged. The dendrite step and the
two-compartment step below are stand-ins written for this experiment (see "Two-compartment
neuron" below), not the students' version.
Summary of the discussion so far: the Fig. 2 pieces are done and working: the signed and unsigned
plasticity rules (plasticity.py), the signed and unsigned burst probabilities and burst sampling
(bursts.py), and the exponential trace (traces.py). Daniel then implemented the somatic compartment
of Stuck §3.1.2 (soma.py, from a Claude-written skeleton), with checks against Norse in
experiments/somaTest.py, and chose convention (b) for the somatic input (a spike is a pulse of
height 1/dt). Dan next asked for this file: a network with one hidden layer, trained with
Burstprop on XOR like Payeur et al. 2021 (docs/papers/payeur2021.pdf), reproducing the results
of their Fig. 4 for both of our rules, unsigned and signed. Claude read Payeur's XOR section and
Methods ("Spiking XOR gate", "Learning protocol"), Stuck §3.1-3.9, the burstlab design notes,
and the four modules, and wrote this script.

What Payeur's Fig. 4 shows
--------------------------
Five populations ("pools") of pyramidal neurons: input 1, input 2, hidden 1, hidden 2, output.
Feedforward connections carry events to somas; feedback from the output carries bursts to the
hidden dendrites, mainly excitatory onto hidden 1 (+) and mainly inhibitory onto hidden 2 (-).
Input "0" is a low event rate and input "1" a high one. A teacher pushes the output dendrites
up (more bursts) when the output should be high and down when it should be low. Panels:

    b  input event rates (ER) for the four input conditions, presented one after another
    c  output ER before and after learning (mean +- 2 s.d. over 5 initializations)
    d  during learning: the teaching input and the output burst probability (BP)
    e  during learning: the hidden BPs follow the output burst rate (BR), inverted for hidden 2
    f  hidden ER before and after learning: the two hidden pools diverge

This script makes the same panels, plus a learning curve, once per rule:

    runs/xor_fig4_unsigned.png, runs/xor_fig4_signed.png, runs/xor_fig4_compare.png

What is ours (Stuck 2025) and what is Payeur's
----------------------------------------------
Ours, from the burstlab modules:
  - soma: dimensionless LIF, eqs. 2-3 (soma.soma_step), input I~_s = sum_j w_ij S_j / dt
    (soma.somatic_input, Dan's convention (b))
  - dendrite: leaky integrator, eq. 5, no threshold (below)
  - bursts: unsigned sigmoid(v_d) or signed max(0, +-tanh(v_d)) (bursts.py), with the signed
    model's dendritic reset after a burst (Stuck §3.1.4)
  - traces: eligibility trace E~ and rate estimate E- (traces.trace_step, eq. 4, in Hz)
  - hidden regularizer: eq. 10 with our sign (architecture §12), R = +2 gamma_R (E^_h - E-),
    and Stuck's teacher, eq. 9, as an option (--teacher stuck), T = +2 gamma_T (E^ - E-)
  - plasticity: plasticity.signed_rule / plasticity.unsigned_rule (eqs. 13-14, our sign)

Payeur's, from the XOR Methods:
  - pools of pyramidal neurons, two hidden pools with opposite-sign feedback from the output
  - the teacher (the default): Payeur's XOR error current, c / (E- - e_min) or
    -c / (e_max - E-) (see teacher() below). Stuck's squared-error teacher pushes weakly on an
    output that should fire but barely does, and runs with it got stuck there.
  - private background noise on every soma, which keeps every neuron firing a little
  - learning protocol: each example starts with a prediction interval (no teacher, plasticity
    off), then a learning interval with the teacher on and plasticity gated on (M(t) = 1 in
    Payeur eq. 4). The four examples are presented in order, for 500 epochs.

Simplifications, all [derived] choices to revisit:
  - Durations are shortened to suit our 10 ms time constants: Payeur's 8 s examples exist to let
    a tau_avg = 2 s moving average settle. Our states reset between examples (Stuck §3.9.1) and
    settle in about 5 tau = 50 ms. Defaults: 50 ms prediction, 200 ms learning, so 1 s of
    simulated time per epoch. (With 100 ms of learning, 500 epochs weren't enough.)
  - Pools are 50 neurons (Payeur: 500), fully connected between pools with signed, plastic
    weights (Stuck §3.2), instead of sparse excitatory/inhibitory synapses.
  - Input pools are Poisson spike sources (Stuck §3.8), not driven LIF pools.
  - Feedback weights are fixed: +beta onto hidden 1, -beta onto hidden 2 (Payeur's +/- pattern;
    like feedback alignment, they don't learn). The hidden-to-output weights start with the same
    signs, so the feedback is aligned with them at the start.
  - Unsigned model only: an output neuron with no error still bursts with p = 0.5, so the hidden
    dendrites get a baseline feedback that carries no error (replication notes §2). Payeur also
    had to compensate the feedback onto hidden 2, with a constant dendritic current. We do the
    same for both hidden pools: before training, we measure each hidden neuron's mean feedback
    with no teacher, and inject minus that as a constant dendritic current (--no-compensation
    turns it off). The signed model emits no bursts at zero error, so it needs none.

Results with the defaults (5 seeds, 500 epochs; runs/xor_fig4_summary.json):
  - signed: XOR solved in 3 of 5 seeds. In those, hidden 1's response to (1, 1) falls and
    hidden 2's rises, the divergence of Payeur's Fig. 4f.
  - unsigned: solved in 0 of 5. (1, 1) does come down, but each run ends up firing for only one
    of the two 'true' inputs.
  - The failed runs fail the same way: the network locks onto one input, and the output for the
    other 'true' input falls to about 0 Hz. A neuron that doesn't fire can't burst, so nothing
    then pulls it back up. Seeds 1 and 2 did this under every setting tried (noise, teacher
    strength, feedback strength, slower learning over 1000 epochs).

Run:

    uv run python -m burstlab.experiments.xor_Payeur_fig4                 # both rules, 5 seeds
    uv run python -m burstlab.experiments.xor_Payeur_fig4 --rule signed --seeds 1 --epochs 100

One run (one rule, one seed, 500 epochs) takes a few minutes on one CPU core. --jobs runs that
many in parallel (default: up to 4). Figures and a JSON summary go to --out-dir (default runs/).
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from burstlab import bursts, plasticity
from burstlab.soma import SomaParameters, SomaState, initial_soma_state, soma_step, somatic_input
from burstlab.traces import trace_step

# The four XOR conditions in Payeur's order, and the desired output for each.
CONDITIONS = [(0, 0), (1, 0), (0, 1), (1, 1)]
XOR_TARGET = [0, 1, 1, 0]
POOLS = ["input 1", "input 2", "hidden 1", "hidden 2", "output"]


# --- Configuration --------------------------------------------------------------------------------
@dataclass(frozen=True)
class XorConfig:
    """Every setting of one run. Times in seconds, rates in Hz, voltages and currents
    dimensionless (Stuck eq. 3: rest 0, threshold 1)."""

    rule: str = "signed"  # "signed" or "unsigned"
    n_per_pool: int = 50  # neurons per population (Payeur: 500)
    dt: float = 1e-3  # Stuck §3.9: forward Euler at 1 ms
    tau_s: float = 10e-3  # soma, Stuck §3.1.2
    tau_d: float = 10e-3  # dendrite, Stuck §3.1.4
    tau_trace: float = 10e-3  # eligibility trace and rate estimate, Stuck §3.1.3 and §3.4.1
    # Input pools: "0" is a low Poisson rate and "1" a high one (Payeur: about 2 and 10 Hz).
    rate_low: float = 10.0
    rate_high: float = 50.0
    # Output rates for XOR false / true: the teacher's targets (or Payeur's e_min, e_max). Their
    # midpoint is the decision threshold (the dotted line in Payeur's Fig. 4c).
    target_low: float = 10.0
    target_high: float = 50.0
    teacher: str = "payeur"  # "payeur" (Payeur's XOR error current) or "stuck" (eq. 9)
    gamma_t: float = 5e-3  # "stuck" teacher strength (Stuck: 1e-3 with a 20-200 Hz range)
    c_teach: float = 2.0  # "payeur" teacher strength, Hz (Payeur: c of about 1 nA Hz)
    t_max: float = 3.0  # "payeur" teacher: clip |T| here, since c / (rate - e_min) diverges
    gamma_r: float = 1e-5  # hidden regularizer strength (Stuck §3.4.2)
    target_hidden: float = 20.0  # hidden target rate for the regularizer (Stuck §3.4.2)
    eta: float = 2e-8  # learning rate per step, with traces in Hz (replication notes §5)
    p_baseline: float = 0.5  # B_P in the unsigned rule (Stuck eq. 14)
    # Initial weights, in the units of soma.somatic_input: one spike of weight w moves v by
    # w / tau_s. Feedforward weights are Normal(mean, std); feedback is fixed at +-beta. The
    # input weights are broad (std >> mean, like Stuck's zero-mean init), so hidden neurons
    # differ in how much they prefer input 1 or input 2. The output weights start the output
    # between e_min and e_max for every input, as Payeur's did (Methods, "Initialization").
    w_in_mean: float = 1e-4
    w_in_std: float = 1.5e-3
    w_out_mean: float = 7e-4  # hidden 1 -> output starts at +mean
    w_out_inh: float = 0.8  # hidden 2 -> output starts at -w_out_inh * mean
    w_out_std: float = 3.5e-4
    beta: float = 2e-4  # feedback weight magnitude: + onto hidden 1, - onto hidden 2
    # Private background noise onto every hidden and output soma (Payeur Methods, "Noise"): a
    # white-noise current with this mean and s.d. (dimensionless; at 0.3 / 0.4 a neuron with no
    # input fires at about 2.5 Hz). It keeps every neuron firing a little, so an output that
    # should fire but has fallen silent can still burst and recover. Without it, runs got stuck
    # with one 'true' condition at 0 Hz: no events, so no bursts, so no learning.
    noise_mean: float = 0.3
    noise_std: float = 0.4
    compensate: bool = True  # unsigned only: cancel the zero-error feedback (module docstring)
    # Protocol (Payeur's Methods, shortened; see the module docstring).
    t_predict: float = 0.05  # per example: teacher off, plasticity off
    t_learn: float = 0.20  # per example: teacher on, plasticity on (Payeur's M(t) = 1)
    t_test: float = 0.5  # per condition, for the before/after measurements
    epochs: int = 500  # Payeur: 500
    seed: int = 0


# --- Two-compartment neuron -----------------------------------------------------------------------
# Stand-in for the student-written two-compartment step (architecture §5, phase 1 step 2). It
# composes soma.soma_step (Dan's) with the dendrite of eq. 5 and bursts.py's samplers. When the
# students' version lands, replace pyramidal_step with it and check this script still runs.
class PyramidalState(NamedTuple):
    """State of a population of two-compartment neurons between timesteps.

    soma:    SomaState, the dimensionless somatic potential (soma.py).
    v_d:     dimensionless dendritic potential V~_d (Stuck eq. 5), shape (batch, n).
    trace:   exponential trace of this population's events, in Hz (Stuck eq. 4). It is both the
             presynaptic eligibility trace E~ (for the next layer's plasticity) and the rate
             estimate E- (for the teacher and regularizer), as replication notes §6 suggests.
    bursts:  this step's burst train, for the feedback to the layer below on the next step
             (Stuck §3.9: feedback is read from the previous step). For the signed model it is
             B+ - B- (Stuck eq. 8), so it can be -1, 0 or 1.
    """

    soma: SomaState
    v_d: Tensor
    trace: Tensor
    bursts: Tensor


def initial_pyramidal_state(batch: int, n: int) -> PyramidalState:
    """Everything at rest (Stuck §3.9.1: all states are reset at the start of each sample)."""
    zeros = torch.zeros(batch, n)
    return PyramidalState(initial_soma_state(zeros), zeros, zeros, zeros)


def dendrite_integrate(v_d: Tensor, i_d: Tensor, dt: float, tau_d: float) -> Tensor:
    """One forward-Euler step of tau_d * dV~_d/dt = -V~_d + I~_d (Stuck eq. 5). No threshold."""
    return v_d + (dt / tau_d) * (-v_d + i_d)


def pyramidal_step(
    i_soma: Tensor,
    i_dend: Tensor,
    state: PyramidalState,
    cfg: XorConfig,
    gen: torch.Generator,
) -> tuple[dict[str, Tensor], PyramidalState]:
    """Advance a population of two-compartment neurons by one timestep.

    i_soma: (batch, n) dimensionless somatic input I~_s (from soma.somatic_input).
    i_dend: (batch, n) dimensionless dendritic input I~_d (feedback + teacher or regularizer).

    Order within a step: the soma integrates and fires (soma.soma_step), the dendrite integrates
    (eq. 5), each event becomes a burst with a probability set by the new v_d (bursts.py), and
    in the signed model a burst resets the dendrite to 0 (Stuck §3.1.4). The dendrite never
    acts on the soma: it only labels events as bursts.

    Returns a dict of this step's signals (events, bursts, the burst probability and, for the
    signed model, the two burst trains) and the new state.
    """
    # Background noise (XorConfig.noise_*): a Gaussian current whose per-step s.d. is scaled by
    # sqrt(tau_s / dt), so the fluctuations of v don't depend on dt.
    noise_scale = cfg.noise_std * (cfg.tau_s / cfg.dt) ** 0.5
    
    #(batch, n_neurons) dimensionless somatic input I~_s for this step
    i_soma = i_soma + cfg.noise_mean + noise_scale * torch.randn(i_soma.shape, generator=gen)

    #Here soma_step takes in our the initial input (i_soma), our state (in this case state: PyramidalState.soma or state.soma ),
    #  our soma parameters (tau_s: float = 10e-3, v_th: float = 1.0, v_reset: float = 0.0), and then dt (1e-3 from Stuck)
    #  Inside of soma_step we run these methods call:
    # v = soma_integrate(state.v, i_tilde, dt, p.tau_s) 
    # events = soma_fire(v, p.v_th)
    # v_after_reset = soma_reset(v, events, p.v_reset)
    
    # return events, SomaState(v=v_after_reset)
    events, soma_state = soma_step(i_soma, state.soma, SomaParameters(tau_s=cfg.tau_s), cfg.dt)

    # Here is where it calls the dendrite integrate function from above that is 
    # eq. 5 from Stuck's paper
    v_d = dendrite_integrate(state.v_d, i_dend, cfg.dt, cfg.tau_d)
    
    u = torch.rand(v_d.shape, generator=gen)
    out = {"events": events}
    #Our unsigned rule here
    if cfg.rule == "unsigned":
        b = bursts.sample_unsigned_bursts(events, v_d, u)
        out["bursts"] = b
        out["p"] = bursts.unsigned_burst_prob(v_d)
        signed_train = b
    #Our signed rule here
    else:
        b_plus, b_minus = bursts.sample_signed_bursts(events, v_d, u)
        p_plus, p_minus = bursts.signed_burst_probs(v_d)
        out["bursts_plus"], out["bursts_minus"] = b_plus, b_minus
        out["p"] = p_plus - p_minus  # net signed burst probability, in [-1, 1]
        signed_train = b_plus - b_minus
        v_d = torch.where((b_plus + b_minus) > 0, 0.0, v_d)  # reset to baseline after a burst
    #Our trace here
    trace = trace_step(state.trace, events, cfg.dt, cfg.tau_trace)
    return out, PyramidalState(soma_state, v_d, trace, signed_train)


# --- Network --------------------------------------------------------------------------------------
@dataclass
class Network:
    """The 2-2-1-pool network of Payeur's Fig. 4a, one tensor per connection.

    W1: (2N, 2N) input pools -> hidden pools (somatic, events, plastic)
    W2: (N, 2N)  hidden pools -> output pool (somatic, events, plastic)
    B:  (2N, N)  output pool -> hidden pools (dendritic, bursts, fixed; +beta onto hidden 1 and
        -beta onto hidden 2)
    i_comp: (1, 2N) constant dendritic current onto the hidden neurons (unsigned compensation)
    The layout is nn.Linear's: rows are postsynaptic neurons, columns presynaptic.
    """

    W1: Tensor
    W2: Tensor
    B: Tensor
    i_comp: Tensor


def init_network(cfg: XorConfig, gen: torch.Generator) -> Network:
    """Random initial weights (see XorConfig for the scales)."""
    n = cfg.n_per_pool
    w1 = cfg.w_in_mean + cfg.w_in_std * torch.randn(2 * n, 2 * n, generator=gen)
    w2 = cfg.w_out_std * torch.randn(n, 2 * n, generator=gen)
    w2[:, :n] += cfg.w_out_mean  # hidden 1 -> output: mainly excitatory
    w2[:, n:] -= cfg.w_out_inh * cfg.w_out_mean  # hidden 2 -> output: mainly inhibitory
    b = torch.full((2 * n, n), cfg.beta)
    b[n:] *= -1  # Payeur Fig. 4a: + onto hidden 1, - onto hidden 2
    return Network(w1, w2, b, torch.zeros(1, 2 * n))


def input_rates(cond: tuple[int, int], cfg: XorConfig) -> Tensor:
    """(1, 2N) Poisson rates of the two input pools for one XOR condition."""
    n = cfg.n_per_pool
    r = [cfg.rate_high if x else cfg.rate_low for x in cond]
    return torch.cat([torch.full((1, n), r[0]), torch.full((1, n), r[1])], dim=1)

#Here is the teacher that Payeur showed in his figure 4 discription.
#I know that we give it to our output dendrites but need to review the Payeur paper
# a little more to ge a better grasp
#TODO: Look through Payeur to make sure this makes sense
def teacher(rate: Tensor, desired: int, cfg: XorConfig) -> Tensor:
    """Teaching input to the output dendrites, from each output neuron's rate estimate E- (Hz).

    "stuck": Stuck eq. 9 with our sign, T = +2 gamma_T (E^ - E-), a squared-error gradient with
        target E^ = target_high or target_low.
    "payeur": Payeur's XOR error current (Methods, "Error generation"), the gradient of a cross
        entropy on the rate rescaled to [e_min, e_max] = [target_low, target_high]:
            desired 1:  T = +c / (E- - e_min)        desired 0:  T = -c / (e_max - E-)
        and T = 0 once E- is within 1 Hz of the desired end. It grows without bound as an output
        that should fire falls silent, which keeps the output from dying for the 'true' inputs.
        We clip it to +-t_max (Payeur assumes e_min < E- < e_max, so never needed to).

    The burst rule still needs output events: a neuron that doesn't fire can't burst, whatever T is.
    """
    if cfg.teacher == "stuck":
        target = cfg.target_high if desired else cfg.target_low
        return 2 * cfg.gamma_t * (target - rate)
    e_min, e_max = cfg.target_low, cfg.target_high
    if desired:
        t = cfg.c_teach / (rate - e_min).clamp(min=1e-6)
        t = torch.where(rate >= e_max - 1.0, 0.0, t)
    else:
        t = -cfg.c_teach / (e_max - rate).clamp(min=1e-6)
        t = torch.where(rate <= e_min + 1.0, 0.0, t)
    return t.clamp(-cfg.t_max, cfg.t_max)


def run_example(
    net: Network,
    cond_idx: int,
    cfg: XorConfig,
    gen: torch.Generator,
    n_steps: int,
    learn_from: int | None,
    record: bool = False,
) -> dict[str, Tensor]:
    """Present one XOR condition for n_steps, starting from rest.

    learn_from: the step at which the learning interval starts (teacher on and plasticity on),
    or None to keep both off throughout (the prediction-only, "before"/"after" measurement).
    With record=True, returns per-step pool averages for the figure.
    """
    n = cfg.n_per_pool
    rates = input_rates(CONDITIONS[cond_idx], cfg)
    hid = initial_pyramidal_state(1, 2 * n)
    out = initial_pyramidal_state(1, n)
    in_trace = torch.zeros(1, 2 * n)
    keys = ["er_in1", "er_in2", "er_h1", "er_h2", "er_out", "p_h1", "p_h2", "p_out",
            "br_out", "teacher"]  # fmt: skip
    rec = {k: torch.zeros(n_steps) for k in keys} if record else {}
    out_events = torch.zeros(1, n)

    for k in range(n_steps):
        learning = learn_from is not None and k >= learn_from
        # Input pools: Poisson events (Stuck §3.8), and their eligibility trace for W1.
        in_events = (torch.rand(rates.shape, generator=gen) < rates * cfg.dt).float()
        in_trace = trace_step(in_trace, in_events, cfg.dt, cfg.tau_trace)

        # Hidden pools. Somatic input: this step's input events (eq. 6). Dendritic input: last
        # step's output bursts through B (eqs. 7-8, divided by dt like the somatic input), the
        # unsigned compensation, and while learning the regularizer (eqs. 10, 12).
        i_fb = (out.bursts @ net.B.T) / cfg.dt + net.i_comp
        if learning:
            i_fb = i_fb + 2 * cfg.gamma_r * (cfg.target_hidden - hid.trace)
        h_sig, hid_new = pyramidal_step(somatic_input(in_events, net.W1, cfg.dt), i_fb, hid,
                                        cfg, gen)  # fmt: skip

        # Output pool. Somatic input: this step's hidden events. Dendritic input: the teacher
        # while learning (eq. 9 with our sign, eq. 11), nothing otherwise.
        teach = teacher(out.trace, XOR_TARGET[cond_idx], cfg) if learning else torch.zeros(1, n)
        o_sig, out_new = pyramidal_step(somatic_input(h_sig["events"], net.W2, cfg.dt), teach,
                                        out, cfg, gen)  # fmt: skip

        if learning:
            # Plasticity, gated on only in the learning interval (Payeur eq. 4, M(t) = 1).
            # Presynaptic traces are taken after this step's events, as in fig2.py.
            # W1 learns from the input traces and the hidden bursts, W2 from the hidden traces
            # and the output bursts. Each rule averages over the batch (here 1) and returns dW.
            if cfg.rule == "unsigned":
                b_p = cfg.p_baseline
                net.W1 += plasticity.unsigned_rule(
                    in_trace, h_sig["bursts"], h_sig["events"], cfg.eta, b_p
                )
                net.W2 += plasticity.unsigned_rule(
                    hid_new.trace, o_sig["bursts"], o_sig["events"], cfg.eta, b_p
                )
            else:
                net.W1 += plasticity.signed_rule(
                    in_trace, h_sig["bursts_plus"], h_sig["bursts_minus"], cfg.eta
                )
                net.W2 += plasticity.signed_rule(
                    hid_new.trace, o_sig["bursts_plus"], o_sig["bursts_minus"], cfg.eta
                )

        if record:
            ev_h = h_sig["events"][0]
            rec["er_in1"][k] = in_events[0, :n].mean() / cfg.dt
            rec["er_in2"][k] = in_events[0, n:].mean() / cfg.dt
            rec["er_h1"][k] = ev_h[:n].mean() / cfg.dt
            rec["er_h2"][k] = ev_h[n:].mean() / cfg.dt
            rec["er_out"][k] = o_sig["events"].mean() / cfg.dt
            rec["p_h1"][k] = h_sig["p"][0, :n].mean()
            rec["p_h2"][k] = h_sig["p"][0, n:].mean()
            rec["p_out"][k] = o_sig["p"].mean()
            rec["br_out"][k] = out_new.bursts.mean() / cfg.dt  # signed: (B+ - B-) rate
            rec["teacher"][k] = teach.mean()
        out_events = out_events + o_sig["events"]
        hid, out = hid_new, out_new

    rec["out_rate"] = out_events.mean() / (n_steps * cfg.dt)  # mean output ER, Hz
    return rec


def calibrate_compensation(net: Network, cfg: XorConfig, gen: torch.Generator) -> None:
    """Unsigned model: set net.i_comp to minus each hidden neuron's mean zero-error feedback.

    Runs every condition with no teacher and no plasticity and averages the feedback current
    (out.bursts @ B.T / dt) that reaches each hidden neuron. That is the part of the feedback
    that carries no error, the p = 0.5 baseline of replication notes §2. Payeur compensated
    hidden pool 2 the same way, with a constant dendritic current (Methods, "Initialization of
    the network").
    """
    n = cfg.n_per_pool
    n_steps = round(cfg.t_test / cfg.dt)
    total = torch.zeros(1, 2 * n)
    for c in range(4):
        out = initial_pyramidal_state(1, n)
        hid = initial_pyramidal_state(1, 2 * n)
        rates = input_rates(CONDITIONS[c], cfg)
        for _ in range(n_steps):
            in_events = (torch.rand(rates.shape, generator=gen) < rates * cfg.dt).float()
            fb = (out.bursts @ net.B.T) / cfg.dt
            total += fb
            h_sig, hid = pyramidal_step(somatic_input(in_events, net.W1, cfg.dt), fb, hid,
                                        cfg, gen)  # fmt: skip
            _, out = pyramidal_step(somatic_input(h_sig["events"], net.W2, cfg.dt),
                                    torch.zeros(1, n), out, cfg, gen)  # fmt: skip
    net.i_comp = -total / (4 * n_steps)


def measure(net: Network, cfg: XorConfig, gen: torch.Generator) -> dict[str, Tensor]:
    """The "before"/"after" response: each condition in turn, no teacher, plasticity off.

    Returns per-step pool averages, concatenated over the four conditions (Payeur Fig. 4b, c, f).
    """
    n_steps = round(cfg.t_test / cfg.dt)
    recs = [run_example(net, c, cfg, gen, n_steps, None, record=True) for c in range(4)]
    return {k: torch.cat([r[k] for r in recs]) for k in recs[0] if k != "out_rate"}


def condition_rates(net: Network, cfg: XorConfig, gen: torch.Generator) -> list[float]:
    """Mean output ER for each condition over t_test, no teacher, plasticity off."""
    n_steps = round(cfg.t_test / cfg.dt)
    return [run_example(net, c, cfg, gen, n_steps, None)["out_rate"].item() for c in range(4)]


def train(cfg: XorConfig, log_every: int = 50) -> dict:
    """One full run for one seed: initialize, measure "before", train, measure "after".

    The four examples are presented in Payeur's fixed order every epoch. Each example is a
    prediction interval followed by a learning interval (Payeur Methods, "Learning protocol").
    """
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    net = init_network(cfg, gen)
    if cfg.rule == "unsigned" and cfg.compensate:
        calibrate_compensation(net, cfg, gen)

    before = measure(net, cfg, gen)
    n_pred = round(cfg.t_predict / cfg.dt)
    n_steps = n_pred + round(cfg.t_learn / cfg.dt)
    history = []  # per epoch: the output ER of each example's prediction interval
    during = None
    for epoch in range(cfg.epochs):
        record = epoch == min(cfg.epochs - 1, 10)  # an early epoch for panels d and e
        recs = [run_example(net, c, cfg, gen, n_steps, n_pred, record=record) for c in range(4)]
        if record:
            during = {k: torch.cat([r[k] for r in recs]) for k in recs[0] if k != "out_rate"}
        history.append([r["out_rate"].item() for r in recs])
        if log_every and (epoch + 1) % log_every == 0:
            rates = ", ".join(f"{x:5.1f}" for x in history[-1])
            print(f"  [{cfg.rule} seed {cfg.seed}] epoch {epoch + 1:4d}  output ER: {rates} Hz")
    after = measure(net, cfg, gen)
    final = condition_rates(net, cfg, gen)
    return {"before": before, "after": after, "during": during, "history": history,
            "final_rates": final, "solved": is_solved(final, cfg), "net": net}  # fmt: skip


def is_solved(rates: list[float], cfg: XorConfig) -> bool:
    """XOR is solved when the 'true' conditions are above the threshold and the 'false' below."""
    thr = (cfg.target_low + cfg.target_high) / 2
    return all((r > thr) == bool(t) for r, t in zip(rates, XOR_TARGET))


# --- Figures --------------------------------------------------------------------------------------
def binned(x: Tensor, steps_per_bin: int) -> Tensor:
    """Average a per-step signal over bins of steps_per_bin steps (Payeur's population bins)."""
    n = (x.shape[-1] // steps_per_bin) * steps_per_bin
    return x[..., :n].reshape(*x.shape[:-1], -1, steps_per_bin).mean(-1)


def condition_axis(ax, total_time: float) -> None:
    """Label the x axis with the four conditions, as in Payeur's Fig. 4."""
    w = total_time / 4
    ax.set_xticks([w * (i + 0.5) for i in range(4)], [str(c) for c in CONDITIONS])
    for i in range(1, 4):
        ax.axvline(w * i, color="0.85", lw=0.8)
    ax.set_xlim(0, total_time)


def plot_rule(
    runs: list[dict], cfg: XorConfig, out: Path, bin_er: float = 0.1, bin_bp: float = 0.025
) -> None:
    """Payeur Fig. 4b-f (plus a learning curve) for one rule. Panel c and the learning curve use
    every seed; the others show the first seed, as Payeur does. Event rates are averaged over
    the pool in bins of bin_er seconds (Payeur: 0.4 s, for 8 s examples), burst probabilities
    in bins of bin_bp."""
    spb = round(bin_er / cfg.dt)
    r0 = runs[0]
    t_total = 4 * cfg.t_test
    tb = (torch.arange(round(t_total / bin_er)) + 0.5) * bin_er
    thr = (cfg.target_low + cfg.target_high) / 2
    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    fig.suptitle(f"XOR with {cfg.rule} Burstprop (after Payeur et al. 2021, Fig. 4)")

    ax = axes[0, 0]
    ax.plot(tb, binned(r0["before"]["er_in1"], spb), label="input 1")
    ax.plot(tb, binned(r0["before"]["er_in2"], spb), label="input 2")
    ax.set_ylabel("b  ER, input (Hz)")
    ax.legend(fontsize=8)
    condition_axis(ax, t_total)

    ax = axes[0, 1]
    for key, style, label in (("before", "--", "before"), ("after", "-", "after")):
        er = torch.stack([binned(r[key]["er_out"], spb) for r in runs])
        mean = er.mean(0)
        sd = er.std(0) if len(runs) > 1 else torch.zeros_like(mean)
        ax.plot(tb, mean, style, color="C3", label=label)
        ax.fill_between(tb, mean - 2 * sd, mean + 2 * sd, color="C3", alpha=0.15, lw=0)
    ax.axhline(thr, color="k", ls=":", lw=1)
    ax.set_ylabel(f"c  ER, output (Hz)\nmean ± 2 s.d., {len(runs)} seeds")
    ax.legend(fontsize=8)
    condition_axis(ax, t_total)

    # Panels d and e: one early epoch of learning, each example's learning interval shaded.
    d = r0["during"]
    n_ex = len(d["p_out"]) // 4
    t_ex = n_ex * cfg.dt
    spb = round(bin_bp / cfg.dt)
    tb_d = (torch.arange(len(d["p_out"]) // spb) + 0.5) * bin_bp
    learn_spans = [(i * t_ex + cfg.t_predict, (i + 1) * t_ex) for i in range(4)]
    bp_label = "net BP, output" if cfg.rule == "signed" else "BP, output"

    ax = axes[1, 0]
    ax.plot(tb_d, binned(d["p_out"], spb), color="C3", label=bp_label)
    ax.set_ylabel(f"d  {bp_label}")
    ax2 = ax.twinx()
    ax2.plot(tb_d, binned(d["teacher"], spb), "--", color="C6", label="teacher T")
    ax2.set_ylabel("teaching input T (dimensionless)", color="C6")
    ax2.axhline(0, color="C6", lw=0.5)
    for a, b in learn_spans:
        ax.axvspan(a, b, color="0.93", lw=0)
    condition_axis(ax, 4 * t_ex)

    br = binned(d["br_out"], spb)
    for col, (key, sign, name) in enumerate((("p_h1", 1, "hidden 1"), ("p_h2", -1, "hidden 2"))):
        ax = axes[2, col]
        p = binned(d[key], spb)
        ax.plot(tb_d, p, color="C3", label=f"BP, {name}")
        # Output BR rescaled to the range of the hidden BP (inverted for hidden 2), as in 4e.
        rescaled = sign * (br - br.mean()) / (br.std() + 1e-12) * p.std() + p.mean()
        inv = " (inverted)" if sign < 0 else ""
        ax.plot(tb_d, rescaled, color="C1", alpha=0.8, label=f"output BR{inv}, rescaled")
        for a, b in learn_spans:
            ax.axvspan(a, b, color="0.93", lw=0)
        ax.set_ylabel(f"e({'i' * (col + 1)})  BP, {name}")
        ax.legend(fontsize=8, loc="upper right")
        condition_axis(ax, 4 * t_ex)

    spb = round(bin_er / cfg.dt)
    for col, (key, name) in enumerate((("er_h1", "hidden 1"), ("er_h2", "hidden 2"))):
        ax = axes[3, col]
        ax.plot(tb, binned(r0["before"][key], spb), "--", color="C0", label="before")
        ax.plot(tb, binned(r0["after"][key], spb), "-", color="C0", label="after")
        ax.set_ylabel(f"f({'i' * (col + 1)})  ER, {name} (Hz)")
        ax.legend(fontsize=8)
        condition_axis(ax, t_total)

    ax = axes[1, 1]
    hist = torch.tensor([r["history"] for r in runs]).mean(0)  # (epochs, 4), mean over seeds
    win = max(1, min(25, len(hist) // 5))  # running mean over epochs, to see through the noise
    kernel = torch.ones(1, 1, win) / win
    for c in range(4):
        smooth = torch.nn.functional.conv1d(hist[:, c].view(1, 1, -1), kernel).flatten()
        ax.plot(torch.arange(len(smooth)) + win, smooth, color=f"C{c}",
                label=f"{CONDITIONS[c]} {'true' if XOR_TARGET[c] else 'false'}")  # fmt: skip
    ax.axhline(thr, color="k", ls=":", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"learning curve: output ER (Hz)\nmean of {len(runs)} seeds, "
                  f"{win}-epoch average")  # fmt: skip
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


def plot_compare(results: dict[str, list[dict]], cfg: XorConfig, out: Path) -> None:
    """Output ER per condition after learning, unsigned vs signed, one dot per seed."""
    fig, ax = plt.subplots(figsize=(6, 4))
    thr = (cfg.target_low + cfg.target_high) / 2
    width = 0.38
    for k, (rule, runs) in enumerate(results.items()):
        rates = torch.tensor([r["final_rates"] for r in runs])  # (seeds, 4)
        x = torch.arange(4) + (k - 0.5) * width
        ax.bar(x, rates.mean(0), width, color=f"C{k}", alpha=0.6,
               label=f"{rule} ({sum(r['solved'] for r in runs)}/{len(runs)} solved)")  # fmt: skip
        for s in range(rates.shape[0]):
            ax.plot(x, rates[s], "k.", ms=4)
    ax.axhline(thr, color="k", ls=":", lw=1)
    ax.set_xticks(range(4), [str(c) for c in CONDITIONS])
    ax.set_ylabel("output ER after learning (Hz)")
    ax.set_title("XOR after learning: unsigned vs signed Burstprop")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out}")


# --- Command line ---------------------------------------------------------------------------------
def _train_one(cfg: XorConfig) -> dict:
    """One run, for the process pool: one thread per process, so parallel runs don't compete."""
    torch.set_num_threads(1)
    t0 = time.time()
    res = train(cfg, log_every=100)
    rates = ", ".join(f"{x:5.1f}" for x in res["final_rates"])
    print(f"{cfg.rule} seed {cfg.seed}: after learning {rates} Hz, solved={res['solved']} "
          f"({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    return res


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rule", choices=["unsigned", "signed", "both"], default="both")
    parser.add_argument("--seeds", type=int, default=5, help="initializations (Payeur: 5)")
    parser.add_argument("--epochs", type=int, default=XorConfig.epochs)
    parser.add_argument("--n-per-pool", type=int, default=XorConfig.n_per_pool)
    parser.add_argument("--eta", type=float, default=XorConfig.eta)
    parser.add_argument("--teacher", choices=["payeur", "stuck"], default=XorConfig.teacher)
    parser.add_argument("--gamma-t", type=float, default=XorConfig.gamma_t)
    parser.add_argument("--c-teach", type=float, default=XorConfig.c_teach)
    parser.add_argument("--beta", type=float, default=XorConfig.beta)
    parser.add_argument("--no-compensation", action="store_true",
                        help="unsigned: don't cancel the zero-error feedback")  # fmt: skip
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="runs in parallel, one process each")  # fmt: skip
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    rules = ["unsigned", "signed"] if args.rule == "both" else [args.rule]
    base = XorConfig(epochs=args.epochs, n_per_pool=args.n_per_pool, eta=args.eta,
                     teacher=args.teacher, gamma_t=args.gamma_t, c_teach=args.c_teach,
                     beta=args.beta,
                     compensate=not args.no_compensation)  # fmt: skip
    cfgs = [replace(base, rule=rule, seed=seed) for rule in rules for seed in range(args.seeds)]
    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            all_runs = list(pool.map(_train_one, cfgs))
    else:
        all_runs = [_train_one(cfg) for cfg in cfgs]
    print(f"all runs done in {time.time() - t0:.0f} s")

    results: dict[str, list[dict]] = {}
    summary = {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for rule in rules:
        runs = [r for r, cfg in zip(all_runs, cfgs) if cfg.rule == rule]
        results[rule] = runs
        plot_rule(runs, replace(base, rule=rule), args.out_dir / f"xor_fig4_{rule}.png")
        summary[rule] = {"final_rates": [r["final_rates"] for r in runs],
                         "solved": [r["solved"] for r in runs]}  # fmt: skip
    if len(rules) == 2:
        plot_compare(results, base, args.out_dir / "xor_fig4_compare.png")
    summary["config"] = asdict(base)
    path = args.out_dir / "xor_fig4_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
