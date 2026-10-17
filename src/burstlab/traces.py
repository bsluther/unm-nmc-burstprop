"""Exponential traces of spike trains (Stuck et al. 2025, eq. 4).

AI DISCLOSURE: this skeleton (docstring and signature) was written by Claude Opus. The function body is for the
students to implement.

The same filter serves two purposes:
  - the presynaptic eligibility trace Ẽ_j in the plasticity rule (eqs. 13-14), and
  - the rate estimate Ē_i that the teacher and regularizer compare with target rates (eqs. 9-10).
"""

from torch import Tensor


# TODO: Rewrite docstring in my own words. Still trying to make sense of it so keeping Claude's
#   version for now.
def trace_step(trace: Tensor, event_train: Tensor, dt: float, tau: float) -> Tensor:
    """One forward-Euler step of  tau * dE/dt = -E + S(t)  (eq. 4).

    S(t) is a train of Dirac pulses. With timestep dt, an event in a step counts as a pulse of height 1/dt (so its
    integral over the step is 1). The Euler step is then

        E_new = E + (dt / tau) * (-E + event_train / dt)

    so the trace is in Hz: a neuron firing at a steady r Hz has a trace that averages r. A single event raises the
    trace by 1/tau, after which it decays by a factor (1 - dt/tau) per step.

    trace, event_train: tensors of the same shape. dt, tau: seconds (e.g. 1e-3 and 10e-3).
    Returns the new trace in Hz.

    ### Brian's notes
    Note: In Stuck, the continuous event train is written S(t), see (Stuck 2025, p5)

    The reason for dividing event_train by dt is somewhat gnarly. I spent a few hours trying to grok
    what's going on but didn't get there.
    It has to do with finding the discrete analogue of S(t) = sum_k delta(t - t_k). We put events
    into bins of width dt, so the height of each of those bins (the pulse height) must be 1/dt. That
    makes a bin with an event have area 1, matching the Dirac delta's integral.

    Computing (event_train / dt) is a means to get the pulse height given the 0/1 event.

    In a sense the discrete, forward Euler representation of S(t) is (events[n] / dt). The width
    of each delta(t-t_k) spike is infinitely thin in continuous world. But in discrete world, the
    width is dt. We still want an area of 1, so the pulse height is 1/dt.
    """

    return trace + (dt / tau) * (-trace + event_train / dt)
