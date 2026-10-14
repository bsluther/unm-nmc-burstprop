"""Exponential traces of spike trains (Stuck et al. 2025, eq. 4).

AI DISCLOSURE: this skeleton (docstring and signature) was written by Claude Opus. The function body is for the
students to implement.

The same filter serves two purposes:
  - the presynaptic eligibility trace Ẽ_j in the plasticity rule (eqs. 13-14), and
  - the rate estimate Ē_i that the teacher and regularizer compare with target rates (eqs. 9-10).
"""

from torch import Tensor


def trace_step(trace: Tensor, spikes: Tensor, dt: float, tau: float) -> Tensor:
    """One forward-Euler step of  tau * dE/dt = -E + S(t)  (eq. 4).

    S(t) is a train of Dirac pulses. With timestep dt, a spike in a step counts as a pulse of height 1/dt (so its
    integral over the step is 1). The Euler step is then

        E_new = E + (dt / tau) * (-E + spikes / dt)

    so the trace is in Hz: a neuron firing at a steady r Hz has a trace that averages r. A single spike raises the
    trace by 1/tau, after which it decays by a factor (1 - dt/tau) per step.

    trace, spikes: tensors of the same shape. dt, tau: seconds (e.g. 1e-3 and 10e-3). Returns the new trace.
    """
    raise NotImplementedError
