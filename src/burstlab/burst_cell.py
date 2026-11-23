from typing import NamedTuple

from torch import Tensor
import norse

norse.torch.LIFCell()


class BurstState(NamedTuple):
    """State of one layer of burstprop neurons. Tensors are shaped (batch, neuron), so with a batch
    size of 32 and a layer of 100 neurons, a scalar state tensor is (32, 100).

    Attributes:
        v_s: Somatic potential.
        v_d: Dendritic potential.
        trace: Eligibility trace of this neuron, i.e. the exponential moving average of this
            neuron's events. Biologically each synapse would have a trace of presynaptic activity,
            duplicating information. We store one trace per neuron and share it across every synapse
            this neuron feeds into. Also acts as the rate estimate for the teacher and regularizer.
            Unit: Hz.
    """

    v_s: Tensor
    v_d: Tensor
    trace: Tensor


class BurstParameters(NamedTuple):
    """
    Parameters for a Burstprop neuron.

    Note: the dimensionless form doesn't need E_L (resting potential) or g_L (leak conductance).

    Attributes:
        tau_s: Somatic time constant, in seconds.
        tau_d: Dendritic time constant, in seconds.
        v_th: Threshold voltage.
        v_reset: Reset voltage.
    """

    tau_s: float
    tau_d: float
    v_th: float = 1
    v_reset: float = 0


def burst_step(i_s: Tensor, i_d: Tensor, state: BurstState, p: BurstParameters, dt: float):
    v_s = state.v_s + dt * (i_s - state.v_s) / p.tau_s
    v_d = state.v_d + dt * (i_d - state.v_d) / p.tau_d
    trace = trace_step # YOU ARE HERE
    return
