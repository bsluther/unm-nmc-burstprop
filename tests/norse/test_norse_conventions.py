"""Executable checks for the claims in docs/norse/.

Each test is tagged with the gotcha ID it backs (e.g. G3) in docs/norse/gotchas.md.
If a Norse upgrade breaks one of these, update the doc along with the test.
"""

import warnings

import pytest
import torch

warnings.filterwarnings("ignore", category=FutureWarning)  # torch.jit.script deprecation in norse

import norse.torch as snn
from norse.torch.functional.leaky_integrator_box import LIBoxParameters
from norse.torch.functional.lif import (
    LIFFeedForwardState,
    LIFParameters,
    lif_feed_forward_step,
)
from norse.torch.functional.lif_box import (
    LIFBoxFeedForwardState,
    LIFBoxParameters,
    lif_box_feed_forward_step,
)
from norse.torch.functional.stdp import STDPParameters, STDPState, stdp_step_linear
from norse.torch.module.lif_box import LIFBox


def test_g1_time_constants_are_inverse_seconds():
    p = LIFBoxParameters()
    assert p.tau_mem_inv.item() == pytest.approx(100.0)  # 1 / 10 ms
    # one step of pure decay with dt = 1 ms multiplies v by (1 - dt * tau_mem_inv) = 0.9
    _, s = lif_box_feed_forward_step(
        torch.zeros(1), LIFBoxFeedForwardState(v=torch.tensor([0.5])), p
    )
    assert s.v.item() == pytest.approx(0.45)


def test_g2_input_is_scaled_by_dt_tau_mem_inv():
    _, s = lif_box_feed_forward_step(
        torch.ones(1), LIFBoxFeedForwardState(v=torch.zeros(1)), LIFBoxParameters()
    )
    assert s.v.item() == pytest.approx(0.1)


def test_g3_current_based_lif_filters_input_through_i():
    _, s = lif_feed_forward_step(
        torch.ones(1),
        LIFFeedForwardState(v=torch.zeros(1), i=torch.zeros(1)),
        LIFParameters(),
    )
    assert s.v.item() == pytest.approx(0.1)  # input enters v on the same step...
    assert s.i.item() == pytest.approx(0.8)  # ...and lingers in i, decaying by 1 - dt/tau_syn


def test_g4_threshold_is_strict():
    p = LIFBoxParameters(tau_mem_inv=torch.tensor(0.0))  # freeze dynamics
    z, s = lif_box_feed_forward_step(
        torch.zeros(1), LIFBoxFeedForwardState(v=torch.ones(1)), p
    )
    assert z.item() == 0.0 and s.v.item() == 1.0


def test_g5_spike_emitted_on_same_step_as_crossing():
    z, s = snn.LIFBoxCell()(torch.full((1,), 20.0))
    assert z.item() == 1.0
    assert s.v.item() == 0.0  # reset applied in the same step


def test_g6_box_params_must_be_tensors():
    with pytest.raises(ValueError):
        snn.LIFBoxCell(LIFBoxParameters(tau_mem_inv=100.0, v_leak=0.0))(torch.ones(2, 3))
    with pytest.raises(AttributeError):
        snn.LIBoxCell(LIBoxParameters(tau_mem_inv=100.0, v_leak=0.0))(torch.ones(2, 3))


def test_g7_initial_state_may_be_scalar():
    st = snn.LIFCell().initial_state(torch.zeros(3, 4))
    assert st.v.shape == ()
    assert st.i.shape == (3, 4)
    _, s = snn.LIFBoxCell()(torch.zeros(3, 4))
    assert s.v.shape == (3, 4)  # broadcast after the first step


def test_g8_graph_is_built_even_without_parameters():
    _, s = snn.LIFBoxCell()(torch.ones(3, 4))
    assert s.v.requires_grad and s.v.grad_fn is not None
    with torch.no_grad():
        _, s = snn.LIFBoxCell()(torch.ones(3, 4))
    assert not s.v.requires_grad


def test_g9_namespace_gaps():
    for name in ["LIFBox", "LIBox", "STDPState", "stdp_step_linear", "PoissonEncoderStep", "ExpFilter"]:
        assert not hasattr(snn, name), name
    for name in ["LIFBoxCell", "LIBoxCell", "LIFCell", "LIF", "SequentialState", "Lift"]:
        assert hasattr(snn, name), name


def test_g10_poisson_defaults_differ():
    torch.manual_seed(0)
    x = torch.ones(20000)
    assert snn.poisson_encode(x, 50).mean().item() / 1e-3 == pytest.approx(100, rel=0.05)
    assert snn.poisson_encode_step(x).mean().item() == 1.0  # f_max=1000 Hz -> spike every step
    assert snn.poisson_encode(torch.ones(2, 784), 100).shape == (100, 2, 784)  # time first


def test_g10_poisson_rates_in_hz_via_f_max_one():
    torch.manual_seed(0)
    rates_hz = torch.full((20000,), 20.0)
    spikes = snn.poisson_encode(rates_hz, 100, f_max=1.0)
    assert spikes.mean().item() / 1e-3 == pytest.approx(20, rel=0.05)


def test_g11_stdp_trace_is_rate_times_dt():
    p = STDPParameters(tau_pre_inv=torch.tensor(100.0), tau_post_inv=torch.tensor(100.0))
    st = STDPState(t_pre=torch.zeros(1, 1), t_post=torch.zeros(1, 1))
    ret = st.decay(torch.ones(1, 1), torch.zeros(1, 1), p.tau_pre_inv, p.tau_post_inv, p.a_pre, p.a_post)
    assert ret is None  # mutates in place
    assert st.t_pre.item() == pytest.approx(0.1)  # jump = dt * tau_inv * a

    torch.manual_seed(0)
    st = STDPState(t_pre=torch.zeros(1, 1), t_post=torch.zeros(1, 1))
    spikes = (torch.rand(20000, 1, 1) < 0.05).float()  # 50 Hz at dt = 1 ms
    trace = []
    for z in spikes:
        st.decay(z, torch.zeros(1, 1), p.tau_pre_inv, p.tau_post_inv, p.a_pre, p.a_post)
        trace.append(st.t_pre.item())
    assert sum(trace[1000:]) / len(trace[1000:]) == pytest.approx(0.05, rel=0.1)


def test_g12_stdp_sums_over_batch_and_clamps_to_unit_interval():
    st = STDPState(t_pre=torch.ones(4, 2), t_post=torch.zeros(4, 3))
    w, _ = stdp_step_linear(
        torch.zeros(4, 2), torch.ones(4, 3), torch.zeros(3, 2), st,
        STDPParameters(w_min=-10.0, w_max=10.0),
    )
    assert w.shape == (3, 2)  # (post, pre) like nn.Linear.weight
    assert w[0, 0].item() == pytest.approx(1e-3 * 4 * 0.98)  # summed over batch of 4

    st = STDPState(t_pre=torch.zeros(1, 2), t_post=torch.zeros(1, 3))
    w, _ = stdp_step_linear(torch.zeros(1, 2), torch.zeros(1, 3), -0.5 * torch.ones(3, 2), st)
    assert (w == 0.0).all()  # default hardbound clamps signed weights into [0, 1]


def test_g13_threshold_methods():
    from norse.torch.functional.threshold import threshold

    for m in ["heaviside", "super", "triangle", "tanh", "circ", "heavi_erfc"]:
        threshold(torch.zeros(1), m, 1.0)
    for m in ["logistic", "sigmoid", "adjoint"]:
        with pytest.raises(ValueError):
            threshold(torch.zeros(1), m, 1.0)


def test_logistic_fn_samples_bernoulli_sigmoid():
    # stuck2025-api-map.md: logistic_fn(x, 0.5) ~ Bernoulli(sigmoid(x)), tanh surrogate backward
    torch.manual_seed(0)
    x = torch.full((200000,), 1.0, requires_grad=True)
    z = snn.logistic_fn(x, 0.5)
    assert z.mean().item() == pytest.approx(torch.sigmoid(torch.tensor(1.0)).item(), abs=0.01)
    z.sum().backward()
    assert x.grad[0].item() == pytest.approx(1 - torch.tanh(torch.tensor(0.5)).item() ** 2)


def test_g14_li_params_are_plain_namedtuples():
    assert not hasattr(LIBoxParameters(), "to")
    assert hasattr(LIFBoxParameters(), "to")


def test_g15_default_param_tensors_are_shared():
    assert LIFBoxParameters().v_leak is LIFBoxParameters().v_leak


def test_g16_lifmc_couples_units_not_compartments():
    cell = snn.LIFMCRecurrentCell(3, 4)
    assert cell.g_coupling.shape == (4, 4)


def test_time_lifted_modules_agree():
    x = torch.full((7, 2, 3), 20.0)
    a, _ = snn.Lift(snn.LIFBoxCell())(x)
    b, _ = LIFBox()(x)
    c, _ = snn.lift(lif_box_feed_forward_step, p=LIFBoxParameters())(
        x, state=LIFBoxFeedForwardState(v=torch.zeros(2, 3))
    )
    assert torch.equal(a, b) and torch.equal(b, c)


def test_record_states_stacks_over_time():
    out, states = snn.LIF(record_states=True)(torch.ones(5, 2, 3))
    assert out.shape == (5, 2, 3) and states.v.shape == (5, 2, 3)
    _, state = snn.LIF()(torch.ones(5, 2, 3))
    assert state.v.shape == (2, 3)


def test_sequential_state_returns_per_layer_states():
    model = snn.SequentialState(
        torch.nn.Linear(3, 4), snn.LIFBoxCell(), torch.nn.Linear(4, 2), snn.LIFCell()
    )
    _, states = model(torch.ones(2, 3))
    assert [type(s).__name__ for s in states] == [
        "NoneType", "LIFBoxFeedForwardState", "NoneType", "LIFFeedForwardState",
    ]


def test_custom_state_tuple_pattern():
    from norse.torch.utils import pytree

    class MyState(pytree.StateTuple, metaclass=pytree.MultipleInheritanceNamedTupleMeta):
        v: torch.Tensor
        a: torch.Tensor

    s = MyState(torch.zeros(2), torch.zeros(2))
    assert isinstance(s, tuple)
    assert type(s.to("cpu")) is MyState
    assert s._replace(a=torch.ones(2)).a.sum().item() == 2.0
    assert torch.utils._pytree.tree_map(lambda t: t + 1, s).v.sum().item() == 2.0


def test_lifbox_rate_matches_continuous_lif():
    # f-I curve sanity check: 1/(tau ln(I/(I-1))) = 91 Hz for I = 1.5, tau = 10 ms
    out, _ = LIFBox()(torch.full((1000, 1), 1.5))
    assert out.sum().item() == pytest.approx(91, abs=2)
