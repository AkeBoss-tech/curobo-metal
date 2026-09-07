from __future__ import annotations

import torch

from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import (
    blend_joint_states, calculate_fd_from_position, joint_state_to_tensor,
)
from curobo._src.transition.fns_state_transition import (
    StateFromAcceleration, StateFromPositionTeleport,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.sampling.sequencer_halton import HaltonSequencer
from curobo._src.util.sampling.sequencer_random import RandomSequencer
from curobo._src.util.sampling.sequencer_roberts import RobertsSequencer
from curobo._src.util.state_filter import FilterCfg, JointStateFilter
from curobo._src.util.tensor_util import fd_tensor, stable_topk


def test_joint_state_ops_autograd_and_trajectory_indexing():
    q = torch.arange(24.0).reshape(2, 3, 4).requires_grad_()
    state = JointState.from_position(q, ["a", "b", "c", "d"])
    state.dt = torch.full((2, 2), 0.1)
    calculate_fd_from_position(state)
    assert state.velocity.shape == (2, 2, 4)
    assert state.acceleration.shape == (2, 1, 4)
    state.velocity.sum().backward()
    assert q.grad is not None

    packed = joint_state_to_tensor(JointState.from_position(torch.zeros(2, 4)))
    assert packed.shape == (2, 16)
    reordered = JointState.from_position(torch.arange(8.0).reshape(2, 4), ["a", "b", "c", "d"])
    assert reordered.index_dof(torch.tensor([3, 1])).joint_names == ["d", "b"]

    target = JointState.from_position(torch.zeros(1, 4))
    blend_joint_states(target, JointState.from_position(torch.ones(1, 4)), FilterCoeff(position=0.5))
    torch.testing.assert_close(target.position, torch.full((1, 4), 0.5))


def test_filter_and_transition_semantics():
    cfg = DeviceCfg("cpu")
    start = JointState.from_position(torch.zeros(2, 3), ["a", "b", "c"])
    filt = JointStateFilter(FilterCfg.create(
        {"position": 0.5, "velocity": 1, "acceleration": 1, "jerk": 1},
        dt=0.1, control_space=ControlSpace.ACCELERATION,
        device_cfg=DeviceCfg("cpu"),
    ))
    filt.filter_joint_state(start)
    command = filt.integrate_acc(torch.ones(2, 3), start)
    torch.testing.assert_close(command.velocity, torch.full((2, 3), 0.1))

    acceleration = torch.ones(2, 4, 3, requires_grad=True)
    transition = StateFromAcceleration(cfg, torch.full((4,), 0.1), 3, batch_size=2, horizon=4)
    result = transition.forward(start, acceleration)
    assert result.position.shape == (2, 4, 3)
    result.position.sum().backward()
    assert acceleration.grad is not None

    teleported = StateFromPositionTeleport(cfg, 2, 4).forward(start, torch.ones(2, 4, 3))
    assert teleported.position.shape == (2, 4, 3)
    assert torch.count_nonzero(teleported.velocity) == 0


def test_sampling_is_deterministic_bounded_and_device_resident():
    for cls in (RandomSequencer, HaltonSequencer, RobertsSequencer):
        sequence = cls(3, seed=7)
        first = sequence.random(5)
        sequence.reset()
        assert (first == sequence.random(5)).all()

    sample_buffer = SampleBuffer.create_halton_sample_buffer(
        3, [2, 3, 4], [-2, -3, -4], store_buffer=32, seed=7,
        device_cfg=DeviceCfg("cpu"),
    )
    first = sample_buffer.get_samples(8, bounded=True)
    sample_buffer.reset()
    torch.testing.assert_close(first, sample_buffer.get_samples(8, bounded=True))
    assert first.device.type == "cpu"
    assert torch.all(first >= torch.tensor([-2, -3, -4]))
    assert torch.all(first <= torch.tensor([2, 3, 4]))
    gaussian = sample_buffer.get_gaussian_samples(16)
    assert torch.isfinite(gaussian).all()


def test_tensor_helpers():
    values = torch.arange(8.0).reshape(1, 4, 2)
    torch.testing.assert_close(fd_tensor(values, torch.tensor([0.5, 0.5, 0.5])),
                               torch.full((1, 3, 2), 4.0))
    top, idx = stable_topk(torch.tensor([2.0, 1.0, 3.0]), 2)
    assert top.tolist() == [3.0, 2.0]
    assert idx.tolist() == [2, 0]


def test_mps_without_cpu_fallback_when_available(monkeypatch):
    if not torch.backends.mps.is_available():
        return
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    cfg = DeviceCfg(torch.device("mps"))
    start = JointState.from_position(torch.zeros(1, 2, device="mps"))
    action = torch.ones(1, 3, 2, device="mps", requires_grad=True)
    result = StateFromAcceleration(
        cfg, torch.full((3,), 0.1, device="mps"), 2, horizon=3
    ).forward(start, action)
    result.position.sum().backward()
    assert result.position.device.type == "mps"
