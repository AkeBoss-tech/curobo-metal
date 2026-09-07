from types import SimpleNamespace

import pytest
import torch

from curobo._src.types.control_space import ControlSpace
from curobo._src.types.robot import RobotCfg
from curobo.types import DeviceCfg, JointState


def test_joint_state_constructors_clone_to_and_index() -> None:
    source = torch.arange(24, dtype=torch.float32).view(2, 12)
    state = JointState.from_state_tensor(source, ["a", "b", "c"], dof=3)
    assert state.position.shape == (2, 3)
    assert state.get_state_tensor().shape == (2, 12)

    clone = state.clone()
    assert isinstance(clone, JointState)
    assert clone.position.data_ptr() != state.position.data_ptr()
    assert isinstance(state[torch.tensor([1])], JointState)
    assert state[torch.tensor([1])].shape == (1, 3)

    converted = state.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert converted.dtype == torch.float64
    assert state.dtype == torch.float32


def test_joint_state_zeros_detach_assignment_and_errors() -> None:
    state = JointState.zeros((2, 3), DeviceCfg(), ["a", "b", "c"])
    assert torch.equal(state.dt, state.dt.new_ones(2))
    assert state.control_space is None
    state.control_space = ControlSpace.POSITION

    detached = state.detach()
    assert detached is state
    state[0] = JointState.from_position(state.position.new_full((3,), 4.0), ["a", "b", "c"])
    assert torch.equal(state.position[0], state.position.new_full((3,), 4.0))
    # Pinned cuRobo permits construction of partially described states and
    # rejects invalid names when a name-dependent operation is requested.
    partial = JointState.from_position(torch.zeros(2), ["only_one"])
    with pytest.raises(ValueError, match="requested joint"):
        partial.reorder(["missing"])


def test_robot_cfg_constructor_create_identity_and_cspace() -> None:
    marker = SimpleNamespace(cspace=SimpleNamespace(joint_names=["j0"]))
    config = RobotCfg(marker)

    assert RobotCfg.create(config) is config
    assert config.cspace.joint_names == ["j0"]
    with pytest.raises(NotImplementedError, match="curobo-metal"):
        config.write_config("unused.yml")
