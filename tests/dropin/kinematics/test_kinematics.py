import pytest
import torch

from curobo.kinematics import Kinematics, KinematicsCfg, KinematicsState
from curobo.types import DeviceCfg, JointState


def _model(**kwargs):
    config = KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=kwargs.pop("device_cfg", DeviceCfg())
    )
    return Kinematics(config, **kwargs)


def test_packaged_franka_public_surface_and_shapes():
    model = _model(compute_jacobian=True, compute_com=True)
    assert model.dof == 7
    assert model.joint_names == [f"panda_joint{i}" for i in range(1, 8)]
    assert model.tool_frames == ["panda_hand"]
    assert model.total_spheres == 65

    q = model.default_joint_position.repeat(2, 3, 1)
    state = model.compute_kinematics(
        JointState.from_position(q, joint_names=model.joint_names)
    )
    assert isinstance(state, KinematicsState)
    assert state.tool_poses.position.shape == (2, 3, 1, 3)
    assert state.tool_poses.quaternion.shape == (2, 3, 1, 4)
    assert state.tool_jacobians.shape == (2, 3, 1, 6, 7)
    assert state.robot_spheres.shape == (2, 3, 65, 4)
    assert state.robot_com.shape == (2, 3, 4)
    torch.testing.assert_close(
        state.tool_poses.quaternion.norm(dim=-1), torch.ones(2, 3, 1)
    )


def test_input_ranks_joint_validation_buffers_and_errors():
    model = _model()
    one = model.compute_kinematics(JointState.from_position(model.default_joint_position))
    batch = model.compute_kinematics(
        JointState.from_position(model.default_joint_position.repeat(4, 1))
    )
    assert one.tool_poses.position.shape[:2] == (1, 1)
    assert batch.tool_poses.position.shape[:2] == (4, 1)
    assert model._buffers["idxs_env"].shape == (4,)

    with pytest.raises(ValueError, match="Joint names do not match"):
        model.compute_kinematics(
            JointState.from_position(
                model.default_joint_position, joint_names=list(reversed(model.joint_names))
            )
        )
    with pytest.raises(ValueError, match="dof"):
        model._forward(torch.zeros(1, 1, 6))
    with pytest.raises(ValueError, match="batch and horizon"):
        model.update_batch_size(0, 1)


@pytest.mark.parametrize("batch,horizon", [(1, 1), (2, 4)])
def test_gradients(batch, horizon):
    model = _model(compute_jacobian=True)
    q = model.default_joint_position.repeat(batch, horizon, 1).requires_grad_()
    state = model.compute_kinematics(
        JointState.from_position(q, joint_names=model.joint_names)
    )
    loss = (
        state.tool_poses.position.square().sum()
        + state.tool_poses.quaternion.square().sum()
        + state.robot_spheres[..., :3].square().sum()
        + state.tool_jacobians.square().sum()
    )
    gradient = torch.autograd.grad(loss, q)[0]
    assert gradient.shape == q.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_with_cpu_fallback_disabled(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = DeviceCfg(torch.device("mps"), torch.float32)
    model = _model(device_cfg=device, compute_jacobian=True)
    q = model.default_joint_position.repeat(2, 1).requires_grad_()
    state = model.compute_kinematics(
        JointState.from_position(q, joint_names=model.joint_names)
    )
    torch.autograd.grad(state.tool_poses.position.sum(), q)
    assert state.tool_poses.position.device.type == "mps"
