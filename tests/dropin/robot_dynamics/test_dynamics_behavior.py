"""Executable portable behavior for the pinned robot dynamics facade."""

from __future__ import annotations

import pytest
from curobo.types import DeviceCfg
import torch

from curobo._src.robot.dynamics import Dynamics, DynamicsCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState


def _dynamics() -> tuple[Dynamics, KinematicsCfg]:
    cfg = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=DeviceCfg("cpu"))
    return Dynamics(DynamicsCfg(cfg.kinematics_config, cfg.device_cfg)), cfg


def _state(cfg: KinematicsCfg, *, batch: int = 2, horizon: int = 3) -> JointState:
    q = torch.zeros((batch, horizon, cfg.dof), requires_grad=True)
    return JointState(
        q,
        torch.zeros_like(q),
        torch.zeros_like(q),
        joint_names=cfg.kinematics_config.joint_names,
        device_cfg=DeviceCfg("cpu"),
    )


def test_dynamics_lifecycle_and_spatial_gravity_convention() -> None:
    dynamics, cfg = _dynamics()
    assert dynamics.dof == cfg.dof
    assert dynamics._tau_buffer is None
    dynamics.setup_batch_size(2, 3)
    assert dynamics._tau_buffer.shape == (6, cfg.dof)
    torch.testing.assert_close(
        dynamics.config.get_gravity_spatial(),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 9.81]),
    )
    with pytest.raises(ValueError, match="positive"):
        dynamics.setup_batch_size(0)


def test_config_validation_metadata_aliases_and_live_gravity() -> None:
    dynamics, cfg = _dynamics()
    # These fields are consumed by source-compatible callers inspecting the
    # compiled RNEA layout.  Here they remain meaningful portable tensors.
    assert dynamics._fixed_transforms.shape == (cfg.kinematics_config.num_links, 4, 4)
    assert dynamics._link_masses_com.shape == (cfg.kinematics_config.num_links, 4)
    assert dynamics._link_inertias.shape == (cfg.kinematics_config.num_links, 8)
    assert dynamics._level_starts[-1].item() == cfg.kinematics_config.num_links

    state = _state(cfg, batch=1, horizon=1)
    before = dynamics.compute_inverse_dynamics(state)
    dynamics.config.gravity[:] = [0.0, 0.0, 0.0]
    after = dynamics.compute_inverse_dynamics(state)
    assert not torch.allclose(before, after)

    with pytest.raises(ValueError, match="three values"):
        DynamicsCfg(cfg.kinematics_config, cfg.device_cfg, [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        DynamicsCfg(cfg.kinematics_config, cfg.device_cfg, [0.0, float("nan"), 0.0])


def test_external_wrenches_broadcast_and_autograd() -> None:
    dynamics, cfg = _dynamics()
    state = _state(cfg)
    dynamics.setup_batch_size(1)
    baseline = dynamics.compute_inverse_dynamics(state)
    wrench = torch.zeros((cfg.kinematics_config.num_links, 6), requires_grad=True)
    # A link-local force produces a nonzero generalized load on a nontrivial
    # Franka configuration.  The same [link, 6] input broadcasts over B/H.
    with torch.no_grad():
        wrench[-1, 5] = 1.0
    loaded = dynamics.compute_inverse_dynamics(state, wrench)
    assert loaded.shape == state.position.shape
    assert not torch.allclose(loaded, baseline)
    grad_q, grad_wrench = torch.autograd.grad(loaded.square().sum(), (state.position, wrench))
    assert torch.isfinite(grad_q).all() and torch.isfinite(grad_wrench).all()


def test_external_wrench_rank_and_joint_reorder_are_deterministic() -> None:
    dynamics, cfg = _dynamics()
    state = _state(cfg, batch=1, horizon=2)
    all_wrenches = torch.zeros((1, 2, cfg.kinematics_config.num_links, 6))
    all_wrenches[..., -1, 3] = 2.0
    expected = dynamics.compute_inverse_dynamics(state, all_wrenches)
    reverse = list(reversed(cfg.kinematics_config.joint_names))
    reordered = JointState(
        state.position[..., list(reversed(range(cfg.dof)))],
        state.velocity[..., list(reversed(range(cfg.dof)))],
        state.acceleration[..., list(reversed(range(cfg.dof)))],
        joint_names=reverse,
        device_cfg=DeviceCfg("cpu"),
    )
    # Results are returned in backend order, matching the historical RNEA
    # facade's internal-order contract after name normalization.
    torch.testing.assert_close(expected, dynamics.compute_inverse_dynamics(reordered, all_wrenches))
    with pytest.raises(ValueError, match="f_ext"):
        dynamics.compute_inverse_dynamics(state, torch.zeros((1, 6)))


def test_forward_mass_rollout_and_mutation_lifecycle() -> None:
    dynamics, cfg = _dynamics()
    position = torch.zeros((2, cfg.dof))
    state = JointState(position, torch.zeros_like(position), torch.zeros_like(position), device_cfg=DeviceCfg("cpu"))
    torque = dynamics.compute_inverse_dynamics(state)
    acceleration = dynamics.compute_forward_dynamics(state, torque)
    torch.testing.assert_close(acceleration, torch.zeros_like(acceleration), atol=5e-4, rtol=5e-4)
    matrix = dynamics.get_mass_matrix(position)
    assert matrix.shape == (2, cfg.dof, cfg.dof)
    rollout = dynamics.rollout(state, torch.zeros((2, 2, cfg.dof)), 0.01)
    assert rollout.position.shape == (2, 3, cfg.dof)
    dynamics.update_link_inertial("panda_link1", mass=3.0)
    assert dynamics.kinematics_config.get_link_masses_com("panda_link1")[-1].item() == 3.0
    assert dynamics._link_masses_com[
        dynamics._get_link_index("panda_link1"), 3
    ].item() == 3.0
    dynamics.update_link_inertia("panda_link1", torch.eye(3))
    assert dynamics._link_inertias[
        dynamics._get_link_index("panda_link1"), :3
    ].tolist() == [1.0, 1.0, 1.0]
    with pytest.raises(ValueError, match="at least one"):
        dynamics.update_link_inertial("panda_link1")
    with pytest.raises(ValueError, match="cannot be empty"):
        dynamics.update_links_inertial({})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_external_wrench_autograd_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the composed link-frame wrench path on Apple Metal itself."""
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo._src.types.device_cfg import DeviceCfg

    cfg = KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=DeviceCfg(torch.device("mps"))
    )
    dynamics = Dynamics(DynamicsCfg(cfg.kinematics_config, cfg.device_cfg))
    q = torch.zeros((1, 2, cfg.dof), device="mps", requires_grad=True)
    state = JointState(q, torch.zeros_like(q), torch.zeros_like(q))
    wrench = torch.zeros((cfg.kinematics_config.num_links, 6), device="mps", requires_grad=True)
    with torch.no_grad():
        wrench[-1, 5] = 1.0
    result = dynamics.compute_inverse_dynamics(state, wrench)
    grad_q, grad_wrench = torch.autograd.grad(result.square().sum(), (q, wrench))
    assert result.device.type == grad_q.device.type == grad_wrench.device.type == "mps"
