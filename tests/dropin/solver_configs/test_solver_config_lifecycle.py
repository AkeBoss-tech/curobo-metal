"""Portable behavior coverage for the three pinned solver config factories."""

import pytest
import torch

from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.trajectory import TrajInterpolationType


def _robot(device_cfg=DeviceCfg()):
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=device_cfg)
    return RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)


def test_ik_factory_keeps_portable_core_and_validates_lifecycle():
    cfg = IKSolverCfg.create(
        "franka.yml", max_batch_size=2, multi_env=True, max_goalset=3,
        num_seeds=4, optimization_dt=0.05, velocity_regularization_weight=0.1,
        acceleration_regularization_weight=0.2, use_cuda_graph=True,
    )
    assert cfg.core_cfg.robot_config is cfg.robot_config
    assert cfg.requested_use_cuda_graph is True
    assert cfg.use_cuda_graph is False
    assert cfg.max_batch_size == 2 and cfg.multi_env and cfg.max_goalset == 3
    cloned = cfg.clone(num_seeds=5)
    assert cloned.num_seeds == 5 and cfg.num_seeds == 4
    assert cloned.core_cfg is not cfg.core_cfg
    assert cfg.update(exit_early=False) is cfg and not cfg.exit_early
    with pytest.raises(ValueError, match="num_seeds"):
        cfg.clone(num_seeds=0)
    with pytest.raises(ValueError, match="optimization_dt"):
        cfg.clone(optimization_dt=0.0)
    with pytest.raises(TypeError, match="unknown"):
        cfg.clone(not_a_field=True)


def test_mpc_factory_preserves_deceleration_and_strict_command_rate():
    cfg = MPCSolverCfg.create(
        "franka.yml", optimization_dt=0.04, num_control_points=8,
        squared_l2_regularization_weight=[0.1, 0.2], deceleration_time=0.3,
        max_batch_size=2, multi_env=True, use_cuda_graph=True,
    )
    assert cfg.optimization_dt == pytest.approx(0.04)
    assert cfg.deceleration_time == pytest.approx(0.3)
    assert cfg.requested_use_cuda_graph and not cfg.use_cuda_graph
    assert cfg.clone(cold_start_optimization_num_iters=6).cold_start_optimization_num_iters == 6
    with pytest.raises(ValueError, match="interpolation_steps"):
        MPCSolverCfg.create("franka.yml", interpolation_steps=3)
    with pytest.raises(ValueError, match="deceleration_profile"):
        cfg.clone(deceleration_profile="linear")
    with pytest.raises(ValueError, match="warm_start_optimization_num_iters"):
        cfg.clone(warm_start_optimization_num_iters=0)


def test_trajopt_factory_and_legacy_direct_constructor_have_coherent_core():
    cfg = TrajOptSolverCfg.create(
        "franka.yml", num_seeds=3, minimum_trajectory_dt=0.01,
        maximum_trajectory_dt=0.1, interpolation_dt=0.02,
        override_optimizer_num_iters={"lbfgs": 7}, use_cuda_graph=True,
    )
    assert cfg.max_iterations == 7
    assert cfg.optimizer_name == "lbfgs"
    assert cfg.requested_use_cuda_graph and not cfg.use_cuda_graph
    assert cfg.clone(action_horizon=8).action_horizon == 8

    direct = TrajOptSolverCfg(
        [], _robot(), action_horizon=6, max_iterations=2,
        interpolation_type=TrajInterpolationType.LINEAR_CUDA,
        use_cuda_graph_value=False, random_seed_value=7,
    )
    assert direct.core_cfg.robot_config is direct.robot_config
    assert direct.random_seed == 7
    with pytest.raises(ValueError, match="minimum_trajectory_dt"):
        direct.clone(minimum_trajectory_dt=0.3, maximum_trajectory_dt=0.1)
    with pytest.raises(ValueError, match="optimizer_name"):
        direct.clone(optimizer_name="warp")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_factories_assemble_mps_native_configs_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device_cfg = DeviceCfg(torch.device("mps"), torch.float32)
    for factory in (IKSolverCfg.create, MPCSolverCfg.create, TrajOptSolverCfg.create):
        cfg = factory("franka.yml", device_cfg=device_cfg, use_cuda_graph=True)
        assert cfg.device_cfg.device.type == "mps"
        assert cfg.use_cuda_graph is False
        assert cfg.requested_use_cuda_graph is True
