"""Lifecycle coverage for the portable pinned TrajOpt configuration API."""

from copy import deepcopy

import pytest
import torch

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.trajectory import TrajInterpolationType


def _robot(device_cfg: DeviceCfg = DeviceCfg()) -> RobotCfg:
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=device_cfg)
    return RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)


def test_factory_preserves_structured_inputs_and_selects_position_interpolation():
    optimizer = [{"optimizer": {"solver_type": "es", "n_iters": 9}}]
    transition = {"transition_model_cfg": {"control_space": "POSITION"}}
    metrics = {"rollout": {"name": "portable-metrics"}}
    original_optimizer = deepcopy(optimizer)
    original_transition = deepcopy(transition)
    cfg = TrajOptSolverCfg.create(
        _robot(), optimizer_configs=optimizer, transition_model=transition,
        metrics_rollout=metrics, override_optimizer_num_iters={"es": 6},
        use_cuda_graph=True,
    )
    assert cfg.interpolation_type is TrajInterpolationType.LINEAR_CUDA
    assert cfg.portable_interpolation_type is TrajInterpolationType.LINEAR_CUDA
    assert cfg.optimizer_name == "es" and cfg.max_iterations == 6
    assert cfg.optimizer_configs == original_optimizer
    assert cfg.optimizer_rollout_configs == [original_transition]
    assert cfg.metrics_rollout_config == metrics
    assert optimizer == original_optimizer and transition == original_transition
    assert cfg.requested_use_cuda_graph and not cfg.use_cuda_graph


def test_factory_compiles_scene_cache_and_preserves_cuda_spline_boundary():
    cfg = TrajOptSolverCfg.create(
        _robot(), scene_model="collision_test.yml", collision_cache={"cuboid": 3},
        max_batch_size=2, multi_env=True,
    )
    assert isinstance(cfg.scene_collision_cfg, SceneCollisionCfg)
    assert cfg.scene_collision_cfg.num_envs == 2
    assert cfg.scene_collision_cfg.cache == {"cuboid": 3}
    assert cfg.interpolation_type is TrajInterpolationType.BSPLINE_KNOTS_CUDA
    assert cfg.portable_interpolation_type is TrajInterpolationType.LINEAR_CUDA


def test_clone_update_are_transactional_and_keep_core_coherent():
    cfg = TrajOptSolverCfg.create(_robot(), num_seeds=2, max_batch_size=2)
    clone = cfg.clone(num_seeds=5, non_terminal_tool_pose_weight_factor=0.3)
    assert clone.num_seeds == 5 and cfg.num_seeds == 2
    assert clone.core_cfg is not cfg.core_cfg
    assert clone.robot_config is clone.core_cfg.robot_config
    with pytest.raises(ValueError, match="minimum_trajectory_dt"):
        cfg.update(minimum_trajectory_dt=1.0, maximum_trajectory_dt=0.1)
    assert cfg.minimum_trajectory_dt < cfg.maximum_trajectory_dt
    assert cfg.update(non_terminal_tool_pose_weight_factor=0.2) is cfg
    assert cfg.non_terminal_tool_pose_weight_factor == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"optimizer_configs": []}, "optimizer_configs"),
        ({"optimizer_configs": ["trajopt/unknown.yml"]}, "not portable"),
        ({"collision_cache": {"cuboid": -1}}, "collision_cache"),
        ({"transition_model": {"transition_model_cfg": []}}, "transition_model_cfg"),
    ],
)
def test_factory_rejects_invalid_portable_inputs(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        TrajOptSolverCfg.create(_robot(), **kwargs)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_factory_keeps_mps_config_native_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    cfg = TrajOptSolverCfg.create(
        _robot(DeviceCfg(torch.device("mps"))), device_cfg=DeviceCfg(torch.device("mps")),
        transition_model={"transition_model_cfg": {"control_space": "POSITION"}},
        optimizer_configs=[{"optimizer": {"solver_type": "adam"}}],
    )
    assert cfg.device_cfg.device.type == "mps"
    assert cfg.portable_interpolation_type is TrajInterpolationType.LINEAR_CUDA
