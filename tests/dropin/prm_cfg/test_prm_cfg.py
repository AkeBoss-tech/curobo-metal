"""Behavioral tests for the portable pinned-V2 PRM configuration boundary."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.types.device_cfg import DeviceCfg


def test_factory_resolves_short_robot_scene_and_task_mappings() -> None:
    cfg = PRMGraphPlannerCfg.create(
        "franka.yml",
        graph_planner_config={
            "graph_planner": {
                "max_nodes": 37,
                "new_nodes_per_iteration": 5,
                "neighbors_per_node": 3,
                "ellipsoid_projection_method": "approximate",
                "edge_step": 0.1,
            }
        },
        rollout={"rollout": {"sum_horizon": True, "sampler_seed": 9}},
        transition_model={
            "transition_model_cfg": {
                "control_space": "ACCELERATION",
                "dt_traj_params": {"base_dt": 0.02, "base_ratio": 1.0, "max_dt": 0.02},
            }
        },
        scene_model="collision_test.yml",
        collision_cache={"cuboid": 2},
        use_cuda_graph_for_rollout=False,
        self_collision_check=False,
        graph_path_finder_seed=11,
        device_cfg=DeviceCfg("cpu"),
    )

    # Franka's two locked finger joints are removed, matching the c-space
    # seen through MotionPlannerCfg rather than the raw URDF's nine joints.
    assert cfg.action_dim == 7
    assert cfg.max_nodes == 37
    assert cfg.neighbors_per_node == 3
    assert cfg.ellipsoid_projection_method == "approximate"
    assert isinstance(cfg.rollout_config, RobotRolloutCfg)
    assert cfg.rollout_config.sum_horizon is True
    assert cfg.rollout_config.sampler_seed == 9
    assert cfg.rollout_config.transition_model_cfg.dt_traj_params.base_dt == 0.02
    assert isinstance(cfg.scene_collision_cfg, SceneCollisionCfg)
    assert cfg.scene_collision_cfg.cache == {"cuboid": 2}
    assert cfg.use_cuda_graph_for_rollout is False
    assert cfg.self_collision_check is False
    assert cfg.graph_path_finder_seed == 11


def test_factory_default_paths_compile_without_optional_nvidia_task_corpus() -> None:
    cfg = PRMGraphPlannerCfg.create("franka.yml", device_cfg=DeviceCfg("cpu"))
    assert cfg.action_dim == 7
    assert isinstance(cfg.rollout_config, RobotRolloutCfg)
    with pytest.raises(FileNotFoundError, match="portable package only falls back"):
        PRMGraphPlannerCfg.create("franka.yml", graph_planner_config="missing.yml", device_cfg=DeviceCfg("cpu"))
    with pytest.raises(ValueError, match="unknown collision_cache"):
        PRMGraphPlannerCfg.create("franka.yml", collision_cache={"warp": 1}, device_cfg=DeviceCfg("cpu"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_nodes": 1}, "max_nodes"),
        ({"sample_rejection_ratio": 0}, "sample_rejection_ratio"),
        ({"neighbors_per_node_growth_factor": 0.9}, "neighbors_per_node_growth_factor"),
        ({"ellipsoid_projection_method": "warp"}, "ellipsoid_projection_method"),
        ({"action_lower_bounds": torch.tensor([0.0]), "action_upper_bounds": torch.tensor([0.0])}, "smaller"),
        ({"action_lower_bounds": torch.tensor([0.0])}, "supplied together"),
    ],
)
def test_direct_config_rejects_unrepresentable_or_invalid_planner_state(kwargs, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        PRMGraphPlannerCfg(**kwargs)


def test_config_bounds_are_canonicalized_and_clone_is_independent() -> None:
    cfg = PRMGraphPlannerCfg(
        action_lower_bounds=[-1.0, -2.0],
        action_upper_bounds=[1.0, 2.0],
        connection_radius=0.7,
        device_cfg=DeviceCfg("cpu"),
    )
    assert cfg.action_dim == 2
    assert cfg.action_lower_bounds.device.type == "cpu"
    clone = cfg.clone(max_nodes=31)
    clone.action_lower_bounds.add_(0.1)
    assert clone.max_nodes == 31
    assert not torch.equal(clone.action_lower_bounds, cfg.action_lower_bounds)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_config_canonicalizes_portable_bounds_to_mps_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    cfg = PRMGraphPlannerCfg(
        device_cfg=DeviceCfg(device="mps"),
        action_lower_bounds=[-1.0, -1.0],
        action_upper_bounds=[1.0, 1.0],
    )
    assert cfg.action_lower_bounds.device.type == "mps"
    assert cfg.action_upper_bounds.device.type == "mps"
