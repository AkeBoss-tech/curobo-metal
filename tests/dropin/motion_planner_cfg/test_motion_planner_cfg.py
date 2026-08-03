"""Behavioral coverage for the typed portable MotionPlannerCfg boundary."""

import pytest

from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import SceneCfg
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg


def test_factory_compiles_scene_yaml_cache_and_graph_to_typed_configs():
    cfg = MotionPlannerCfg.create(
        "franka.yml",
        scene_model="collision_test.yml",
        collision_cache={"cuboid": 4, "mesh": 0, "voxel": 0},
        num_ik_seeds=3,
        num_trajopt_seeds=2,
        graph_planner_config={"max_nodes": 17, "new_nodes_per_iteration": 3},
        use_cuda_graph=True,
        store_debug=True,
    )

    assert isinstance(cfg.scene_collision_cfg, SceneCollisionCfg)
    assert isinstance(cfg.scene_collision_cfg.scene_model, SceneCfg)
    assert cfg.scene_collision_cfg.cache == {"cuboid": 4, "mesh": 0, "voxel": 0}
    assert cfg.ik_solver_config.scene_collision_cfg is cfg.scene_collision_cfg
    assert isinstance(cfg.graph_planner_config, PRMGraphPlannerCfg)
    assert cfg.graph_planner_config.max_nodes == 17
    assert cfg.graph_planner_config.action_lower_bounds.device.type == "cpu"
    assert cfg.ik_solver_config.num_seeds == 3
    assert cfg.trajopt_solver_config.num_seeds == 2
    # Debug has the same observable effect as upstream: graph capture is not
    # requested.  Portable buffers remain ordinary CPU/MPS state.
    assert not cfg.trajopt_solver_config.use_cuda_graph

    planner = MotionPlanner(cfg)
    assert isinstance(planner.scene_collision_checker, SceneCollision)
    assert planner.scene_collision_checker.num_envs == 1
    planner.destroy()


def test_factory_compiles_per_batch_scene_models_and_default_seed_scaling():
    scenes = [
        {"cuboid": {"left": {"dims": [1, 1, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}}},
        {"cuboid": {"right": {"dims": [1, 1, 1], "pose": [2, 0, 0, 1, 0, 0, 0]}}},
    ]
    cfg = MotionPlannerCfg.create(
        "franka.yml",
        scene_model=scenes,
        collision_cache={"primitive": 1},
        max_batch_size=2,
        multi_env=True,
        use_cuda_graph=False,
    )
    assert isinstance(cfg.scene_collision_cfg.scene_model, list)
    assert cfg.scene_collision_cfg.num_envs == 2
    assert cfg.ik_solver_config.num_seeds == 16
    assert cfg.trajopt_solver_config.num_seeds == 2
    assert cfg.graph_planner_config.use_cuda_graph_for_rollout is False


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"num_ik_seeds": 0}, "num_ik_seeds"),
        ({"position_tolerance": 0.0}, "position_tolerance"),
        ({"interpolation_dt": float("nan")}, "interpolation_dt"),
        ({"collision_cache": {"bad": 1}}, "unknown collision_cache"),
        ({"graph_planner_config": None, "multi_env": True, "scene_model": [{}], "max_batch_size": 2}, "scene_model length"),
    ],
)
def test_factory_rejects_invalid_capacity_and_world_requests(kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        MotionPlannerCfg.create("franka.yml", **kwargs)


def test_factory_rejects_scene_lists_without_per_environment_mode():
    with pytest.raises(ValueError, match="multi_env=True"):
        MotionPlannerCfg.create("franka.yml", scene_model=[{}])


def test_factory_reuses_a_typed_scene_collision_config_across_all_children():
    scene = SceneCollisionCfg(
        scene_model=SceneCfg.create(
            {"cuboid": {"table": {"dims": [1, 1, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}}}
        ),
        cache={"cuboid": 1},
    )
    cfg = MotionPlannerCfg.create("franka.yml", scene_model=scene)

    assert cfg.scene_collision_cfg is scene
    assert cfg.ik_solver_config.scene_collision_cfg is scene
    assert cfg.trajopt_solver_config.scene_collision_cfg is scene
    assert cfg.graph_planner_config.scene_collision_cfg is scene
    assert cfg.robot_config is cfg.ik_solver_config.robot_config


def test_direct_construction_rejects_incoherent_shape_or_scene_records():
    cfg = MotionPlannerCfg.create("franka.yml", scene_model="collision_test.yml")

    bad_traj = cfg.trajopt_solver_config.clone(max_batch_size=2)
    with pytest.raises(ValueError, match="max_batch_size"):
        MotionPlannerCfg(cfg.ik_solver_config, bad_traj, device_cfg=cfg.device_cfg)

    orphan_scene = SceneCollisionCfg(scene_model=SceneCfg.create({}))
    with pytest.raises(ValueError, match="scene_collision_cfg"):
        MotionPlannerCfg(
            cfg.ik_solver_config,
            cfg.trajopt_solver_config,
            cfg.graph_planner_config,
            orphan_scene,
            cfg.device_cfg,
        )


def test_clone_is_independent_but_preserves_shared_child_scene_aliases():
    cfg = MotionPlannerCfg.create("franka.yml", scene_model="collision_test.yml", random_seed=7)
    clone = cfg.clone()

    assert clone is not cfg
    assert clone.ik_solver_config is not cfg.ik_solver_config
    assert clone.scene_collision_cfg is not cfg.scene_collision_cfg
    assert clone.ik_solver_config.scene_collision_cfg is clone.scene_collision_cfg
    assert clone.trajopt_solver_config.scene_collision_cfg is clone.scene_collision_cfg
    assert clone.graph_planner_config.scene_collision_cfg is clone.scene_collision_cfg
    assert clone.requested_use_cuda_graph

    updated_ik = clone.ik_solver_config.clone(core_cfg=clone.ik_solver_config.core_cfg.clone(random_seed=99))
    clone.update(ik_solver_config=updated_ik)
    assert clone.ik_solver_config.random_seed == 99
    assert cfg.ik_solver_config.random_seed == 7


def test_clone_rejects_unknown_or_cross_device_updates_without_migrating_tensors():
    cfg = MotionPlannerCfg.create("franka.yml")
    with pytest.raises(TypeError, match="unknown MotionPlannerCfg"):
        cfg.clone(unknown=True)
    with pytest.raises(ValueError, match="cannot migrate"):
        cfg.clone(device_cfg=DeviceCfg(device="mps"))
