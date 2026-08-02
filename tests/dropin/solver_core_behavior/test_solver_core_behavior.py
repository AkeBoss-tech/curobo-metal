"""Behavioral coverage for the portable V2 solver-core component."""

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_core_cfg import (
    SolverCoreCfg,
    create_scene_collision_cfg,
    create_solver_core_cfg,
    create_metrics_rollout_config,
    create_rollout_configs,
    resolve_yaml_configs,
)
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg


def _robot(device_cfg=DeviceCfg()):
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=device_cfg)
    return RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)


def test_cfg_accepts_cuda_default_but_marks_eager_portable_execution():
    cfg = SolverCoreCfg(_robot(), optimizer_configs=[{"solver_type": "lbfgs"}], use_cuda_graph=True)
    assert cfg.requested_use_cuda_graph
    assert not cfg.use_cuda_graph
    with pytest.raises(ValueError, match="nonnegative"):
        SolverCoreCfg(_robot(), optimizer_configs=[], random_seed=-1)


def test_factory_builds_native_portable_records_and_scene_configuration():
    robot = _robot()
    device = robot.device_cfg
    optimizer_dicts = [{"optimizer": {"solver_type": "lbfgs", "num_iters": 3}, "rollout": {}}]
    transition = {"transition_model_cfg": {"portable": True}}
    rollout = create_rollout_configs(optimizer_dicts, transition, robot, device, 0.03, self_collision_check=False)
    metrics = create_metrics_rollout_config({"rollout": {}}, transition, robot, device)
    scene = create_scene_collision_cfg(
        SceneCfg(cuboid=[Cuboid("far", pose=[3, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])]),
        {"cuboid": 1}, device,
    )
    cfg = create_solver_core_cfg(
        robot, optimizer_dicts, {"rollout": {}}, transition, scene.scene_model,
        device, collision_cache={"cuboid": 1}, override_optimizer_num_iters={"lbfgs": 25},
    )
    assert rollout[0].collision_activation_distance == 0.03
    assert not rollout[0].self_collision_check
    assert metrics.device_cfg == device
    assert cfg.optimizer_configs[0].num_iters == 25
    assert cfg.scene_collision_cfg.num_envs == 1


def test_goal_criteria_sampling_and_inertial_mutation_lifecycle():
    device = DeviceCfg()
    scene = create_scene_collision_cfg(
        SceneCfg(cuboid=[Cuboid("far", pose=[10, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])]),
        {"cuboid": 1}, device,
    )
    core = SolverCore(SolverCoreCfg(_robot(device), optimizer_configs=[{}], scene_collision_cfg=scene))
    state = core.default_joint_state.unsqueeze(0)
    goal, changed = core.prepare_goal_buffer(
        SolveState(SolveMode.SINGLE, 1, 1, num_seeds=2), None,
        current_state=state, goal_state=state,
    )
    assert changed and goal.get_index_size() == 2
    core.enable_tool_pose_tracking([core.tool_frames[0]], non_terminal_weight_factor=0.2)
    assert torch.all(core._tool_pose_criteria[core.tool_frames[0]].non_terminal_pose_axes_weight_factor == 0.2)
    core.disable_tool_pose_tracking()
    assert not bool(core._tool_pose_criteria[core.tool_frames[0]].terminal_pose_axes_weight_factor.any())
    samples = core.sample_configs(5)
    assert samples.shape[1] == core.action_dim
    assert samples.shape[0] <= 5
    name = core._kinematics._model.link_names[0]
    core.update_link_inertial(name, mass=1.25, com=torch.zeros(3), inertia=torch.ones(6))
    assert core._kinematics._model.mass[0].item() == pytest.approx(1.25)
    with pytest.raises(NotImplementedError, match="CUDA graph"):
        core.reset_cuda_graph()


def test_resolve_yaml_configs_accepts_robot_yaml_and_mapping_inputs():
    robot, optimizers, metrics, transition, scene = resolve_yaml_configs(
        "franka.yml", [{"optimizer": {"solver_type": "lbfgs"}}],
        {"rollout": {}}, {"transition_model_cfg": {}},
        {"cuboid": {}}, DeviceCfg(),
    )
    assert isinstance(robot, RobotCfg)
    assert optimizers[0]["optimizer"]["solver_type"] == "lbfgs"
    assert metrics["rollout"] == {}
    assert transition["transition_model_cfg"] == {}
    assert scene == {"cuboid": {}}
