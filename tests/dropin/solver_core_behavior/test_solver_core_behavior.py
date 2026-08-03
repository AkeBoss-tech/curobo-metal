"""Behavioral coverage for the portable V2 solver-core component."""

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import Cuboid, SceneCfg, Sphere
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
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
    TimeTrajCfg,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.rollout.rollout_robot_cfg import RobotRolloutCfg
from curobo._src.optim.optim_factory import create_optimization_config


def _robot(device_cfg=DeviceCfg()):
    kin = KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=device_cfg)
    return RobotCfg(kin.kinematics_config.robot_cfg, device_cfg=device_cfg)


def _executable_core_cfg():
    robot = _robot()
    transition = RobotStateTransitionCfg(
        robot_config=robot,
        dt_traj_params=TimeTrajCfg(0.05, 1.0, 0.05),
        device_cfg=robot.device_cfg,
        horizon=4,
        control_space=ControlSpace.ACCELERATION,
    )
    rollout = RobotRolloutCfg(robot.device_cfg, transition_model_cfg=transition)
    optimizer = create_optimization_config({"solver_type": "lbfgs"}, robot.device_cfg)
    return SolverCoreCfg(
        robot, robot.device_cfg, [optimizer], [rollout], rollout, use_cuda_graph=True
    )


def test_cfg_accepts_cuda_default_but_marks_eager_portable_execution():
    cfg = SolverCoreCfg(_robot(), optimizer_configs=[{"solver_type": "lbfgs"}], use_cuda_graph=True)
    assert cfg.requested_use_cuda_graph
    assert not cfg.use_cuda_graph
    cloned = cfg.clone(random_seed=99)
    assert cloned.random_seed == 99
    assert cloned.requested_use_cuda_graph and not cloned.use_cuda_graph
    assert cloned.optimizer_configs == cfg.optimizer_configs
    assert cloned.optimizer_configs is not cfg.optimizer_configs
    with pytest.raises(ValueError, match="nonnegative"):
        SolverCoreCfg(_robot(), optimizer_configs=[], random_seed=-1)
    with pytest.raises(TypeError, match="random_seed"):
        SolverCoreCfg(_robot(), optimizer_configs=[], random_seed=True)
    with pytest.raises(TypeError, match="unknown"):
        cfg.clone(cuda_graph_handle=None)


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
    # Shape/lifecycle callers retain the upstream hook, but no CUDA graph is
    # captured on the portable backend.
    assert core.reset_cuda_graph() is None


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


class _RolloutProbe:
    def __init__(self):
        self.batch_sizes = []
        self.goals = []
        self.shape_resets = 0
        self.scene_collision_checker = None

    def update_batch_size(self, value):
        self.batch_sizes.append(value)

    def update_params(self, value):
        self.goals.append(value)

    def reset_shape(self):
        self.shape_resets += 1


class _OptimizerProbe:
    def __init__(self):
        self.problem_sizes = []
        self.goals = []
        self.shape_resets = 0

    def update_num_problems(self, value):
        self.problem_sizes.append(value)

    def update_rollout_params(self, value):
        self.goals.append(value)

    def reset_shape(self):
        self.shape_resets += 1


def test_goal_shape_updates_attached_execution_consumers_and_value_updates_refresh_goals():
    core = SolverCore(SolverCoreCfg(_robot(), optimizer_configs=[]))
    rollout = _RolloutProbe()
    optimizer = _OptimizerProbe()
    core.metrics_rollout = rollout
    core._optimizer = optimizer
    state = core.default_joint_state.unsqueeze(0)
    solve_state = SolveState(SolveMode.SINGLE, 2, 1, num_ik_seeds=3)

    goal, changed = core.prepare_goal_buffer(
        solve_state, None, current_state=state.repeat_seeds(2), goal_state=state.repeat_seeds(2)
    )
    assert changed
    assert core.problem_batch_size == 6
    assert rollout.batch_sizes == [6]
    assert optimizer.problem_sizes == [6]
    assert rollout.goals == [goal]
    assert optimizer.goals == [goal]
    assert core.task_initialized

    refreshed, changed = core.prepare_goal_buffer(
        solve_state, None, current_state=state.repeat_seeds(2), goal_state=state.repeat_seeds(2)
    )
    assert not changed
    assert len(rollout.goals) == len(optimizer.goals) == 2
    assert refreshed is core.goal_buffer


def test_world_replacement_propagates_to_attached_rollouts_and_clears_lifecycle_state():
    core = SolverCore(SolverCoreCfg(_robot(), optimizer_configs=[]))
    rollout = _RolloutProbe()
    core.metrics_rollout = rollout
    state = core.default_joint_state.unsqueeze(0)
    core.prepare_goal_buffer(SolveState(SolveMode.SINGLE, 1, 1), None, current_state=state)
    scene = SceneCfg(sphere=[Sphere("guard", position=[2.0, 0.0, 0.0], radius=0.1)])

    core.update_world(scene)
    assert core.scene_collision_checker is rollout.scene_collision_checker
    assert core.scene_generation == 1
    assert rollout.shape_resets >= 2  # goal shape + world cache invalidation
    assert core.config.scene_collision_cfg.scene_model is scene
    with pytest.raises(NotImplementedError, match="YAML/USD/Warp"):
        core.update_world("scene.yml")

    core.destroy()
    assert core.goal_buffer is None
    assert core.solve_state is None
    assert not core.task_initialized


def test_direct_typed_config_builds_rollouts_optimizer_and_attachment_lifecycle():
    core = SolverCore(_executable_core_cfg())
    assert core.metrics_rollout is not None
    assert core.auxiliary_rollout is not None
    assert len(core.optimizer_rollouts) == 2
    assert len(core.optimizers) == 1
    assert core.optimizer is not None
    assert core.action_horizon == 4
    assert core.init_state.position.shape == (1, core.action_dim)
    old_attachment = core.attachment_manager

    core.update_world(SceneCfg(sphere=[Sphere("guard", position=[2.0, 0.0, 0.0], radius=0.1)]))
    assert core.attachment_manager is not old_attachment
    assert core.attachment_manager._scene_collision is core.scene_collision_checker
    # The regular reset hook must remain valid when V2's CUDA-graph default is
    # accepted but deliberately compiled to eager CPU/MPS execution.
    core.reset_cuda_graph()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_solver_core_goal_and_world_lifecycle_stays_on_mps_without_fallback():
    device = DeviceCfg(device=torch.device("mps"))
    core = SolverCore(SolverCoreCfg(_robot(device), device_cfg=device, optimizer_configs=[]))
    state = core.default_joint_state.unsqueeze(0)
    goal, changed = core.prepare_goal_buffer(
        SolveState(SolveMode.SINGLE, 1, 1, num_seeds=2), None,
        current_state=state, goal_state=state,
    )
    assert changed
    assert goal.goal_js.position.device.type == "mps"
    core.update_world(SceneCfg(sphere=[Sphere("guard", position=[2.0, 0.0, 0.0], radius=0.1)]))
    samples = core.sample_configs(2)
    assert samples.device.type == "mps"
