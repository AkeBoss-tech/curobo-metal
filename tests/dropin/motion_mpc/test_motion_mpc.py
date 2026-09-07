import inspect

import pytest
import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_planner_result import GraspPlanResult
from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.manager_seed import SeedManager
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import SolveState
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.model_predictive_control import (
    ModelPredictiveControl, ModelPredictiveControlCfg,
    ModelPredictiveControlResult,
)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.motion_retargeter import MotionRetargeterCfg, RetargetResult


def _planner(batch=1):
    cfg = MotionPlannerCfg.create(
        "franka.yml", num_ik_seeds=2, num_trajopt_seeds=1,
        use_cuda_graph=True, max_batch_size=batch,
        device_cfg=DeviceCfg("cpu"),
    )
    cfg.trajopt_solver_config.max_iterations = 2
    return MotionPlanner(cfg)


def test_public_aliases_signatures_and_config_defaults():
    assert ModelPredictiveControlResult.__name__ == "MPCSolverResult"
    assert GraspPlanResult().planning_time == 0.0
    signature = inspect.signature(MotionPlanner.plan_cspace)
    assert signature.parameters["max_attempts"].default == 5
    assert signature.parameters["enable_graph_attempt"].default == 1
    cfg = MotionPlannerCfg.create("franka.yml", use_cuda_graph=True, device_cfg=DeviceCfg("cpu"))
    assert cfg.ik_solver_config.use_cuda_graph is False
    assert cfg.device_cfg.device.type == "cpu"


def test_motion_planner_cspace_and_batch_facade_execute():
    planner = _planner()
    start = planner.default_joint_state
    goal = JointState.from_position(start.position + 0.01, planner.joint_names)
    result = planner.plan_cspace(goal, start, max_attempts=1)
    assert result.success.tolist() == [[True]]
    assert result.js_solution.position.shape == (1, 1, 81, 9)
    assert result.interpolated_trajectory.position.shape == (1, 1, 1000, 9)
    assert result.interpolated_last_tstep.tolist() == [[21]]

    batch = BatchMotionPlanner(planner.config)
    starts = JointState.from_position(
        start.position.repeat(2, 1), planner.joint_names
    )
    goals = JointState.from_position(
        starts.position + 0.01, planner.joint_names
    )
    batch_result = batch.plan_cspace(
        goals, starts, max_attempts=1, success_ratio=1.0
    )
    assert batch_result.success.shape == (2, 1)


def test_mpc_setup_goal_and_receding_horizon_result():
    cfg = ModelPredictiveControlCfg.create(
        "franka.yml", warm_start_optimization_num_iters=2,
        cold_start_optimization_num_iters=2,
        device_cfg=DeviceCfg("cpu"),
    )
    mpc = ModelPredictiveControl(cfg)
    current = mpc.default_joint_state
    goal = JointState.from_position(
        current.position + 0.01, mpc.joint_names
    )
    assert mpc.setup(current)
    assert mpc.update_goal_state(goal)
    result = mpc.optimize_action_sequence(current)
    assert result.success.tolist() == [[True]]
    assert result.next_action.position.shape == (1, 7)
    assert result.action_sequence.position.shape == (1, 5, 7)
    try:
        mpc.reset_cuda_graph()
    except NotImplementedError as error:
        assert "CUDA graph" in str(error)
    else:
        raise AssertionError("raw CUDA graph reset must fail explicitly")


def test_goal_and_seed_managers_are_deterministic():
    device = DeviceCfg("cpu")
    solve = SolveState(SolveMode.BATCH, 2, 1, num_seeds=3)
    state = JointState.from_position(torch.zeros(2, 2), ["a", "b"])
    manager = GoalManager(device)
    goal = manager.create_goal_buffer(solve, goal_js=state, current_js=state)
    assert goal.get_index_size() == 6
    # GoalManager reports requested problem batches; seed-expanded indexing is
    # exposed by GoalRegistry.get_index_size().
    assert manager.get_batch_size() == 2

    seeds = SeedManager(
        device, 2, torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]),
        random_seed=9, action_horizon=3,
    )
    first = seeds.generate_random_actions(2, 4)
    seeds.reset_seed()
    torch.testing.assert_close(first, seeds.generate_random_actions(2, 4))


def test_retarget_configuration_and_sequence_type():
    criteria = {"panda_hand": ToolPoseCriteria.track_position_and_orientation()}
    cfg = MotionRetargeterCfg.create(
        "franka.yml", criteria, num_envs=2, num_seeds_global=2,
        num_seeds_local=1,
        device_cfg=DeviceCfg("cpu"),
    )
    assert cfg.tool_frames == ["panda_hand"]
    sequence = SequenceGoalToolPose(
        ["panda_hand"], torch.zeros(3, 2, 1, 1, 3),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(3, 2, 1, 1, 1),
    )
    assert sequence.get_frame(1).position.shape == (2, 1, 1, 1, 3)
    assert RetargetResult(JointState.from_position(torch.zeros(2, 7))).trajectory is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_motion_planner_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device = DeviceCfg(torch.device("mps"), torch.float32)
    cfg = MotionPlannerCfg.create(
        "franka.yml", device_cfg=device, num_ik_seeds=2,
        num_trajopt_seeds=1, use_cuda_graph=True,
    )
    cfg.trajopt_solver_config.max_iterations = 2
    planner = MotionPlanner(cfg)
    start = planner.default_joint_state
    goal = JointState.from_position(
        start.position + 0.01, planner.joint_names
    )
    result = planner.plan_cspace(goal, start, max_attempts=1)
    assert bool(result.success.all().item())
    assert result.js_solution.position.device.type == "mps"
