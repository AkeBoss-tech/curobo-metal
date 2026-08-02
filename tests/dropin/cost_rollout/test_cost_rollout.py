import importlib
import torch
from curobo._src.types.device_cfg import DeviceCfg

def test_pinned_cost_modules_import():
    names = [
        "cost_base", "cost_base_cfg", "cost_cspace_base", "cost_cspace_cfg",
        "cost_cspace_dist", "cost_cspace_dist_cfg", "cost_cspace_position",
        "cost_cspace_state", "cost_cspace_type", "cost_pose_type",
        "cost_scene_collision", "cost_scene_collision_cfg", "cost_self_collision",
        "cost_self_collision_cfg", "cost_support_polygon", "cost_support_polygon_cfg",
        "cost_tool_pose", "cost_tool_pose_cfg",
    ]
    for name in names:
        importlib.import_module(f"curobo._src.cost.{name}")

def test_cspace_distance_is_differentiable():
    from curobo._src.cost.cost_cspace_dist import CSpaceDistCost
    from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
    value = torch.tensor([[[0., 0.], [1., 2.]]], requires_grad=True)
    goal = torch.tensor([[1., 1.]])
    cost = CSpaceDistCost(CSpaceDistCostCfg(weight=2., device_cfg=DeviceCfg()))(value, goal)
    cost.sum().backward()
    assert cost.shape == value.shape
    assert value.grad is not None and torch.isfinite(value.grad).all()

def test_cost_collection_and_goal_registry():
    from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints
    from curobo._src.rollout.goal_registry import GoalRegistry
    values = CostCollection(); values.add(torch.ones(2, 3, 1), "one")
    assert values.get_sum(False).shape == (2, 3)
    assert CostsAndConstraints(costs=values).get_sum_cost(True).tolist() == [[3.], [3.]]
    goal = GoalRegistry.create_idx(2, True, 3, DeviceCfg())
    assert goal.get_index_size() == 6
    assert goal.idxs_env.tolist() == [[0], [0], [0], [1], [1], [1]]

def test_public_rosenbrock_rollout_matches_formula_and_autograd():
    from curobo.rollout import RosenbrockCfg, RosenbrockRollout
    rollout = RosenbrockRollout(RosenbrockCfg(DeviceCfg(), a=1., b=100.))
    action = torch.tensor([[[0., 0.], [1., 1.]]], requires_grad=True)
    result = rollout.evaluate_action(action)
    cost = result.costs_and_constraints.get_sum_cost()
    assert torch.allclose(cost, torch.tensor([[1., 0.]]))
    cost.sum().backward()
    assert action.grad is not None

def test_robot_rollout_surface():
    from curobo._src.rollout.rollout_robot import RobotRollout
    action = torch.zeros(2, 4, 3)
    result = RobotRollout().evaluate_action(action)
    assert result.state.position.shape == action.shape


def test_goal_registry_repeats_indices_and_copies_buffers():
    from curobo._src.rollout.goal_registry import GoalRegistry
    from curobo._src.state.state_joint import JointState

    registry = GoalRegistry.create_idx(2, True, 3, DeviceCfg())
    assert registry.idxs_goal_js.squeeze(-1).tolist() == [0, 0, 0, 1, 1, 1]
    assert registry.idxs_env.squeeze(-1).tolist() == [0, 0, 0, 1, 1, 1]
    assert registry.get_index_size() == 6

    source = GoalRegistry(goal_js=JointState.from_position(torch.tensor([[1.], [2.]])))
    target = GoalRegistry()
    target.copy_(source)
    assert target.goal_js is not source.goal_js
    assert target.goal_js.position.tolist() == [[1.], [2.]]


def test_cost_manager_runs_cspace_and_convergence_with_autograd():
    from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
    from curobo._src.rollout.goal_registry import GoalRegistry
    from curobo._src.state.state_joint import JointState
    from curobo._src.state.state_robot import RobotState

    manager = RobotCostManager(DeviceCfg())
    manager.initialize_from_config(RobotCostManagerCfg(
        target_cspace_dist_cfg=CSpaceDistCostCfg(weight=2.0, device_cfg=DeviceCfg())
    ))
    position = torch.tensor([[[0.0, 1.0], [1.0, 2.0]]], requires_grad=True)
    state = RobotState(JointState.from_position(position))
    goal = GoalRegistry(goal_js=JointState.from_position(torch.tensor([[1.0, 1.0]])))
    metrics = manager.compute_convergence(state, goal)
    assert metrics.names == ["target_cspace_dist_tolerance"]
    value = metrics.get_sum(False)
    value.sum().backward()
    assert position.grad is not None and torch.isfinite(position.grad).all()


def test_cost_manager_missing_component_and_cuda_graph_boundaries():
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.rollout_robot import RobotRollout
    manager = RobotCostManager(DeviceCfg())
    try:
        manager.enable_cost_component("missing")
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("missing components must not silently succeed")
    try:
        RobotRollout().reset_cuda_graph()
    except NotImplementedError:
        pass
    else:
        raise AssertionError("CUDA graph capture cannot be emulated on CPU/MPS")
