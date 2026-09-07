"""Plan a nonzero joint-space move twice, preserving input and failure semantics."""

import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState
from application_support import emit, tensor

planner = MotionPlanner(MotionPlannerCfg.create("franka.yml"))
current = JointState.from_position(planner.default_joint_state.position.reshape(1, -1).clone(), planner.joint_names)
original = current.position.clone()
goal = current.clone()
goal.position[:, 0] += 0.1
observations = {}
for index in range(2):
    result = planner.plan_cspace(goal, current)
    assert result is not None and result.success.all().item()
    path = result.js_solution.reorder(planner.joint_names).position
    assert path.shape[-2] > 1
    assert (path[..., -1, :] - goal.position).abs().max().item() < 0.05
    assert (path[..., 0, :] - current.position).abs().max().item() < 0.05
    assert (path[..., -1, :] - path[..., 0, :]).abs().max().item() > 0.02
    assert torch.equal(current.position, original)
    observations[str(index)] = {"success": tensor(result.success), "path": tensor(path, values=False),
                                "endpoint": tensor(path[..., -1, :])}
assert planner.plan_cspace(goal, current, max_attempts=0) is None
observations["zero_attempts_is_none"] = True
emit(observations)
