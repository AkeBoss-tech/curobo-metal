"""Plan a hold trajectory and check public path endpoints."""

import torch
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import JointState
from application_support import emit, tensor

planner = MotionPlanner(MotionPlannerCfg.create("franka.yml"))
state = JointState.from_position(planner.default_joint_state.position.reshape(1, -1), planner.joint_names)
result = planner.plan_cspace(state.clone(), state)
assert result is not None and result.success.all()
path = result.js_solution.reorder(planner.joint_names).position
assert torch.allclose(path[..., 0, :], state.position, atol=0.05)
assert torch.allclose(path[..., -1, :], state.position, atol=0.05)
emit({"success": tensor(result.success), "path": tensor(path, values=False),
      "endpoint": tensor(path[..., -1, :])})
