"""Solve a moved reachable goal and reject a distant unreachable goal."""

import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState
from application_support import emit, tensor

solver = InverseKinematics(InverseKinematicsCfg.create("franka.yml"))
current = JointState.from_position(solver.default_joint_state.position.reshape(1, -1).clone(), solver.joint_names)
original = current.position.clone()
target = current.clone()
target.position[:, 0] += 0.1
frame = solver.tool_frames[-1]
target_pose = solver.compute_kinematics(target).tool_poses.get_link_pose(frame).clone()
goal = GoalToolPose.from_poses({frame: target_pose}, ordered_tool_frames=solver.tool_frames)
result = solver.solve_pose(goal, current_state=current)
assert result.success.all().item()
achieved = solver.compute_kinematics(result.js_solution.reorder(solver.joint_names)).tool_poses.get_link_pose(frame)
error = torch.linalg.vector_norm(achieved.position - target_pose.position, dim=-1)
assert error.max().item() < 0.005
assert torch.equal(current.position, original)
impossible = target_pose.clone()
impossible.position[..., 0] += 10.0
bad_goal = GoalToolPose.from_poses({frame: impossible}, ordered_tool_frames=solver.tool_frames)
bad = solver.solve_pose(bad_goal, current_state=current)
assert not bad.success.any().item()
emit({"success": tensor(result.success), "position": tensor(achieved.position),
      "solution": tensor(result.js_solution.position, values=False),
      "joint_names": result.js_solution.joint_names, "unreachable_success": tensor(bad.success)})
