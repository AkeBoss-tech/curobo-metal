"""Repeat an IK solve and preserve public result schema."""

import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState
from application_support import emit, tensor

solver = InverseKinematics(InverseKinematicsCfg.create("ur10e.yml"))
current = JointState.from_position(solver.default_joint_state.position.reshape(1, -1), solver.joint_names)
frame = solver.tool_frames[-1]
pose = solver.compute_kinematics(current).tool_poses.get_link_pose(frame)
goal = GoalToolPose.from_poses({frame: pose}, ordered_tool_frames=solver.tool_frames)
first = solver.solve_pose(goal, current_state=current)
second = solver.solve_pose(goal, current_state=current)
assert first.success.all() and second.success.all()
emit({"first": tensor(first.js_solution.position, values=False),
      "second": tensor(second.js_solution.position, values=False), "success": tensor(second.success)})
