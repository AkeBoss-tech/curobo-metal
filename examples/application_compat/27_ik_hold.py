"""Solve the robot's current end-effector pose without moving the input."""

import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.types import GoalToolPose, JointState
from application_support import emit, tensor

solver = InverseKinematics(InverseKinematicsCfg.create("franka.yml"))
current = JointState.from_position(solver.default_joint_state.position.reshape(1, -1), solver.joint_names)
frame = solver.tool_frames[-1]
pose = solver.compute_kinematics(current).tool_poses.get_link_pose(frame)
result = solver.solve_pose(GoalToolPose.from_poses({frame: pose}, ordered_tool_frames=solver.tool_frames),
                           current_state=current)
assert result.success.all()
emit({"success": tensor(result.success), "solution": tensor(result.js_solution.position, values=False)})
