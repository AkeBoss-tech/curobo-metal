"""Build a goal set from public Pose values and select one goal."""

import torch
from curobo.types import DeviceCfg, GoalToolPose, Pose
from application_support import emit, tensor

cfg = DeviceCfg()
pose = Pose.from_list([0.1, 0.2, 0.3, 1, 0, 0, 0], cfg)
goal = GoalToolPose.from_poses({"tool": pose})
selected = goal[0]
assert selected.tool_frames == ["tool"]
assert torch.equal(selected.position.reshape(-1, 3), pose.position)
emit({"goal": tensor(goal.position), "selected": tensor(selected.position)})
