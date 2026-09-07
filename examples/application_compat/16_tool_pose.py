"""Create and address multiple tool poses by frame name."""

import torch
from curobo.types import DeviceCfg, ToolPose
from application_support import emit, tensor

cfg = DeviceCfg()
position = cfg.to_device([[[[1, 0, 0], [0, 2, 0]]]])
quaternion = cfg.to_device([[[[1, 0, 0, 0], [1, 0, 0, 0]]]])
tools = ToolPose(["left", "right"], position, quaternion)
right = tools.get_link_pose("right")
assert torch.equal(right.position, cfg.to_device([[0, 2, 0]]))
emit({"all": tensor(tools.position), "right": tensor(right.position),
      "frames": tools.tool_frames})
