"""Construct a full joint state and preserve every derivative through cloning."""

import torch
from curobo.types import DeviceCfg, JointState
from application_support import emit, tensor

cfg = DeviceCfg()
position = cfg.to_device([[0.1, 0.2, 0.3]])
state = JointState(position, 2 * position, 3 * position, ["a", "b", "c"], 4 * position)
clone = state.clone()
assert clone.joint_names == state.joint_names
assert clone.position.data_ptr() != state.position.data_ptr()
emit({"position": tensor(clone.position), "velocity": tensor(clone.velocity),
      "acceleration": tensor(clone.acceleration), "jerk": tensor(clone.jerk)})
