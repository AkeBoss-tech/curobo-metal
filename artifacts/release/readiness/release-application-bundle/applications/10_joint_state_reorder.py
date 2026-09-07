"""Round-trip a joint state through an arbitrary name order."""

import torch
from curobo.types import DeviceCfg, JointState
from application_support import emit, tensor

cfg = DeviceCfg()
names = ["j0", "j1", "j2", "j3"]
state = JointState.from_position(cfg.to_device([[0.0, 1.0, 2.0, 3.0]]), names)
roundtrip = state.reorder(["j2", "j0", "j3", "j1"]).reorder(names)
assert torch.equal(roundtrip.position, state.position)
emit({"position": tensor(roundtrip.position), "velocity": tensor(roundtrip.velocity),
      "names": roundtrip.joint_names})
