"""Stack, repeat, index, and concatenate public pose batches."""

import torch
from curobo.types import DeviceCfg, Pose
from application_support import emit, tensor

cfg = DeviceCfg()
a = Pose.from_list([1, 0, 0, 1, 0, 0, 0], cfg)
b = Pose.from_list([0, 2, 0, 1, 0, 0, 0], cfg)
stacked = a.stack(b)
repeated = stacked[0].repeat(3)
assert stacked.position.shape == (2, 3)
assert repeated.position.shape == (3, 3)
assert torch.equal(repeated.position[0], a.position[0])
emit({"stacked": tensor(stacked.position), "repeated": tensor(repeated.position)})
