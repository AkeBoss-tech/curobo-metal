"""Repeat and slice joint-state batches without sharing mutable storage."""

import torch
from curobo.types import DeviceCfg, JointState
from application_support import emit, tensor

cfg = DeviceCfg()
state = JointState.from_position(cfg.to_device([[0.2, -0.1, 0.4]]), ["a", "b", "c"])
batch = state.repeat([5, 1])
selected = batch[2].clone()
selected.position.add_(1)
assert batch.position.shape == (5, 3)
assert not torch.equal(selected.position, batch.position[2])
emit({"batch": tensor(batch.position), "selected": tensor(selected.position)})
