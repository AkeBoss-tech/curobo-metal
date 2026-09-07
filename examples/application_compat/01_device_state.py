"""Allocate through the default configuration, clone, mutate, and serialize state."""

import io

import torch
from curobo.types import DeviceCfg, JointState
from application_support import emit, tensor

cfg = DeviceCfg()
state = JointState.from_position(cfg.to_device([[0.1, -0.2, 0.3]]), ["a", "b", "c"])
clone = state.clone()
clone.position.add_(1.0)
assert torch.allclose(clone.position - state.position, torch.ones_like(state.position))
buffer = io.BytesIO()
torch.save(state.position, buffer)
buffer.seek(0)
restored = torch.load(buffer, map_location=cfg.device, weights_only=True)
assert torch.equal(restored, state.position)
assert cfg.cpu().device.type == "cpu"
emit({"state": tensor(state.position), "restored": tensor(restored), "names": state.joint_names})
