"""Round-trip a rigid pose through its homogeneous matrix."""

import torch
from curobo.types import DeviceCfg, Pose
from application_support import emit, tensor

cfg = DeviceCfg()
pose = Pose.from_list([0.2, -0.3, 0.4, 0.9238795, 0, 0.3826834, 0], cfg)
matrix = pose.get_matrix()
restored = Pose.from_matrix(matrix)
assert torch.allclose(restored.get_matrix(), matrix, atol=1e-6)
emit({"matrix": tensor(matrix), "position": tensor(restored.position),
      "quaternion": tensor(restored.quaternion)})
