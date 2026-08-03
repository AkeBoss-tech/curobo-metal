# Portable camera and LiDAR observations

`curobo._src.types.camera.CameraObservation` and
`curobo._src.types.lidar.LidarObservation` are real PyTorch value models on
CPU and Apple Metal.  They retain the pinned V2 fields and constructor order,
and add explicit lifecycle helpers that are safe for portable applications:

- `validate(...)` checks ranks, batch compatibility, calibration and device
  ownership without requiring CUDA, Warp, or a host tensor copy.
- `clone`, `copy_`, `detach`, `requires_grad_`, and `to` preserve nested
  `Pose` state and all optional tensor fields.
- `as_dict`/`from_dict` and `save_to_file`/`load_from_file` use ordinary
  tensor payloads; they are suitable for local replay, not a hardware-driver
  recording format.
- Camera pinhole projection and structured depth extraction are differentiable
  on CPU/MPS.  A LiDAR range image can be converted to points with evenly
  spaced azimuth and the supplied elevation bounds; invalid/out-of-range
  pixels are deterministically zeroed.

The value types do **not** provide sensor drivers, raw CUDA/Warp range-image
projection kernels, sparse LiDAR TSDF integration, or exact NVIDIA buffer/ABI
semantics.  `Mapper.integrate(LidarObservation(...))` therefore continues to
raise a precise `NotImplementedError` on the portable dense mapper rather than
pretending that this external backend is available.
