# Perception runtime compatibility

This wave makes every pinned `curobo._src.perception.mapper` module and the
pose-estimation helper modules import safely without Warp, CUDA, Blox, or
`trimesh`.

The public `Mapper`, TSDF integrator, exact dense ESDF, sparse block packing,
mesh extraction, rendering, pose refinement, coordinate conversion,
quantization, checkpoints, point-cloud alignment, and LM primitives execute
with ordinary PyTorch tensors on CPU and float32 MPS. They route to the
production `curobo_metal.ops.perception` implementation. The portable map is a
bounded dense representation with sparse checkpoint/export views; it is not
the upstream Warp hash-table ABI.

Raw Warp kernel functions and builder entry points import but raise a precise
`NotImplementedError`. Lidar integration, Blox representations, raw pointers,
CUDA launch objects, feature-volume texture fusion, and the exact Warp/PBA/JFA
scheduling are not emulated. The portable ESDF computes an exact dense distance
transform, so values follow the documented mathematical contract while
performance and tie behavior need not match those approximate CUDA schedulers.

Checkpoint loading uses Torch's weights-only mode through the high-level
mapper. Optional external mesh packages are not imported at module import time.
