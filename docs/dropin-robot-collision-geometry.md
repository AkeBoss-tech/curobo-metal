# Robot collision geometry

`curobo._src.robot.types.collision_geometry.RobotCollisionGeometry` is the
portable topology record shared by FK output and collision planning.  It holds
one integer link index per collision sphere, with no batch or horizon axes:
the same robot ownership map applies to every `[B, H, S, ...]` tensor emitted
by forward kinematics.

The record validates rank, integer dtype, and link-index bounds; supports
`clone`, `copy_`, `detach`, `contiguous`, MPS/CPU `to`, and `link_mask`.  `to`
never casts topology indices to floats, including when supplied a `DeviceCfg`.
`copy_` deliberately retains the target `num_links` topology and rejects an
incompatible source, matching the pinned V2 buffer's immutable metadata.

The implementation uses ordinary PyTorch integer indexing on CPU and MPS.  It
does not emulate Warp launch buffers, CUDA graph capture, BVH geometry, or raw
CUDA kernel ABI behavior.
