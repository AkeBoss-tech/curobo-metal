# Geometry and graph utility compatibility

The portable graph utilities retain the pinned cuRobo V2 Python call layouts
for nearest-neighbor selection, unique-node filtering, deterministic samples,
ellipsoid transforms, PRM extension, linear connection, and path pruning.
Their implementations use standard differentiable PyTorch operators, so the
same supported calls run on CPU and MPS with fallback disabled.

Geometry helpers provide native tensor camera projection, quaternion output
buffers, convex hull distances, and mesh/voxel helper APIs.  The portable
paths intentionally do not emulate raw Warp kernel objects, CUDA graph capture,
Warp BVH IDs, or analytic continuous collision detection.  Those entry points
remain explicit `NotImplementedError` boundaries instead of presenting a false
drop-in CUDA ABI.

The API surface gate tracks declaration shape separately from behavioral
evidence.  The focused geometry/graph tests cover exact public parameter order,
CPU/MPS tensor behavior, autograd, deterministic sampling, and output-buffer
semantics.
