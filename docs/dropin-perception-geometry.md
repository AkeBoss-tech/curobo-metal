# Perception and geometry compatibility

This wave exposes the pinned cuRoboV2 public `perception`, `scene`, and
`sphere_fit` modules and portable implementations of dense TSDF/ESDF mapping,
depth filtering, rendering, checkpointing, pose/point transforms, camera
projection, mesh triangulation, and deterministic mesh-to-sphere fitting.

The implementation uses ordinary differentiable PyTorch operations on CPU and
Apple MPS. Raw Warp kernels, nvblox/Blox storage, Isaac integrations, GUI
viewers, feature-texture fusion, static obstacle stamping, and native sparse
hash-table mutation are explicit unsupported boundaries. Dense mapping is
intended for bounded manipulation volumes.
