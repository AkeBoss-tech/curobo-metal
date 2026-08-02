# Portable perception API-shape closure

This closure strengthens the executable CPU/MPS high-level perception surface
at the pinned cuRobo V2 revision. `FilterDepth` now performs range-aware
tensor filtering with caller-owned output buffers; `Mapper` supports dense
region/block clearing, state loading, untextured rendering helpers, and the
TSDF facade exposes matching high-level lifecycle methods.

`RobotMesh`, `PoseDetector`, and `SDFPoseDetector` now perform deterministic
plain-tensor surface sampling and centroid registration. They are suitable for
bounded pose-estimation workflows and never require `trimesh` merely to
import. Optional `trimesh` conversion is isolated to `get_trimesh`.

The following remain explicit non-drop-in boundaries: raw Warp mapper kernels,
hash/block-pool pointers, PBA/JFA scheduling, Blox/lidar/texture-feature
fusion, raw Warp mesh IDs, and CUDA numerical parity. The rendering color
channel is deterministic black when no color feature volume was fused; it does
not fabricate texture data.
