# TSDF integrator lifecycle

`BlockSparseTSDFIntegrator` provides a bounded dense PyTorch implementation of
the public block-sparse TSDF integration lifecycle on CPU and Apple MPS.  It
accepts a single `[H,W]` depth image for `num_cameras=1` and a source-shaped
`[num_cameras,H,W]` camera batch otherwise.  Intrinsics and poses can be
broadcast from one camera or supplied per camera.

Every frame is validated before temporal/frustum decay or fusion begins.  A
camera-count, resolution, dtype, or map-device error therefore cannot mutate a
previously valid map.  Finite depth pixels inside the configured range are the
actual integration mask; zero, NaN, infinite, and out-of-range pixels are
valid frame data but contribute no TSDF weight or occupancy.

The portable backend intentionally does not expose Warp block pools, custom
camera/LiDAR kernels, hash-table allocation, feature-volume fusion, or CUDA
graph capture.  LiDAR and feature-volume requests fail explicitly.
