# Robot segmenter lifecycle

`curobo._src.perception.robot_segmenter.RobotSegmenter` provides portable
CPU/MPS depth segmentation using collision spheres.  Its high-level result is
`(mask, filtered_depth)`: `mask` is boolean, and `filtered_depth` retains the
input depth dtype with pixels assigned to the robot set to zero.

## Live camera/calibration handling

The segmenter is safe to reuse with successive mutable `CameraObservation`
instances.  It never retains the first frame through a portable graph cache:
each call uses the supplied depth image, camera pose, and joint state.

Projection rays are cached by image geometry, `depth_to_meter`, and the
intrinsics tensor's identity and in-place mutation version.  Replacing or
editing intrinsics therefore rebuilds the ray grid automatically.  Call
`invalidate_camera_projection()` to force a rebuild, or `reset()` to discard
both calibration and the portable direct-executor state while retaining the
robot model.

Camera intrinsics/depth precision may differ from the robot state precision.
Projection runs in the calibration dtype, and pose/sphere distance operations
perform ordinary differentiable PyTorch promotion.  All tensors must remain
on the same device.

`use_cuda_graph=True` is accepted for cuRobo source compatibility.  On CPU
and MPS it does not capture a CUDA graph; raw CUDA graph capture, Warp mesh
rasterization, packed sphere kernels, and renderer-specific visibility/tie
behavior remain explicit non-portable boundaries.
