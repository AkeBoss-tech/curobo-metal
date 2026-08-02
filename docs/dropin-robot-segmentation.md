# Robot depth segmentation compatibility

`curobo.perception.RobotSegmenter` and
`curobo._src.perception.robot_segmenter.RobotSegmenter` run on CPU and MPS.
They use the configured robot collision spheres as a conservative, portable
robot geometry proxy:

1. depth is projected with the observation intrinsics and `depth_to_meter`;
2. `CameraObservation.pose` transforms camera-frame points into the robot base
   frame; and
3. a pixel is marked when its positive depth point is within
   `distance_threshold` of a positive-radius collision sphere surface.

The returned mask is boolean and has the depth-image shape.  The returned
filtered depth image has robot pixels set to zero.  A single camera batch can
be broadcast over a batch of active joint states.  Invalid/zero-depth pixels
are never marked, including when they project to a sphere centre.

`RobotSegmenter.from_robot_file` accepts the packaged robot YAML name, a path,
or a configuration mapping.  `collision_sphere_buffer` grows the segmenter's
private sphere radii before kinematics are compiled; it does not mutate a
mapping supplied by the caller.

`use_cuda_graph` is accepted for source compatibility and selects the portable
shape-aware direct executor.  No CUDA graph is captured on Metal.  This API is
not a mesh renderer: Warp/Blox mesh rasterisation, packed CUDA sphere kernels,
and renderer-specific occlusion or closest-face tie behavior remain outside
the portable scope.
