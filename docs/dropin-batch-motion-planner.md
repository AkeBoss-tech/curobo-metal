# Batched Motion Planner Compatibility

`curobo.batch_motion_planner.BatchMotionPlanner` runs real batched IK,
trajectory optimization, optional shared-world PRM seeding, and grasp-stage
composition on CPU and MPS. Calls retain first-success rows across retry
attempts, including the selected trajectory and interpolation payloads.

`max_batch_size` remains accepted as the V2 configuration field. The eager
portable backend does not require a statically captured CUDA shape, so a call
can use any non-empty batch size. A `multi_env=True` configuration disables
shared PRM seeding; its per-problem collision worlds continue to route through
the production scene checker.

The familiar `use_cuda_graph` setting is accepted but represents persistent
portable execution state, not CUDA graph capture. Raw CUDA graph controls,
Warp/BVH implementation details, analytic CCD, and CUDA numerical parity are
not provided. Grasp approach and lift solve the real batched stages, but do not
claim the CUDA-only non-terminal linear rollout cost.
