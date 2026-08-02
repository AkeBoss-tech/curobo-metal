# Motion planning, MPC, and retargeting compatibility

The pinned cuRoboV2 public modules `motion_planner`,
`batch_motion_planner`, `model_predictive_control`, and
`motion_retargeter` are available over portable CPU/MPS implementations.
Planning composes production IK and trajectory optimization. MPC provides
stateful goal updates, cold/warm starts, and receding-horizon action results.
The corresponding retargeter lifecycle is documented in
[`dropin-motion-retargeter.md`](dropin-motion-retargeter.md).

`MotionPlanner.plan_pose` also runs that real portable IK-to-TrajOpt
composition for normal and goalset pose requests; it does not accidentally
enter the CUDA-rollout-only implicit-goal path.  Retry attempts repair a
partially successful IK seed population before TrajOpt, and optional PRM
seeds are treated as validated graph attempts.  Warmup supports deterministic
FK-derived goalsets and resource destruction is idempotent on CPU/MPS.

The upstream `use_cuda_graph=True` configuration default is accepted and
compiled to persistent portable caches. Calls that explicitly manipulate an
NVIDIA CUDA Graph fail with `NotImplementedError`. Runtime world mutation,
dynamic inertial mutation, advanced multi-goal grasp approach/lift geometry,
and exact Warp/LM seeded-IK internals remain bounded gaps.
