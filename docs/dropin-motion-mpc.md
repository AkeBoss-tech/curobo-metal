# Motion planning, MPC, and retargeting compatibility

The pinned cuRoboV2 public modules `motion_planner`,
`batch_motion_planner`, `model_predictive_control`, and
`motion_retargeter` are available over portable CPU/MPS implementations.
Planning composes production IK and trajectory optimization. MPC provides
stateful goal updates, cold/warm starts, and receding-horizon action results.

The upstream `use_cuda_graph=True` configuration default is accepted and
compiled to persistent portable caches. Calls that explicitly manipulate an
NVIDIA CUDA Graph fail with `NotImplementedError`. Runtime world mutation,
dynamic inertial mutation, advanced multi-goal grasp approach/lift geometry,
and exact Warp/LM seeded-IK internals remain bounded gaps.
