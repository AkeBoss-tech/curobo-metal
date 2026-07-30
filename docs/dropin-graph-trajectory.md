# Graph and trajectory drop-in compatibility

This wave provides the pinned cuRoboV2 graph-planner, trajectory utility, and
trajectory-optimizer import paths without importing CUDA, Warp, or NetworkX.
`PRMGraphPlanner` routes batched feasibility and deterministic roadmap planning
through `curobo_metal.ops.graph_planning`; `TrajOptSolver.solve_cspace` routes
seeded trajectory optimization through `curobo_metal.ops.trajectory`.

The public enum values, result/config field names, primary method signatures,
seed generation, interpolation, mutable graph lifecycle, and command-buffer
lifecycle follow pinned revision
`8e734f3ced1df898990bcd92de40abce475907db`.

The historical names `LINEAR_CUDA` and
`get_cuda_linear_interpolation` execute portable PyTorch operations on CPU or
MPS. Raw Warp kernels, CUDA graph capture/reset, and CUDA B-spline-knot kernels
raise `NotImplementedError`. `solve_pose` requires an explicit `goal_state`;
pose-to-joint conversion remains the responsibility of the IK compatibility
layer. Sampled edge checks are deterministic but are not a continuous
collision certificate.
