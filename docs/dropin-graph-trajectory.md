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
MPS. `TrajOptSolver` supports deterministic seed preparation/ranking, active
joint-name reduction, bounded c-space sampling, runtime tool-pose tracking,
finite-difference retiming, dense interpolation limits, serializable debug
state, c-space solves, and pose solves composed through the portable IK solver.
Calls requesting more returned plans than supplied seeds increase the number of
optimized seeds and results are ranked by actual trajectory objective.

Raw Warp kernels, CUDA graph capture/reset, per-seed variable dt, and CUDA
B-spline-knot kernels remain explicit unavailable boundaries. The historical
`BSPLINE_KNOTS_CUDA` configuration falls back to endpoint-preserving linear
portable interpolation; it is not claimed to reproduce the CUDA spline kernel.
Sampled edge checks are deterministic but are not a continuous collision
certificate.
