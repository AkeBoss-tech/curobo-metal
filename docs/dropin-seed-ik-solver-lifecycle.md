# Portable seeded-IK solver lifecycle

`curobo._src.solver.seed_ik.SeedIKSolver` implements the pinned cuRobo seeded
IK public surface with a deterministic damped Gauss--Newton/LM solve over
standard PyTorch tensors.  CPU and float32 MPS runs retain batch, seed,
goal-set, current-state velocity, mini-batch, result-ranking, reset and
destroy/recreate behavior.

`SeedIKSolverCfg.create()` accepts a packaged robot config name, an existing
path, a robot mapping, or a `RobotCfg`.  Runtime values are validated before
allocation: iteration grouping, positive mini-batch capacity, damping range,
joint-limit margin, nonnegative weights/tolerances, deterministic sampler
seed, and `DeviceCfg`/`RobotCfg` types are checked explicitly.

`destroy()` releases shape-dependent portable buffers but does **not** make the
solver unusable.  The next `solve_single` or `solve_batch` recreates them;
this matches the expected reusable high-level MotionGen lifecycle without
claiming CUDA graph object persistence.

Current-state velocity constraints require finite, positive `dt` and matching
device/dtype tensors.  Supplied seeds must be finite and have the solver's
device, dtype, batch and DOF dimensions.

The implementation deliberately does not emulate packed Warp LM kernels,
CUDA graph capture, raw CUDA streams, or their ABI/numerical scheduling.
`use_cuda_graph=True` remains accepted for upstream configuration
compatibility, but execution stays ordinary deterministic PyTorch and reports
`cuda_graph: false` in result metrics.
