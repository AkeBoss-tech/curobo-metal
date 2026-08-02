# Portable solver configuration compatibility

`IKSolverCfg`, `MPCSolverCfg`, and `TrajOptSolverCfg` accept the pinned cuRobo
V2 factory inputs and create configuration records consumed by the portable
CPU/MPS solvers. Robot YAML paths, `RobotCfg` values, scene mappings, capacity
bounds, tolerances, seed controls, collision flags, timing options, and
optimizer records can therefore remain in application code when switching the
dependency to `curobo-metal`.

Each config validates capacities and timing at construction. `clone(**fields)`
returns an independent, validated configuration; `copy` is an alias; and
`update(**fields)` validates before mutating the existing value. The legacy
direct `TrajOptSolverCfg([], robot_cfg, ...)` constructor is normalized into a
portable `SolverCoreCfg` so old concise call sites remain executable.

The upstream defaults request CUDA graph capture. Portable configs retain that
intent in `requested_use_cuda_graph`, but `use_cuda_graph` is always `False`:
execution uses persistent CPU/MPS state instead of presenting a fake CUDA graph
object. Raw CUDA graph capture, Warp rollout buffers, and the packed CUDA
B-spline implementation remain explicit unsupported backend APIs; high-level
IK, MPC, and trajectory optimization use the existing differentiable PyTorch
paths instead.

MPC retains its upstream fixed command-rate rule: `interpolation_steps` must
be four. TrajOpt preserves `BSPLINE_KNOTS_CUDA` as the compatibility default;
the portable solver maps it to its composed PyTorch interpolation path at
execution rather than invoking a CUDA kernel.
