# Portable TrajOpt configuration

`curobo._src.solver.solver_trajopt_cfg.TrajOptSolverCfg` preserves the pinned
V2 configuration fields and `create()` factory on CPU and Apple Metal. It
accepts a robot YAML/dictionary/`RobotCfg`, optimizer and rollout
YAML-shaped records, scene/cache settings, batch/goalset capacities, seed
counts, dt bounds, interpolation limits, and deterministic debug/seed
controls.

The factory deep-copies caller records, compiles scene/cache input into a
`SceneCollisionCfg`, preserves metrics and transition records for inspection,
and derives linear interpolation for a transition with
`control_space: POSITION`. `clone()` makes an independent core record and
`update()` is transactional: invalid changes leave the original configuration
untouched.

CUDA graph capture is accepted to retain application configuration, recorded
as `requested_use_cuda_graph`, and always disabled for executable portable
CPU/MPS configuration (`use_cuda_graph` is `False`). The pinned default
`BSPLINE_KNOTS_CUDA` also remains visible as `interpolation_type`; its raw
CUDA spline kernel is not emulated. Use `portable_interpolation_type` to
observe the endpoint-preserving `LINEAR_CUDA` execution fallback, or choose
`LINEAR`, `CUBIC`, or `QUINTIC` explicitly when appropriate.

The configuration deliberately does not fabricate CUDA Graph, Warp rollout,
or raw B-spline-kernel objects. Those ABI-level APIs remain explicit backend
boundaries; `TrajOptSolver` uses the production composed-PyTorch trajectory
optimizer instead.
