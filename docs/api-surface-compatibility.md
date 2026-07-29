# API surface compatibility — Wave 7C

This surface targets pinned cuRobo revision
`8e734f3ced1df898990bcd92de40abce475907db` while retaining the established
MotionGen integration vocabulary. It is an adapter, not an assertion that
CUDA kernels or upstream optimizer implementations exist on Metal.

`curobo_metal.api_compat` provides serializable pose, bound, smoothness,
collision, run-weight and offset-waypoint cost values; IK, trajectory and graph
solver values; optimizer/interpolation enums; per-call retry/timeout/retiming
configuration; result/status values; and a strict MotionGen facade.

Supported configuration is compiled into production fields: IK seed count,
iteration limit, convergence tolerances and random seed; trajectory horizon,
time step, iteration limit, interpolation time step, bounds weight and
velocity/acceleration/jerk weights. Joint-space single and batch calls,
warmup, deterministic retry, linear interpolation, time dilation, and graph
cache reset are exposed. Pose calls remain available from the production
facade.

Accepted values are never no-ops. Particle/ES optimizers, retract seeds,
non-converged IK success, multiple explicit trajectory seeds, spline/cubic
interpolation, sweep collision, non-summed collision, run weights, offset
waypoints, graph-only planning, partial IK and finetune modes raise
`UnsupportedCompatOption` with the rejected field. Runtime timeout raises
`TimeoutError`. CUDA graphs, mesh/voxel/ESDF worlds, attachments and runtime
world mutation retain the production facade's explicit rejection behavior.

The canonical machine-readable classification is
`artifacts/correctness/api_surface.json`.
