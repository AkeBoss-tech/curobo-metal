# Portable trajectory execution and seeds

`curobo._src.util.trajectory_execution_manager`, `trajectory_seed_generator`,
and `trajectory` execute with ordinary PyTorch tensors on CPU and MPS.  The
command manager keeps the pinned state/action/metrics lifecycle, warm-start
shifts actions while repeating the terminal action, and exposes commands from
the requested interpolated window.

Linear, cubic (Catmull--Rom), and quintic (smoothstep) retiming preserve
endpoints, batching, device residency, and position autograd.  Seed generation
provides constant, endpoint-interpolated, and acceleration-integrated
deceleration seeds.  This is a portable behavioral implementation, not a claim
of byte-identical SciPy/CUDA numerical output.

`QUARTIC` remains unavailable because the pinned upstream implementation itself
raises for it. `BSPLINE_KNOTS_CUDA` and raw Warp interpolation kernels remain
explicitly unavailable: they depend on cuRobo's CUDA spline/Warp ABI. Use
`LINEAR`, `CUBIC`, or `QUINTIC` on CPU/MPS instead.
