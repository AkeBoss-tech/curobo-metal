# Portable trajectory execution and seeds

`curobo._src.util.trajectory_execution_manager`, `trajectory_seed_generator`,
and `trajectory` execute with ordinary PyTorch tensors on CPU and MPS.  The
command manager keeps the pinned state/action/metrics lifecycle, warm-start
shifts actions while repeating the terminal action, and exposes commands from
the requested interpolated window.

The manager also offers portable streaming controls: `peek_next_command`,
`command_index`, `remaining_commands`, `reset_command_index`, and
`clear_buffers`. Buffer replacement is validated atomically for shared leading
batch/seed dimensions, horizon structure, device, and dtype, so invalid
updates cannot corrupt an executable queue. State and action widths may differ,
as they do for position-state versus control-action rollouts. Commands remain ordinary PyTorch
views on CPU or MPS and preserve position autograd. The manager intentionally
does not execute commands on an external robot, create a CUDA stream, or
capture a CUDA graph; callers own that transport boundary.

Linear, cubic (Catmull--Rom), and quintic (smoothstep) retiming preserve
endpoints, batching, device residency, and position autograd.  Seed generation
provides constant, endpoint-interpolated, and acceleration-integrated
deceleration seeds. Endpoint interpolation accepts a singleton start state for
a batch of goal sets. Deceleration accepts scalar, per-batch, or ``[batch, 1]``
timing data; a requested stop time is clipped to the seed horizon and creates a
stationary suffix. The discrete acceleration integral is normalized per batch
so every nonzero velocity reaches a rest knot without reversing direction. This
is a portable behavioral implementation, not a claim of byte-identical
SciPy/CUDA numerical output.

`QUARTIC` remains unavailable because the pinned upstream implementation itself
raises for it. `BSPLINE_KNOTS_CUDA` and raw Warp interpolation kernels remain
explicitly unavailable: they depend on cuRobo's CUDA spline/Warp ABI. Use
`LINEAR`, `CUBIC`, or `QUINTIC` on CPU/MPS instead.
