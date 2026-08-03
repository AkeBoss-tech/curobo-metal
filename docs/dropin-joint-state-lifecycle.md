# JointState composition lifecycle

The portable `curobo._src.state.state_joint.JointState` supports the pinned
state lifecycle on CPU and Metal tensors: fixed-width derivative packing,
cloning/copying, state/seed indexing, time scaling, finite differences, and
joint-name reordering.

`JointState.stack(other)` follows cuRobo's historical behavior: it appends
waypoints along the second-to-last trajectory axis; it does not introduce a
new tensor dimension. `append_joints` accepts a locked state with a singleton
or broadcast-compatible batch/trajectory prefix and preserves explicit
velocity, acceleration, and jerk values when supplied. Missing derivative
values become zeros for the appended joints when the active state materializes
that channel. A locked state's `dt` takes precedence, as it does upstream.

Packed B-spline knot append is intentionally unsupported when both states own
knot buffers. The pinned CUDA implementation raises for that same ambiguous
layout; callers should compose trajectories before assigning a shared knot
representation. All portable paths use standard PyTorch operations, preserve
first-order autograd, and work on CPU or fallback-disabled MPS. CUDA JIT
packed-buffer helpers are not exposed as a Metal substitute.
