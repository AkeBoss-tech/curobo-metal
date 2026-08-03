# Tool-pose value lifecycle

`ToolPose` is the portable FK-output value (`[B,H,L,3/4]`), while
`GoalToolPose` adds a goalset axis (`[B,H,L,G,3/4]`).
`SequenceGoalToolPose` is time-first (`[T,B,L,G,3/4]`) and exposes each frame
as a horizon-one `GoalToolPose` view.

The three types validate rank, final coordinate width, matching position and
quaternion leading dimensions, device/dtype, and unambiguous link names at
construction. They support named link extraction, link reordering, batch/frame
selection that retains their public rank, clone/detach/contiguous/copy, normal
PyTorch device/dtype movement, and first-order autograd on CPU and float32 MPS.
`GoalToolPose.get_goalset()` converts a selected target to the 4D FK-compatible
layout without a host round trip.

The values deliberately use ordinary PyTorch tensors. They do not expose
cuRobo's raw CUDA graph buffers, Warp structs, or CUDA-only packed-kernel ABI;
those platform-specific internals remain explicit non-Metal boundaries.
