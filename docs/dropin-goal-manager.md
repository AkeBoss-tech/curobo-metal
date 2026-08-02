# GoalManager compatibility

`curobo._src.solver.manager_goal.GoalManager` provides the V2 goal-buffer
lifecycle on CPU and Apple Metal. It creates seed-expanded `GoalRegistry`
buffers, reports structural changes as `(goal, reference_updated)`, updates
preallocated payloads in place, and retains a larger goal-set allocation when
a later request has fewer goals. In that case the first new goal fills unused
cached goal-set slots, as in cuRobo V2.

Joint, pose, and timing values must already live on the configured device;
cross-device calls fail instead of creating an implicit CPU fallback. Optional
state channels that were not materialized during setup are not allocated during
a value-only update, so the backing registry remains stable. CUDA graph
handles, streams, and packed-kernel ABI objects are intentionally not part of
the portable manager.
