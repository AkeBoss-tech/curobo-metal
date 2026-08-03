# Portable MPC solver lifecycle

`curobo._src.solver.solver_mpc.MPCSolver` now exposes a real
`TrajectoryExecutionManager` over normal PyTorch CPU/MPS buffers.  The manager
serves exactly `interpolation_steps` commands from a plan, then asks the solver
for a warm-start replan.  This matches the useful receding-horizon lifecycle
without representing a CUDA graph, stream, or packed device buffer.

`reset_robot_id` preserves queued plans for unselected batch rows and replaces
only the selected row with a current-state seed.  Failed optimizer rows return
the first command from the same safe-deceleration buffer recorded in the MPC
result; no infeasible command is exposed before the fallback plan.

All commands, seeds, and result tensors remain on the configured CPU or Apple
MPS device.  CUDA graph reset/capture, Warp rollout buffers, and raw CUDA
execution objects continue to raise explicit `NotImplementedError` boundaries.
