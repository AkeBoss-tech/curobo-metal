# Portable MPC runtime behavior

`curobo.model_predictive_control.ModelPredictiveControl` provides a stateful
CPU/MPS receding-horizon controller over the portable IK and trajectory
solvers. Setup fixes the requested batch and a single tracked tool frame;
fixed pose goals are forwarded through IK to a joint-space endpoint. The
facade exposes goal buffers, warm-start seeds, solve state, debug lifecycle
metadata, and `SceneCollision` ownership.

Pass a portable `SceneCollision` at construction or call
`update_world(SceneCfg)`. The same instance is forwarded to the pose-IK and
trajectory layers, so supported named obstacle cache mutation remains visible
to the controller.

`optimize_next_action` owns a portable action-buffer cursor.  Its first call
builds a cold plan; subsequent calls consume one command each without
unnecessarily re-running trajectory optimization, and re-plan after the
fixed-horizon buffer is exhausted.  `optimize_action_sequence` deliberately
re-optimizes and returns the whole horizon.  Both paths expose the action
buffer, command interval, and solve/cursor state through `MPCSolverResult` and
`debug_dump`.

When an optimizer reports an infeasible batch row, the configured fallback
builds a differentiable bounded joint hold/deceleration plan from the measured
velocity.  `linear`, `cosine`, and `exponential` profiles are available on
CPU/MPS; the status remains failed so a safe fallback is never misreported as
a converged trajectory.  This replaces only the lifecycle semantics of the
upstream CUDA execution manager, not its graph-resident buffer ABI.

This is ordinary PyTorch execution: direct CUDA graph manipulation, raw
CUDA/Warp collision objects, multi-tool/multi-goal/time-varying pose tracking,
and runtime inertial mutation raise explicit `NotImplementedError` boundaries.
The portable trajectory optimizer does not claim NVIDIA CUDA graph or Warp
kernel equivalence.
