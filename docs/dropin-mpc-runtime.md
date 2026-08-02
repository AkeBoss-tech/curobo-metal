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

This is ordinary PyTorch execution: direct CUDA graph manipulation, raw
CUDA/Warp collision objects, multi-tool/multi-goal/time-varying pose tracking,
and runtime inertial mutation raise explicit `NotImplementedError` boundaries.
The portable trajectory optimizer does not claim NVIDIA CUDA graph or Warp
kernel equivalence.
