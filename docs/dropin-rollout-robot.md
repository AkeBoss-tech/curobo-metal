# Portable `RobotRollout`

`curobo._src.rollout.rollout_robot.RobotRollout` composes the portable
`RobotStateTransition` and `RobotCostManager` backends on CPU or MPS.  It
keeps separate optimization, constraint, hybrid, metric, and convergence
collections, maintains goal/seed state across updates, supports deterministic
bounded action sampling, and remains differentiable through the ordinary
PyTorch rollout path.

Goal updates retain the existing particle registry for value-only changes, so
successive optimizer iterations keep their state lifecycle.  Changing the
problem batch, seed multiplier, or goal-index layout rebuilds that registry
instead of reusing stale index tensors.  Actions must be `[batch, horizon,
dof]` floating tensors on the configured device; evaluation never hides a
CPU/MPS transfer.  Configuration-owned action bounds are normalised to the
selected device only when generating random samples, which permits lightweight
third-party transition adapters while preserving output residency.

CUDA graph capture/replay, CUDA streams, and the upstream packed CUDA robot
model ABI are deliberately not emulated.  `reset_cuda_graph()` therefore
raises `NotImplementedError`, while normal `use_cuda_graph=True` configuration
loads and executes eagerly on the selected portable device.
