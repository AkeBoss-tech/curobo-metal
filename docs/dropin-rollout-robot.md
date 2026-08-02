# Portable `RobotRollout`

`curobo._src.rollout.rollout_robot.RobotRollout` composes the portable
`RobotStateTransition` and `RobotCostManager` backends on CPU or MPS.  It
keeps separate optimization, constraint, hybrid, metric, and convergence
collections, maintains goal/seed state across updates, supports deterministic
bounded action sampling, and remains differentiable through the ordinary
PyTorch rollout path.

CUDA graph capture/replay, CUDA streams, and the upstream packed CUDA robot
model ABI are deliberately not emulated.  `reset_cuda_graph()` therefore
raises `NotImplementedError`, while normal `use_cuda_graph=True` configuration
loads and executes eagerly on the selected portable device.
