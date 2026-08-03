# Portable RobotRollout

`curobo._src.rollout.rollout_robot.RobotRollout` runs the pinned V2 rollout
lifecycle with ordinary differentiable PyTorch tensors on CPU and Apple MPS.
It owns independent transition/cost-manager instances for optimizer and metrics
paths, supports deterministic action samples, goal seed expansion, cost toggles,
and state/command forwarding.

When `RobotRolloutCfg.scene_collision_cfg` is supplied without an explicit
checker, the rollout constructs the portable `SceneCollision` implementation.
An explicitly supplied checker always takes precedence.  Goal actions and
goal/index payloads must already reside on the configured device; the rollout
rejects mixed CPU/MPS values rather than making hidden copies that would break
autograd or device-residency guarantees.

`use_cuda_graph=True` remains configuration-compatible but does not create or
emulate CUDA graphs.  Explicit CUDA graph reset/capture APIs raise
`NotImplementedError`; normal rollout, collision, and cost execution stays
eager and device-resident.
