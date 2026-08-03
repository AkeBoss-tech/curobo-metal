# Portable robot cost manager

`curobo._src.rollout.cost_manager.RobotCostManager` composes the portable
tool-pose, c-space, self-collision, and scene-collision cost implementations
for regular PyTorch CPU and MPS rollouts.  Its public registry, enable/disable,
batch setup, reset, `update_dt`, cost, and convergence APIs follow cuRobo V2.

Configuration is built with `RobotCostManagerCfg.create(...)`.  Each supplied
cost config must use the manager's `DeviceCfg`; mixing CPU config tensors into
an MPS manager fails at initialization, rather than inserting a hidden copy.
Reconfiguration is transactional for the manager: validation errors leave the
previous valid cost registry usable.  State tensors, torque, collision spheres,
goal indices, and goal poses likewise remain on the configured device.

The manager uses eager native PyTorch dispatch and autograd.  It intentionally
does not expose CUDA cost streams, CUDA event synchronization, CUDA graph
capture, or Warp collision-kernel ABI.  Scene collision still requires a
portable `SceneCollision`/checker, and swept collision retains the existing
sampled query semantics rather than claiming CUDA analytic CCD.
