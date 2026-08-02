# Portable transition runtime

`curobo._src.transition` provides differentiable CPU and MPS state rollout for
position, velocity, acceleration, and B-spline control spaces.  It accepts a
state table plus per-rollout `start_state_idx`, exposes compact clique action
expansion, command shifting/filtering, and returns `RobotState` values with
production whole-body FK, robot spheres, and (when configured) inverse-dynamics
torques.

The implementation uses normal PyTorch tensors and production Metal-capable
operators.  It does not expose CUDA packed trajectory buffers, CUDA stream or
graph capture, Warp spline kernels, or their pointer ABI.  Those backend-only
paths remain deliberately unavailable rather than silently running on CPU.
