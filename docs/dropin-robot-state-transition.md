# Portable RobotStateTransition

`curobo._src.transition.RobotStateTransition` provides differentiable
position, velocity, acceleration, and B-spline action rollouts with ordinary
PyTorch tensors on CPU and Apple MPS.  Planner rollout and robot-command
expansion intentionally own separate transition/buffer lifecycles, matching
the observable CUDA API without retaining CUDA graphs or packed device
buffers.

The facade validates joint width, dtype, and device before a step.  It exposes
the compiled robot's scaled `JointLimits` for action bounds, produces FK,
spheres, and optional inverse-dynamics torque through the portable production
backends, and permits runtime timestep/batch updates.  Autograd remains valid
through both rollout and command generation.

Raw CUDA streams, graph capture, packed transition kernels, and CUDA ABI
buffers are deliberately not emulated.  Dynamics-link inertial mutation is
available only when the portable dynamics model was configured.
