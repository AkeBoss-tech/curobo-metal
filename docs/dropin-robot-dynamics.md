# Portable robot dynamics compatibility

`curobo._src.robot.dynamics.Dynamics` runs differentiable inverse dynamics on
CPU and float32 MPS using the production parent-index whole-body RNEA backend.
It accepts position, velocity, and acceleration shaped `[dof]`, `[batch, dof]`,
or `[batch, horizon, dof]`; the torque result preserves that shape.  Named
joints are normalized to the configured robot order.

`f_ext` accepts link-local `[moment_x, moment_y, moment_z, force_x, force_y,
force_z]` wrenches shaped `[links, 6]`, `[flat_batch, links, 6]`, or with the
same leading dimensions as the state.  They are rotated to world coordinates
and subtracted through the geometric Jacobian, so gradients flow to both robot
state and wrench values. `setup_batch_size` retains the expected reusable
buffer lifecycle, while eager PyTorch owns the actual differentiable values.

The facade also exposes portable forward dynamics, mass-matrix, and
semi-implicit rollout helpers for direct adapters.  CUDA packed RNEA buffers,
raw kernel launch ABI, and paired NVIDIA numerical equivalence remain outside
this backend's supported contract.
