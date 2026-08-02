# Portable rollout and cost-manager compatibility

`curobo._src.rollout` provides eager, differentiable CPU and MPS execution
for goal indexing, rollout metrics, Rosenbrock rollouts, and robot cost-manager
composition.  Cost terms use the production portable PyTorch implementations;
they retain batch/seed index buffers, mutable enable/disable state, cost and
constraint aggregation, and convergence outputs.

`RobotCostManager.initialize_from_config()` is safely reconfigurable: it
replaces stale component instances, keeps the caller's self-collision
configuration reusable when interpolation scaling is needed, disables
collision terms for robots with no spheres, and does not register a scene
term without its checker.  State tensors are never copied between CPU and
MPS by the manager; a device mismatch raises before evaluation so the caller
keeps control of residency and autograd.

The upstream CUDA implementation overlaps cost terms on CUDA streams and may
capture fixed-shape graphs.  Metal executes the same portable terms eagerly,
so `reset_cuda_graph()` is intentionally unsupported.  This is a functional
portable compatibility boundary, not a CUDA ABI or performance-equivalence
claim.
