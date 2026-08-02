# Portable rollout and cost-manager compatibility

`curobo._src.rollout` provides eager, differentiable CPU and MPS execution
for goal indexing, rollout metrics, Rosenbrock rollouts, and robot cost-manager
composition.  Cost terms use the production portable PyTorch implementations;
they retain batch/seed index buffers, mutable enable/disable state, cost and
constraint aggregation, and convergence outputs.

The upstream CUDA implementation overlaps cost terms on CUDA streams and may
capture fixed-shape graphs.  Metal executes the same portable terms eagerly,
so `reset_cuda_graph()` is intentionally unsupported.  This is a functional
portable compatibility boundary, not a CUDA ABI or performance-equivalence
claim.
