# Portable rollout goals and Rosenbrock

`curobo._src.rollout.goal_registry.GoalRegistry` supports V2-style goal,
current-state, seed-goal, multi-environment, and seed-expanded index buffers
on CPU and MPS.  Index buffers are `int32`, retain their caller-selected
device, and can be repeated or transformed by a batch-selection matrix.
`copy_` preserves preallocated buffers where possible and, with
`allow_clone=False`, leaves a missing destination untouched just as a
caller-owned buffer API should.

`RosenbrockRollout` is a differentiable generalized Rosenbrock test rollout.
It accepts `[batch, horizon, action_dim]` actions (including a runtime horizon
different from the configured sampling horizon), has deterministic
seeded Halton sampling, returns V2 rollout metrics/cost collections, and
supports autograd on CPU and fallback-disabled MPS.  `use_cuda_graph=True`
uses the package's shape-stable direct executor: it preserves lifecycle and
reset semantics but does not claim raw CUDA Graph capture or CUDA ABI parity.

The registry and rollout reject hidden device copies and invalid rank/shape
inputs.  Exact CUDA graph scheduling, allocator behavior, and NVIDIA random
stream parity remain outside the portable contract.
