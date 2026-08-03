# Goal, seed, and seeded-IK iteration managers

The portable implementations of `GoalManager`, `SeedManager`, and
`SeedIterationStateManager` preserve the value lifecycle required by V2 solver
calls on CPU and Apple Metal:

- Goal buffers are keyed by problem shape, update existing payload storage in
  place, and retain a larger goal-set allocation for later smaller goal sets.
- Action and trajectory seeds use deterministic, resettable Halton sampling;
  supplied seed layouts are normalized to batch-major optimizer tensors.
- Seeded IK uses per-candidate trust-region acceptance, bounded damping, and
  optional joint-limit convergence checks.

Inputs must already reside on the configured device.  Compound joint and pose
payloads are validated channel-by-channel, and malformed batch/goal-set/seed
layouts fail before an optimizer observes them.  CPU and MPS use normal PyTorch
tensors, including autograd through seed construction and accepted iteration
values. CUDA graph capture, Warp LM steps, streams, and packed raw-buffer ABIs
remain explicit unavailable boundaries; the portable managers do not fabricate
those CUDA-only objects.
