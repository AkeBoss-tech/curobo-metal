# Portable gradient optimizer core lifecycle

`GradientOptCore` provides the portable lifecycle behind CPU/MPS
gradient-based optimizers: shape validation, ordinary PyTorch cost/VJP
evaluation, per-problem best tracking, bounded fixed-candidate line searches,
debug traces, warm starts, masked reinitialization, resize/reset/shift hooks,
and deterministic device-resident state.

A zero-scale candidate is always included in a line search. This guarantees a
batch member is never forced to accept an invalid or uphill proposed step.
When directly initialized from the default single-problem configuration, a
batched `[B, H, D]` seed grows the portable runtime state and rollout batch
callbacks to `B` deterministically.

Solver-parameter changes are validated atomically: an invalid iteration
configuration is restored rather than leaving a partially mutated core.
The core calls the owner initial-state callback with a masked reinitialization
mask so quasi-Newton histories can preserve unaffected batch rows.

CUDA Graph dispatch, CUDA kernel line searches, Warp rollout buffers, and raw
pointer ABI behavior remain explicitly unavailable on this backend.
