# Robot collision parameters and joint limits

`SelfCollisionKinematicsCfg` preserves cuRoboV2's compact canonical pair
configuration: `-inf` disables a pair when compiling a distance matrix, and
all other upper-triangular values enable it.  The portable implementation adds
CPU/MPS-safe lifecycle helpers that are useful around mutable planning scenes:

- `clone()`, `to(...)`, and `copy_()` retain a compatible tensor buffer where
  possible and otherwise replace it atomically;
- `pair_mask()` exposes the symmetric diagnostic matrix without a CUDA/Warp
  kernel; and
- `reindex_spheres(new_to_old)` reduces or reorders a sphere set, dropping
  pairs that no longer have both endpoints.

`JointLimits` stores each constraint as `[lower, upper] x DOF` tensors and
continues to expose the pinned fields and position aliases.  Its portable
extensions are device-local named reindexing (including a strict subset),
clone-aware `copy_`, named lower/upper queries, `as_dict`, differentiable
`clamp_position`, and `with_position_margin`.  A position margin shrinks only
position bounds and rejects a collapsed range.

All helpers use ordinary PyTorch tensors, preserve CPU/MPS residency, and are
covered with fallback-disabled MPS tests.  They do not emulate cuRobo's raw
Warp/CUDA collision launch-buffer ABI; production self-collision evaluation
routes through the differentiable CPU/MPS collision cost instead.
