# Portable mapper-storage lifecycle

`BlockSparseTSDF` provides the usable cuRobo mapper-storage lifecycle over the
production dense CPU/MPS perception map. It is not a representation of the
upstream CUDA/Warp hash table or block pool.

## Supported portable behavior

- Dense TSDF/weight/occupancy/ESDF/gradient/generation storage on CPU and
  fallback-disabled float32 MPS.
- One or more independent environments. `data` remains the environment-zero
  compatibility view; `get_data(environment)` and `get_stats(environment=...)`
  select one environment, while default statistics aggregate all environments.
- Per-frame observation snapshots and deterministic `new_blocks` diagnostics
  per environment; selected-environment reset without changing other batch
  entries.
- Clone-owned dense exports and self-describing `state_dict` checkpoints,
  validated state loading, dense import, coordinate-cache invalidation, memory
  accounting, and source-shaped diagnostic keys.
- Adapting an existing production mapper when geometry, batch size, block size,
  truncation distance, and weight limit agree; unrelated depth gate settings
  may differ because storage does not own them.

## Explicit boundaries

Raw sparse block allocation/free lists, hash probing/compaction, Warp
conversion, CUDA graph storage, feature-volume/RGB control grids, and Blox
storage are intentionally unavailable. Dense diagnostics describe dense map
coverage rather than claiming CUDA sparse-pool occupancy or performance.
