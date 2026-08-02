# Portable mapper checkpoint persistence

`Mapper.save_blocks()` writes the pinned `curobo.mapper_blocks` metadata envelope and a CPU tensor clone of the portable dense map. `Mapper.import_blocks()` checks voxel geometry, block size, feature/static metadata, and grid center before restoring it into an empty CPU/MPS mapper.

The reader also validates the shape of a V2 compact sparse payload so users receive a clear diagnostic. It does not decode or import it: Warp hash tables, block-pool allocation state, packed fp16 accumulators, and their CUDA ABI have no safe dense Metal interpretation. A sparse payload therefore raises `NotImplementedError` at import time rather than producing an approximate map.

`import_weight` is supported for portable dense checkpoints. It replaces positive voxel confidence with the supplied value and must be at least the mapper's configured minimum TSDF weight.
