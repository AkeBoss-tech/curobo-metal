"""Mesh extraction is exposed by :class:`Mapper` on CPU/MPS.

Keep this compatibility export lazy: ``mesh_extractor`` imports the marching
cubes lookup table module, and eager re-exporting here creates a cycle during
normal mapper construction.
"""

def extract_mesh_block_sparse(*args, **kwargs):
    from curobo._src.perception.mapper.mesh_extractor import extract_mesh_block_sparse as _extract
    return _extract(*args, **kwargs)

__all__ = ["extract_mesh_block_sparse"]
