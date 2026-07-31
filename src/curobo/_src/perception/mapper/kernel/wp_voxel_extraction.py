from curobo._src.perception.mapper._portable import dense_state


def extract_occupied_voxels_block_sparse(tsdf, surface_only=False, sdf_threshold=None, minimum_tsdf_weight=0.1, grid_shape=None):
    state = dense_state(tsdf)
    return state.occupancy.nonzero(as_tuple=False)


extract_surface_voxels_block_sparse = extract_occupied_voxels_block_sparse


def extract_matching_voxels_block_sparse(tsdf, block_mask, surface_only=False, sdf_threshold=None, minimum_tsdf_weight=0.1):
    values = extract_occupied_voxels_block_sparse(tsdf, surface_only, sdf_threshold, minimum_tsdf_weight)
    return values[block_mask[: len(values)].bool()] if len(block_mask) >= len(values) else values
