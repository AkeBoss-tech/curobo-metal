from . import raw_kernel

compute_aabb_block_bounds = raw_kernel
stamp_scene_obstacles = raw_kernel
stamp_obstacles = raw_kernel


def clear_static_channel(tsdf_data):
    if hasattr(tsdf_data, "static"):
        tsdf_data.static.zero_()
    return tsdf_data
