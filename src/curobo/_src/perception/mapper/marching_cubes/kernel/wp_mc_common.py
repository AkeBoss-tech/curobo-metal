"""Raw marching-cubes lookup ABI boundaries."""

from curobo._src.perception.mapper._portable import unsupported_kernel

TRIANGLE_TABLE = ()
NUM_TRIANGLES_TABLE = ()
EDGE_OWNER_OFFSETS = ()
local_edge_to_array_idx = unsupported_kernel
array_idx_to_local_edge = unsupported_kernel
binary_search_int32 = unsupported_kernel
binary_search_int64 = unsupported_kernel
interpolate_edge_vertex = unsupported_kernel
get_edge_vertex = unsupported_kernel


class MCLookupTables:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "raw Warp marching-cubes tables are unavailable; use Mapper.extract_mesh"
        )
