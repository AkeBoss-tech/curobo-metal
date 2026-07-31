"""Raw Warp mesh-alignment ABI boundaries.

Use :mod:`curobo._src.perception.pose_estimation.util` for portable tensor
alignment.
"""

from curobo._src.perception.mapper._portable import unsupported_kernel

quat_from_wxyz_array = unsupported_kernel
vec3_from_array = unsupported_kernel
transform_point_inverse = unsupported_kernel
transform_vector = unsupported_kernel
mesh_surface_distance_query_kernel = unsupported_kernel
jacobian_reduce_kernel = unsupported_kernel
