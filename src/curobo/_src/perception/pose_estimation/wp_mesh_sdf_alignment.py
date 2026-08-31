"""Raw Warp mesh-alignment ABI boundaries.

Use :mod:`curobo._src.perception.pose_estimation.util` for portable tensor
alignment.
"""

from __future__ import annotations

from curobo._src.perception.mapper._portable import unsupported_kernel

try:
    import warp as _wp
except ImportError:
    class _PortableWarp:
        @staticmethod
        def func(function):
            return function

        @staticmethod
        def kernel(*args, **kwargs):
            del args, kwargs
            return lambda function: function

    _wp = _PortableWarp()

wp = _wp


@wp.func
def quat_from_wxyz_array(q: wp.array(dtype=wp.float32)) -> wp.quat:
    return unsupported_kernel(q)


@wp.func
def vec3_from_array(v: wp.array(dtype=wp.float32)) -> wp.vec3:
    return unsupported_kernel(v)


@wp.func
def transform_point_inverse(
    position: wp.vec3,
    quat: wp.quat,
    point: wp.vec3,
) -> wp.vec3:
    return unsupported_kernel(position, quat, point)


@wp.func
def transform_vector(
    quat: wp.quat,
    vec: wp.vec3,
) -> wp.vec3:
    return unsupported_kernel(quat, vec)


@wp.kernel(enable_backward=False)
def mesh_surface_distance_query_kernel(
    observed_points: wp.array(dtype=wp.vec3),
    n_points: wp.int32,
    obj_position: wp.array(dtype=wp.float32),
    obj_quaternion: wp.array(dtype=wp.float32),
    mesh_id: wp.uint64,
    max_distance: wp.float32,
    distance_threshold: wp.float32,
    distance_values: wp.array(dtype=wp.float32),
    gradients_world: wp.array(dtype=wp.vec3),
    valid_mask: wp.array(dtype=wp.int32),
):
    return unsupported_kernel(
        observed_points,
        n_points,
        obj_position,
        obj_quaternion,
        mesh_id,
        max_distance,
        distance_threshold,
        distance_values,
        gradients_world,
        valid_mask,
    )


@wp.kernel(enable_backward=False)
def jacobian_reduce_kernel(
    observed_points: wp.array(dtype=wp.vec3),
    distance_values: wp.array(dtype=wp.float32),
    gradients_world: wp.array(dtype=wp.vec3),
    valid_mask: wp.array(dtype=wp.int32),
    n_points: wp.int32,
    use_huber: wp.int32,
    huber_delta: wp.float32,
    JtJ_out: wp.array(dtype=wp.float32),
    Jtr_out: wp.array(dtype=wp.float32),
    sum_sq_residuals: wp.array(dtype=wp.float32),
    valid_count: wp.array(dtype=wp.int32),
):
    return unsupported_kernel(
        observed_points,
        distance_values,
        gradients_world,
        valid_mask,
        n_points,
        use_huber,
        huber_delta,
        JtJ_out,
        Jtr_out,
        sum_sq_residuals,
        valid_count,
    )
