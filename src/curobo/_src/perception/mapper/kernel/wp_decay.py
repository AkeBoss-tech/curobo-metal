from curobo._src.perception.mapper._portable import dense_state


def decay_and_recycle(tsdf, decay_factor=0.95):
    state = dense_state(tsdf)
    state.weight.mul_(decay_factor)
    return state


def launch_recycle(tsdf, num_blocks=None):
    return dense_state(tsdf)


def apply_decay_from_frustum_flags(tsdf, *, num_blocks, time_decay, frustum_decay):
    return decay_and_recycle(tsdf, time_decay * frustum_decay)


def decay_frustum_aware_multi_sensor(tsdf, **kwargs):
    if kwargs.get("lidar_positions") is not None:
        raise NotImplementedError("lidar frustum decay requires CUDA/Warp")
    return decay_and_recycle(
        tsdf, kwargs.get("time_decay", 1.0) * kwargs.get("frustum_decay", 0.5)
    )


def decay_frustum_aware_multi_camera(
    tsdf, intrinsics, cam_positions, cam_quaternions, img_shape,
    depth_minimum_distance=0.1, depth_maximum_distance=10.0,
    time_decay=1.0, frustum_decay=0.5, num_blocks=None,
):
    return decay_and_recycle(tsdf, time_decay * frustum_decay)
