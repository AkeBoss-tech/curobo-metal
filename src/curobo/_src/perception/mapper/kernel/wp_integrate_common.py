from . import raw_kernel

quat_from_wxyz_array = raw_kernel
vec3_from_array = raw_kernel


def compute_tsdf_weight(depth, voxel_size):
    return max(0.001, min(2.0, 1.0 / max(float(depth) ** 2, 1e-12)))


def floor_div(a, b):
    return int(a) // int(b)


def floor_mod(a, b):
    return int(a) % int(b)
