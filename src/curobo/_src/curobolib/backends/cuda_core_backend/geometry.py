import torch
from curobo_metal.ops.collision import sphere_sphere_signed_distance
from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend._launch import LaunchConfig
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.util import ceil_div
from curobo._src.util.logging import log_and_raise
from .geometry_config import GeometryKernelCfg

COLLISION_PAIR_SIZE = 8
STATIC_SMEM_OVERHEAD = COLLISION_PAIR_SIZE * 33


def self_collision_distance(
    out_distance: torch.Tensor,
    out_vec: torch.Tensor,
    pair_distance: torch.Tensor,
    sparse_index: torch.Tensor,
    robot_spheres: torch.Tensor,
    sphere_padding: torch.Tensor,
    weight: torch.Tensor,
    pair_locations: torch.Tensor,
    block_batch_max_value: torch.Tensor,
    block_batch_max_index: torch.Tensor,
    num_blocks_per_batch: int,
    max_threads_per_block: int,
    batch_size: int,
    horizon: int,
    nspheres: int,
    num_collision_pairs: int,
    store_pair_distance: bool,
    compute_grad: bool,
):
    pairs = pair_locations.reshape(-1, 2).to(dtype=torch.int64)
    spheres = robot_spheres.reshape(batch_size * horizon, nspheres, 4)
    padding = float(sphere_padding.max().item()) if sphere_padding.numel() else 0.0
    result = sphere_sphere_signed_distance(spheres, pairs, padding=padding)
    penetration = (-result.distances).clamp_min(0)
    weighted = penetration * weight.reshape(-1)[0]
    reduced, winners = weighted.max(dim=-1)
    out_distance.copy_(reduced.reshape_as(out_distance))
    if store_pair_distance and pair_distance.numel():
        pair_distance.copy_(weighted.reshape_as(pair_distance))
    if sparse_index.numel():
        sparse_index.copy_(winners.to(sparse_index.dtype).reshape_as(sparse_index))
    if compute_grad and out_vec.numel():
        gradient = -result.gradients
        chosen = gradient.gather(1, winners[:, None, None, None].expand(-1, 1, 2, 3)).squeeze(1)
        out_vec.copy_(chosen.reshape_as(out_vec))
    return [out_distance, out_vec, pair_distance, sparse_index]
