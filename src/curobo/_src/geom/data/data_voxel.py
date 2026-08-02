from __future__ import annotations
from dataclasses import dataclass
import torch
from curobo._src.geom.types import VoxelGrid
from ._portable import PortableObstacleData, PortableWarpStruct, inverse_pose, raw_warp


def _as_index(value, *, device: torch.device) -> torch.Tensor:
    """Return a portable Warp-style ``vec3i`` tensor without host copies."""
    value = torch.as_tensor(value, device=device)
    if value.shape[-1:] != (3,):
        raise ValueError("voxel indices and dimensions must end in dimension 3")
    return value.to(dtype=torch.long)


def _feature_scalar(value, features: torch.Tensor) -> torch.Tensor:
    """Make a scalar on the feature tensor's device while retaining gradients."""
    return torch.as_tensor(value, device=features.device, dtype=features.dtype)


def voxel_idx_to_flat(idx, grid_dims) -> torch.Tensor:
    """Convert ``[..., x, y, z]`` voxel indices to C-order flat indices.

    This is the same layout used by the pinned Warp helper: z changes fastest.
    The vectorized form is intentionally usable from regular PyTorch CPU and MPS
    code rather than only from a Warp kernel.
    """
    device = idx.device if isinstance(idx, torch.Tensor) else (
        grid_dims.device if isinstance(grid_dims, torch.Tensor) else torch.device("cpu")
    )
    index = _as_index(idx, device=device)
    dims = _as_index(grid_dims, device=device)
    return index[..., 0] * dims[..., 1] * dims[..., 2] + index[..., 1] * dims[..., 2] + index[..., 2]


def is_voxel_valid(idx, grid_dims) -> torch.Tensor:
    """Return whether each ``[..., 3]`` voxel index lies within the grid."""
    device = idx.device if isinstance(idx, torch.Tensor) else (
        grid_dims.device if isinstance(grid_dims, torch.Tensor) else torch.device("cpu")
    )
    index = _as_index(idx, device=device)
    dims = _as_index(grid_dims, device=device)
    return ((index >= 0) & (index < dims)).all(dim=-1)


def world_to_voxel_idx(local_pt, grid_dims, voxel_size) -> torch.Tensor:
    """Map local points to Warp-compatible nearest voxel indices.

    Warp's integer conversion truncates toward zero; using ``Tensor.to(long)``
    preserves that convention (rather than applying PyTorch's floor rounding).
    """
    if not isinstance(local_pt, torch.Tensor):
        local_pt = torch.as_tensor(local_pt, dtype=torch.get_default_dtype())
    if local_pt.shape[-1:] != (3,):
        raise ValueError("local_pt must end in dimension 3")
    dims = _as_index(grid_dims, device=local_pt.device).to(dtype=local_pt.dtype)
    size = torch.as_tensor(voxel_size, device=local_pt.device, dtype=local_pt.dtype)
    if bool((size <= 0).any()):
        raise ValueError("voxel_size must be positive")
    return (local_pt / size + dims * 0.5).to(dtype=torch.long)


def sample_voxel_sdf(features, layer_start_idx, idx, grid_dims, default_val) -> torch.Tensor:
    """Nearest-neighbour ESDF sample as ``[..., (value, valid)]``.

    Features are flat C-order values, matching ``VoxelData.features`` and the
    pinned Warp function. Invalid lookups deliberately return ``default_val``
    and a zero valid flag rather than indexing out of bounds.
    """
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 1 or not features.is_floating_point():
        raise ValueError("features must be a flat floating-point tensor")
    # Warp reads fp16 storage into fp32 arithmetic. Retain larger user dtypes
    # for ordinary PyTorch callers while preserving fp16's pinned semantics.
    values = features.float() if features.dtype == torch.float16 else features
    index = _as_index(idx, device=values.device)
    dims = _as_index(grid_dims, device=values.device)
    if bool((dims <= 0).any()):
        raise ValueError("grid_dims must be positive")
    valid = is_voxel_valid(index, dims)
    safe = torch.minimum(torch.maximum(index, torch.zeros_like(index)), dims - 1)
    flat = voxel_idx_to_flat(safe, dims)
    start = torch.as_tensor(layer_start_idx, device=values.device, dtype=torch.long)
    value = values[start + flat]
    default = _feature_scalar(default_val, values)
    value = torch.where(valid, value, default)
    return torch.stack((value, valid.to(dtype=value.dtype)), dim=-1)


def sample_voxel_sdf_with_grad(
    features, layer_start_idx, local_pt, grid_dims, voxel_size, default_val
) -> torch.Tensor:
    """Trilinearly sample a flat ESDF and return ``[..., sdf, dx, dy, dz]``.

    Boundary interpolation follows the pinned implementation: only valid
    corners contribute, SDF weights are renormalized, and a derivative axis is
    reported only when both corners of at least one pair are in bounds.
    """
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 1 or not features.is_floating_point():
        raise ValueError("features must be a flat floating-point tensor")
    values = features.float() if features.dtype == torch.float16 else features
    if not isinstance(local_pt, torch.Tensor):
        local_pt = torch.as_tensor(local_pt, device=values.device, dtype=values.dtype)
    else:
        local_pt = local_pt.to(device=values.device, dtype=values.dtype)
    if local_pt.shape[-1:] != (3,):
        raise ValueError("local_pt must end in dimension 3")
    dims_i = _as_index(grid_dims, device=values.device)
    if bool((dims_i <= 0).any()):
        raise ValueError("grid_dims must be positive")
    voxel = torch.as_tensor(voxel_size, device=values.device, dtype=values.dtype)
    if bool((voxel <= 0).any()):
        raise ValueError("voxel_size must be positive")
    default = _feature_scalar(default_val, values)

    # The upstream helper switches to nearest-neighbour for degenerate grids.
    if bool((dims_i < 2).any()):
        sampled = sample_voxel_sdf(
            values, layer_start_idx,
            world_to_voxel_idx(local_pt, dims_i, voxel), dims_i, default,
        )
        return torch.cat((sampled[..., :1], torch.zeros_like(local_pt)), dim=-1)

    dims = dims_i.to(dtype=values.dtype)
    coordinate = local_pt / voxel + dims * 0.5 - 0.5
    base = torch.floor(coordinate).to(dtype=torch.long)
    fraction = coordinate - base.to(dtype=values.dtype)

    offsets = torch.tensor(
        [[x, y, z] for x in range(2) for y in range(2) for z in range(2)],
        device=values.device,
        dtype=torch.long,
    )
    corner_idx = base.unsqueeze(-2) + offsets
    valid = is_voxel_valid(corner_idx, dims_i)
    safe = torch.minimum(torch.maximum(corner_idx, torch.zeros_like(corner_idx)), dims_i - 1)
    flat = voxel_idx_to_flat(safe, dims_i)
    start = torch.as_tensor(layer_start_idx, device=values.device, dtype=torch.long)
    samples = values[start + flat]
    samples = torch.where(valid, samples, default)

    bits = offsets.to(dtype=values.dtype)
    weights = torch.where(
        bits == 1,
        fraction.unsqueeze(-2),
        1 - fraction.unsqueeze(-2),
    ).prod(dim=-1)
    valid_weight = weights * valid.to(dtype=values.dtype)
    weight_sum = valid_weight.sum(dim=-1)
    weighted_sum = (samples * valid_weight).sum(dim=-1)
    sdf = torch.where(weight_sum > 0, weighted_sum / weight_sum.clamp_min(torch.finfo(values.dtype).eps), default)

    def _axis_gradient(axis: int) -> torch.Tensor:
        # Offset enumeration is x/y/z nested. These are the four low/high
        # corner pairs along x, y, and z respectively.
        pairs = (([0, 1, 2, 3], [4, 5, 6, 7]),
                 ([0, 1, 4, 5], [2, 3, 6, 7]),
                 ([0, 2, 4, 6], [1, 3, 5, 7]))
        low_positions = torch.tensor(pairs[axis][0], device=values.device)
        high_positions = torch.tensor(pairs[axis][1], device=values.device)
        other_axes = [item for item in range(3) if item != axis]
        pair_bits = offsets[low_positions][:, other_axes].to(dtype=values.dtype)
        pair_fraction = fraction[..., other_axes].unsqueeze(-2)
        weight = torch.where(pair_bits == 1, pair_fraction, 1 - pair_fraction).prod(dim=-1)
        pair_valid = valid[..., low_positions] & valid[..., high_positions]
        diff = samples[..., high_positions] - samples[..., low_positions]
        numerator = (diff * weight * pair_valid.to(dtype=values.dtype)).sum(dim=-1)
        denominator = (weight * pair_valid.to(dtype=values.dtype)).sum(dim=-1)
        return torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(torch.finfo(values.dtype).eps) / voxel,
            torch.zeros_like(numerator),
        )

    # A compact tensor formulation avoids Python scalar control flow over the
    # query batch. The small loop only enumerates the three Cartesian axes.
    gradient = torch.stack([_axis_gradient(axis) for axis in range(3)], dim=-1)
    return torch.cat((sdf.unsqueeze(-1), gradient), dim=-1)

class VoxelDataWarp(PortableWarpStruct): pass
@dataclass(init=False)
class VoxelData(PortableObstacleData):
    @classmethod
    def create_cache(cls,max_n,num_envs,device_cfg,max_voxels=1):
        o=cls._base(max_n,num_envs,device_cfg); o.max_voxels=max_voxels
        o.features=torch.zeros((num_envs,max_n,max_voxels),**device_cfg.as_torch_dict())
        o.xyzr=torch.zeros((num_envs,max_n,max_voxels,4),**device_cfg.as_torch_dict())
        o.params=torch.zeros((num_envs,max_n,4),**device_cfg.as_torch_dict())
        o.dims=torch.zeros((num_envs,max_n,4),**device_cfg.as_torch_dict()); o.max_esdf_distance=100.;o._grids={}; return o
    @classmethod
    def from_voxel_grid(cls,voxel_grid,device_cfg,env_idx=0,num_envs=1,max_n=1):
        n=voxel_grid.feature_tensor.numel() if voxel_grid.feature_tensor is not None else 1
        o=cls.create_cache(max_n,num_envs,device_cfg,n); o.load_batch([voxel_grid],env_idx); return o
    @classmethod
    def from_scene_cfg(cls,scene_cfg,device_cfg,env_idx=0,num_envs=1,max_n=None):
        grids=scene_cfg.voxel; n=max([g.feature_tensor.numel() if g.feature_tensor is not None else 1 for g in grids]+[1])
        o=cls.create_cache(max_n or max(len(grids),1),num_envs,device_cfg,n); o.load_batch(grids,env_idx); return o
    @classmethod
    def from_batch_scene_cfg(cls,scene_cfg_list,device_cfg,max_n=None):
        grids=[g for s in scene_cfg_list for g in s.voxel]; n=max([g.feature_tensor.numel() if g.feature_tensor is not None else 1 for g in grids]+[1])
        o=cls.create_cache(max_n or max([len(s.voxel) for s in scene_cfg_list]+[1]),len(scene_cfg_list),device_cfg,n)
        for i,s in enumerate(scene_cfg_list):o.load_batch(s.voxel,i)
        return o
    def load_batch(self,grids,env_idx):
        if len(grids)>self.max_n:raise ValueError("voxel cache capacity exceeded")
        self.clear(env_idx)
        for g in grids:self.update_data(g,env_idx)
    def update_data(self,g:VoxelGrid,env_idx=0,name=None):
        key=name or g.name
        if self.has_name(key,env_idx):i=self.get_idx(key,env_idx)
        else:
            i=self.get_active_count(env_idx)
            if i>=self.max_n:raise ValueError("voxel cache capacity exceeded")
            self.names[env_idx][i]=key;self.count[env_idx]+=1
        if g.feature_tensor is not None:self.features[env_idx,i,:g.feature_tensor.numel()]=g.feature_tensor.reshape(-1).to(self.device_cfg.device)
        if g.xyzr_tensor is not None:self.xyzr[env_idx,i,:g.xyzr_tensor.reshape(-1,4).shape[0]]=g.xyzr_tensor.reshape(-1,4).to(self.device_cfg.device)
        self.dims[env_idx,i,:3]=torch.as_tensor(g.dims,**self.device_cfg.as_torch_dict());self.dims[env_idx,i,3]=g.voxel_size
        self.params[env_idx,i]=self.dims[env_idx,i]
        self.inv_pose[env_idx,i,:7]=inverse_pose(g.pose or [0,0,0,1,0,0,0],self.device_cfg);self.enable[env_idx,i]=1;self._grids[(env_idx,key)]=g
    def update_features(self,features,name,env_idx=0):
        i=self.get_idx(name,env_idx);self.features[env_idx,i,:features.numel()]=features.reshape(-1)
    def get_voxel_grid(self,name,env_idx=0):return self._grids[(env_idx,name)]
    def get_grid_shape(self,env_idx=0,name=None,idx=0):
        if name is not None:idx=self.get_idx(name,env_idx)
        d=self.dims[env_idx,idx];return torch.Size([round(float(d[j]/d[3])) for j in range(3)])
is_obs_enabled=load_obstacle_transform=compute_local_sdf=compute_local_sdf_with_grad=raw_warp
__all__=["VoxelData","VoxelDataWarp","is_voxel_valid","sample_voxel_sdf","sample_voxel_sdf_with_grad","voxel_idx_to_flat","world_to_voxel_idx","is_obs_enabled","load_obstacle_transform","compute_local_sdf","compute_local_sdf_with_grad"]
