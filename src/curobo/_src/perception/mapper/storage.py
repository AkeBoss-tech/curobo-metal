from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class BlockDataView:
    rgb_grid: torch.Tensor
    coords: torch.Tensor
    num_allocated: int
    origin: torch.Tensor
    voxel_size: float
    block_size: int
    grid_shape: tuple
    color_grid_size: int = 1
    feature_block_grid_size: int = 1
    features: torch.Tensor | None = None
    feature_weight: torch.Tensor | None = None
    feature_dim: int = 0


@dataclass
class OccupiedVoxels:
    centers: torch.Tensor
    block_idx_per_voxel: torch.Tensor
    block_data: BlockDataView
    texture_colors: Optional[torch.Tensor] = None
    texture_valid: Optional[torch.Tensor] = None
    subvoxel_factor: int = 1

    def __len__(self): return len(self.centers)
    def colors_uint8(self, eps=1e-6, prefer_texture=True):
        if prefer_texture and self.texture_colors is not None: return self.texture_colors.to(torch.uint8)
        return self.block_data.rgb_grid.clamp(0,255).to(torch.uint8)
    def features(self, eps=1e-6): return self.block_data.features


@dataclass
class MatchedVoxels:
    voxels: OccupiedVoxels
    block_pool_idx: torch.Tensor
    block_scores: torch.Tensor
    def __len__(self): return len(self.block_scores)
    def scores_per_voxel(self, fill_value=float("nan")): return self.block_scores


class BlockSparseTSDF:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("native Warp block-sparse storage is unavailable; use Mapper")
