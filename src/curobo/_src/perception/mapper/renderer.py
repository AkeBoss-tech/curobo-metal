from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from curobo._src.types.pose import Pose
from curobo.logging import log_and_raise
wp = None


@dataclass
class BlockSparseTSDFRendererCfg:
    """Source-compatible rendering thresholds for a mapper integrator."""

    depth_minimum_distance: float = 0.2
    depth_maximum_distance: float = 15.0
    minimum_tsdf_weight: float = 0.2


@dataclass(frozen=True)
class _RenderInputs:
    """Normalized camera inputs and their output-shape convention."""

    intrinsics: torch.Tensor
    positions: torch.Tensor
    quaternions: torch.Tensor
    camera_count: int
    single_camera: bool


def depth_to_colormap(
    depth: torch.Tensor,
    depth_minimum_distance: float = 0.1,
    depth_maximum_distance: float = 5.0,
    valid_mask: Optional[torch.Tensor] = None,
    invalid_color: Tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    valid = torch.isfinite(depth) & (depth >= depth_minimum_distance) & (depth <= depth_maximum_distance)
    if valid_mask is not None: valid &= valid_mask
    value = ((depth-depth_minimum_distance)/(depth_maximum_distance-depth_minimum_distance)).clamp(0,1)
    color = torch.stack((value, 1-(2*value-1).abs(), 1-value), -1)
    invalid = color.new_tensor(invalid_color)/255
    return torch.where(valid[...,None], color, invalid).mul(255).to(torch.uint8)


def normals_to_colormap(
    normals: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    color = ((normals+1)*127.5).clamp(0,255).to(torch.uint8)
    return torch.where(valid_mask[...,None], color, torch.zeros_like(color))


class BlockSparseTSDFRenderer:
    def __init__(
        self,
        integrator,
    ) -> None:
        self.integrator = integrator
        self.config = BlockSparseTSDFRendererCfg(
            depth_minimum_distance=integrator.config.depth_minimum_distance,
            depth_maximum_distance=integrator.config.depth_maximum_distance,
            minimum_tsdf_weight=integrator.config.minimum_tsdf_weight,
        )
        if hasattr(integrator, "device"):
            self.device = torch.device(integrator.device)
        elif getattr(integrator._tsdf, "_portable_sparse", False):
            self.device = integrator._tsdf.device
        else:
            self.device = torch.device(integrator.config.device)
        self._buffer_size = 0
        self._hit_points = None
        self._hit_normals = None
        self._hit_colors = None
        self._hit_depths = None
        self._hit_mask = None

    def _ensure_buffers(self, n_pixels: int, include_color: bool = False) -> None:
        """Allocate reusable output buffers without forcing an RGB allocation."""
        if self._buffer_size < n_pixels:
            self._hit_points = torch.zeros((n_pixels, 3), dtype=torch.float32, device=self.device)
            self._hit_normals = torch.zeros((n_pixels, 3), dtype=torch.float32, device=self.device)
            self._hit_depths = torch.zeros(n_pixels, dtype=torch.float32, device=self.device)
            self._hit_mask = torch.zeros(n_pixels, dtype=torch.uint8, device=self.device)
            self._hit_colors = None
            self._buffer_size = n_pixels
        if include_color and (
            self._hit_colors is None or self._hit_colors.shape[0] < n_pixels
        ):
            self._hit_colors = torch.zeros((n_pixels, 3), dtype=torch.uint8, device=self.device)

    def _normalize_intrinsics(self, intrinsics: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return ``(N, 3, 3)`` intrinsics and whether input was batched."""
        was_batched = (
            (intrinsics.ndim == 2 and intrinsics.shape != (3, 3))
            or intrinsics.ndim == 3
        )
        value = intrinsics.to(self.device, dtype=torch.float32)
        if value.ndim == 1 and value.shape == (4,):
            out = torch.zeros((1, 3, 3), dtype=value.dtype, device=value.device)
            out[0, 0, 0], out[0, 1, 1] = value[0], value[1]
            out[0, 0, 2], out[0, 1, 2], out[0, 2, 2] = value[2], value[3], 1.0
            return out, was_batched
        if value.ndim == 2 and value.shape == (3, 3):
            return value.unsqueeze(0).contiguous(), was_batched
        if value.ndim == 2 and value.shape[1:] == (4,):
            out = torch.zeros((value.shape[0], 3, 3), dtype=value.dtype, device=value.device)
            out[:, 0, 0], out[:, 1, 1] = value[:, 0], value[:, 1]
            out[:, 0, 2], out[:, 1, 2], out[:, 2, 2] = value[:, 2], value[:, 3], 1.0
            return out, was_batched
        if value.ndim == 3 and value.shape[1:] == (3, 3):
            return value.contiguous(), was_batched
        log_and_raise("intrinsics must have shape (3, 3), (4,), (N, 3, 3), or (N, 4).")

    def _normalize_render_inputs(self, intrinsics: torch.Tensor, pose: Pose) -> _RenderInputs:
        """Normalize camera tensors without silently broadcasting cameras."""
        if pose.position is None or pose.quaternion is None:
            log_and_raise("pose must contain position and quaternion tensors.")
        matrices, was_batched = self._normalize_intrinsics(intrinsics)
        positions = pose.position.to(self.device, dtype=torch.float32)
        quaternions = pose.quaternion.to(self.device, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            log_and_raise(f"pose.position must have shape (N, 3), got {positions.shape}.")
        if quaternions.ndim != 2 or quaternions.shape[1:] != (4,):
            log_and_raise(f"pose.quaternion must have shape (N, 4), got {quaternions.shape}.")
        if positions.shape[0] != quaternions.shape[0]:
            log_and_raise("pose.position and pose.quaternion must have the same camera count.")
        camera_count = positions.shape[0]
        if matrices.shape[0] != camera_count:
            log_and_raise("intrinsics and pose must have the same camera count.")
        if camera_count <= 0:
            log_and_raise("render camera count must be positive.")
        return _RenderInputs(
            matrices.contiguous(), positions.contiguous(), quaternions.contiguous(),
            camera_count, camera_count == 1 and not was_batched,
        )

    def render(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backend = getattr(self.integrator, "mapper", None)
        if backend is None:
            return self.integrator._render_sparse(intrinsics, pose, image_shape)
        return backend.render(intrinsics, pose, image_shape)

    def render_depth(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        return self.render(intrinsics, pose, image_shape)[0]

    def render_normals(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        return self.render(intrinsics, pose, image_shape)[1]

    def render_depth_colormap(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        depth, _, valid = self.render(intrinsics, pose, image_shape)
        return depth_to_colormap(depth, valid_mask=valid)

    def render_normal_colormap(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        return normals_to_colormap(normals, valid)

    def render_color(
        self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        backend = getattr(self.integrator, "mapper", None)
        if backend is not None and hasattr(backend, "render_color"):
            return backend.render_color(intrinsics, pose, image_shape)
        depth, normals, valid = self.render(intrinsics, pose, image_shape)
        return depth, normals, torch.zeros(depth.shape + (3,), device=depth.device, dtype=torch.uint8), valid

    def render_color_only(self, intrinsics: torch.Tensor, pose: Pose, image_shape: Tuple[int, int]) -> torch.Tensor:
        return self.render_color(intrinsics, pose, image_shape)[2]

    def render_shaded(
        self,
        intrinsics: torch.Tensor,
        pose: Pose,
        image_shape: Tuple[int, int],
        light_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        ambient: float = 0.3,
        use_color: bool = True,
    ) -> torch.Tensor:
        integrator = getattr(self, "integrator", None)
        if integrator is not None and hasattr(integrator, "render_shaded"):
            return integrator.render_shaded(
                intrinsics, pose, image_shape, light_direction, ambient, use_color
            )
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        light = normals.new_tensor(light_direction)
        light = light / torch.linalg.vector_norm(light).clamp_min(torch.finfo(normals.dtype).eps)
        intensity = (normals * light).sum(-1).clamp_min(0) * (1 - ambient) + ambient
        return torch.where(valid[..., None], (255 * intensity[..., None]).to(torch.uint8), torch.zeros_like(normals, dtype=torch.uint8))
