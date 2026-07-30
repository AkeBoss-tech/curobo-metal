from dataclasses import dataclass
import torch


def depth_to_colormap(depth, depth_minimum_distance=0.1, depth_maximum_distance=5.0, valid_mask=None, invalid_color=(0,0,0)):
    valid = torch.isfinite(depth) & (depth >= depth_minimum_distance) & (depth <= depth_maximum_distance)
    if valid_mask is not None: valid &= valid_mask
    value = ((depth-depth_minimum_distance)/(depth_maximum_distance-depth_minimum_distance)).clamp(0,1)
    color = torch.stack((value, 1-(2*value-1).abs(), 1-value), -1)
    invalid = color.new_tensor(invalid_color)/255
    return torch.where(valid[...,None], color, invalid).mul(255).to(torch.uint8)


def normals_to_colormap(normals, valid_mask):
    color = ((normals+1)*127.5).clamp(0,255).to(torch.uint8)
    return torch.where(valid_mask[...,None], color, torch.zeros_like(color))


class BlockSparseTSDFRenderer:
    def __init__(self, integrator):
        self.integrator = integrator

    def render(self, intrinsics, pose, image_shape):
        return self.integrator.render(intrinsics, pose, image_shape)

    def render_depth(self, intrinsics, pose, image_shape):
        return self.render(intrinsics, pose, image_shape)[0]

    def render_normals(self, intrinsics, pose, image_shape):
        return self.render(intrinsics, pose, image_shape)[1]

    def render_depth_colormap(self, intrinsics, pose, image_shape):
        depth, _, valid = self.render(intrinsics, pose, image_shape)
        return depth_to_colormap(depth, valid_mask=valid)

    def render_normal_colormap(self, intrinsics, pose, image_shape):
        _, normals, valid = self.render(intrinsics, pose, image_shape)
        return normals_to_colormap(normals, valid)
