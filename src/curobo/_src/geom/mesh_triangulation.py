"""Triangle/quad face conversion without Warp."""

import numpy as np
import torch


def triangulate_mesh_faces(faces):
    tensor = torch.as_tensor(faces)
    if tensor.ndim != 2 or tensor.shape[1] not in (3, 4):
        raise ValueError("faces must have shape [F,3] or [F,4]")
    if tensor.shape[1] == 3:
        return tensor
    result = torch.stack((tensor[:, [0,1,2]], tensor[:, [0,2,3]]), 1).reshape(-1,3)
    return result


def triangulate_quads_warp(faces, device=None):
    return triangulate_mesh_faces(faces).to(device=device or torch.as_tensor(faces).device)


def triangulate_quads_kernel(*args, **kwargs):
    raise NotImplementedError("raw Warp triangulation kernels are unavailable")


__all__ = ["triangulate_mesh_faces", "triangulate_quads_kernel", "triangulate_quads_warp"]
