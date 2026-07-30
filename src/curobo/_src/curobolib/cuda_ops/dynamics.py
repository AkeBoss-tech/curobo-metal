"""Compatibility facade for the CUDA RNEA autograd function."""

from typing import Optional

import torch

_CACHE_FLOATS_PER_LINK = 20


class RNEAForwardFunction(torch.autograd.Function):
    @staticmethod
    def create_buffers(
        batch_size: int,
        num_dof: int,
        num_links: int,
        device_cfg=None,
        with_external_forces: bool = False,
    ) -> dict:
        device = getattr(device_cfg, "device", "cpu")
        dtype = getattr(device_cfg, "dtype", torch.float32)
        result = {
            "tau": torch.zeros((batch_size, num_dof), device=device, dtype=dtype),
            "grad_q": torch.zeros((batch_size, num_dof), device=device, dtype=dtype),
            "grad_qd": torch.zeros((batch_size, num_dof), device=device, dtype=dtype),
            "grad_qdd": torch.zeros((batch_size, num_dof), device=device, dtype=dtype),
            "forward_cache": torch.zeros(
                (batch_size, num_links * _CACHE_FLOATS_PER_LINK),
                device=device, dtype=dtype,
            ),
        }
        if with_external_forces:
            result["grad_f_ext"] = torch.zeros(
                (batch_size, num_links, 6), device=device, dtype=dtype
            )
        return result

    @staticmethod
    def forward(
        ctx, q, qd, qdd, tau, grad_q_buf, grad_qd_buf, grad_qdd_buf,
        forward_cache, fixed_transforms, link_masses_com, link_inertias,
        joint_map_type, joint_map, link_map, joint_offset_map, gravity,
        level_starts, level_links, num_links: int, num_dof: int, n_levels: int,
        threads_per_batch: int, f_ext: Optional[torch.Tensor] = None,
        grad_f_ext_buf: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError(
            "The raw CUDA-buffer RNEA function has no Metal equivalent; use "
            "curobo._src.robot.dynamics.Dynamics for differentiable CPU/MPS RNEA"
        )
