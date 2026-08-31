"""Compatibility facade for the CUDA RNEA autograd function."""

from typing import Optional

import torch
from torch.autograd import Function

from curobo._src.curobolib.backends import dynamics as dynamics_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float32_tensors,
    check_int16_tensors,
    check_int8_tensors,
)
from curobo._src.types.device_cfg import DeviceCfg

_CACHE_FLOATS_PER_LINK = 20


class RNEAForwardFunction(Function):
    @staticmethod
    def create_buffers(
        batch_size: int,
        num_dof: int,
        num_links: int,
        device_cfg: DeviceCfg = DeviceCfg(),
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
        ctx,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        tau: torch.Tensor,
        grad_q_buf: torch.Tensor,
        grad_qd_buf: torch.Tensor,
        grad_qdd_buf: torch.Tensor,
        forward_cache: torch.Tensor,
        fixed_transforms: torch.Tensor,
        link_masses_com: torch.Tensor,
        link_inertias: torch.Tensor,
        joint_map_type: torch.Tensor,
        joint_map: torch.Tensor,
        link_map: torch.Tensor,
        joint_offset_map: torch.Tensor,
        gravity: torch.Tensor,
        level_starts: torch.Tensor,
        level_links: torch.Tensor,
        num_links: int,
        num_dof: int,
        n_levels: int,
        threads_per_batch: int,
        f_ext: Optional[torch.Tensor] = None,
        grad_f_ext_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "The raw CUDA-buffer RNEA function has no Metal equivalent; use "
            "curobo._src.robot.dynamics.Dynamics for differentiable CPU/MPS RNEA"
        )

    @staticmethod
    def backward(ctx, grad_tau: torch.Tensor):
        """Keep the pinned autograd member visible without fabricating a VJP."""
        del ctx, grad_tau
        raise NotImplementedError(
            "The raw CUDA-buffer RNEA VJP has no Metal equivalent; use "
            "curobo._src.robot.dynamics.Dynamics for differentiable CPU/MPS RNEA"
        )
