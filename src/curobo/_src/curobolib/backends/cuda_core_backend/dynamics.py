from typing import Optional

import torch

from .dynamics_config import DynamicsKernelCfg, DynamicsLaunchCfg

get_runtime = launch_kernel = None


def _unsupported(name):
    raise NotImplementedError(
        f"{name} uses raw CUDA RNEA buffers; use curobo._src.robot.dynamics.Dynamics "
        "for portable CPU/MPS inverse dynamics"
    )


def launch_rnea_forward(tau: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor, fixed_transforms: torch.Tensor, link_masses_com: torch.Tensor, link_inertias: torch.Tensor, joint_map_type: torch.Tensor, joint_map: torch.Tensor, link_map: torch.Tensor, joint_offset_map: torch.Tensor, gravity: torch.Tensor, level_starts: torch.Tensor, level_links: torch.Tensor, forward_cache: torch.Tensor, batch_size: int, num_links: int, num_dof: int, n_levels: int, threads_per_batch: int = 1, f_ext: Optional[torch.Tensor] = None):
    return _unsupported("launch_rnea_forward")


def launch_rnea_backward(grad_q: torch.Tensor, grad_qd: torch.Tensor, grad_qdd: torch.Tensor, grad_tau: torch.Tensor, q: torch.Tensor, qd: torch.Tensor, fixed_transforms: torch.Tensor, link_masses_com: torch.Tensor, link_inertias: torch.Tensor, joint_map_type: torch.Tensor, joint_map: torch.Tensor, link_map: torch.Tensor, joint_offset_map: torch.Tensor, gravity: torch.Tensor, level_starts: torch.Tensor, level_links: torch.Tensor, forward_cache: torch.Tensor, batch_size: int, num_links: int, num_dof: int, n_levels: int, threads_per_batch: int = 1, grad_f_ext: Optional[torch.Tensor] = None):
    return _unsupported("launch_rnea_backward")
