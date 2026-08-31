"""Compatibility facade for cuRobo's fused CUDA kinematics autograd function."""

from typing import Optional, Tuple

import torch
from torch.autograd import Function

from curobo._src.curobolib.backends import kinematics as kinematics_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_bool_tensors,
    check_float32_tensors,
    check_int16_tensors,
    check_int32_tensors,
    check_int8_tensors,
)
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


class KinematicsFusedFunction(Function):
    @staticmethod
    def create_buffers(batch: int, horizon: int, kinematics_config: KinematicsParams, device_cfg: DeviceCfg = DeviceCfg()):
        zeros = lambda *shape: torch.zeros(shape, device=device_cfg.device, dtype=device_cfg.dtype)
        link_position = zeros(batch, horizon, kinematics_config.num_pose_links, 3)
        link_quaternion = zeros(batch, horizon, kinematics_config.num_pose_links, 4)
        robot_spheres = zeros(batch, horizon, kinematics_config.num_spheres, 4)
        jacobian = zeros(batch, horizon, kinematics_config.num_pose_links, 6, kinematics_config.num_dof)
        com = zeros(batch, horizon, 4)
        cumul_mat = zeros(batch, horizon, kinematics_config.num_links, 3, 4)
        grad_out_q = zeros(batch, horizon, kinematics_config.num_dof)
        return {
            "batch_link_position": link_position,
            "batch_link_quaternion": link_quaternion,
            "batch_robot_spheres": robot_spheres,
            "batch_com": com,
            "batch_jacobian": jacobian,
            "batch_cumul_mat": cumul_mat,
            "grad_out_q": grad_out_q,
            "grad_out_q_jacobian": torch.zeros_like(grad_out_q),
            "grad_in_link_pos": torch.zeros_like(link_position),
            "grad_in_link_quat": torch.zeros_like(link_quaternion),
            "grad_in_robot_spheres": torch.zeros_like(robot_spheres),
            "grad_in_com": torch.zeros_like(com),
        }

    @staticmethod
    def forward(ctx, joint_seq: torch.Tensor, batch_link_position: torch.Tensor, batch_link_quaternion: torch.Tensor, batch_robot_spheres: torch.tensor, batch_com: torch.Tensor, batch_jacobian: torch.Tensor, batch_cumul_mat: torch.Tensor, kinematics_config: KinematicsParams, grad_out: torch.Tensor, grad_out_q_jacobian: torch.Tensor, grad_in_link_pos: torch.Tensor, grad_in_link_quat: torch.Tensor, grad_in_robot_spheres: torch.Tensor, grad_in_com: torch.Tensor, compute_jacobian: bool, compute_spheres: bool, compute_com: bool, env_query_idx: torch.Tensor, horizon: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "KinematicsFusedFunction consumes CUDA-specific packed buffers; use "
            "curobo.kinematics.Kinematics for CPU/MPS FK, spheres, and Jacobians"
        )

    @staticmethod
    def backward(ctx, grad_in_link_pos: Optional[torch.Tensor], grad_in_link_quat: Optional[torch.Tensor], grad_in_spheres: Optional[torch.Tensor], grad_in_com: Optional[torch.Tensor], grad_in_link_jacobian: Optional[torch.Tensor]):
        del ctx, grad_in_link_pos, grad_in_link_quat, grad_in_spheres, grad_in_com, grad_in_link_jacobian
        raise NotImplementedError(
            "KinematicsFusedFunction's CUDA packed-buffer VJP is unavailable; "
            "use curobo.kinematics.Kinematics with ordinary PyTorch autograd"
        )
