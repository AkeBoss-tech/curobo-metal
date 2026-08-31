"""Pinned CUDA kinematics call shapes with an explicit portable boundary."""

import torch

from .kinematics_config import KinematicsKernelCfg, KinematicsLaunchCfg

get_runtime = launch_kernel = None


def _unsupported(name):
    raise NotImplementedError(
        f"{name} uses raw CUDA pointer buffers; use curobo.kinematics.Kinematics "
        "for portable CPU/MPS forward kinematics and Jacobians"
    )


def launch_kinematics_forward(
    link_pos: torch.Tensor, link_quat: torch.Tensor, batch_center_of_mass: torch.Tensor, global_cumul_mat: torch.Tensor, joint_vec: torch.Tensor,
    fixed_transform: torch.Tensor, link_masses_com: torch.Tensor, joint_map_type: torch.Tensor, joint_map: torch.Tensor, link_map: torch.Tensor,
    tool_frame_map: torch.Tensor, joint_offset_map: torch.Tensor, batch_size: int, horizon: int, n_joints: int,
    compute_com: bool = False,
):
    del (link_pos, link_quat, batch_center_of_mass, global_cumul_mat, joint_vec,
         fixed_transform, link_masses_com, joint_map_type, joint_map, link_map,
         tool_frame_map, joint_offset_map, batch_size, horizon, n_joints, compute_com)
    return _unsupported("launch_kinematics_forward")


def launch_kinematics_forward_spheres(
    link_pos: torch.Tensor, link_quat: torch.Tensor, batch_robot_spheres: torch.Tensor, batch_center_of_mass: torch.Tensor,
    global_cumul_mat: torch.Tensor, joint_vec: torch.Tensor, fixed_transform: torch.Tensor, robot_spheres: torch.Tensor,
    link_masses_com: torch.Tensor, joint_map_type: torch.Tensor, joint_map: torch.Tensor, link_map: torch.Tensor, tool_frame_map: torch.Tensor,
    link_sphere_map: torch.Tensor, joint_offset_map: torch.Tensor, env_query_idx: torch.Tensor, num_envs: int, batch_size: int,
    horizon: int, n_joints: int, num_spheres: int, output_threads_per_batch: int,
    write_global_cumul: bool = True, compute_com: bool = False,
):
    del (link_pos, link_quat, batch_robot_spheres, batch_center_of_mass,
         global_cumul_mat, joint_vec, fixed_transform, robot_spheres,
         link_masses_com, joint_map_type, joint_map, link_map, tool_frame_map,
         link_sphere_map, joint_offset_map, env_query_idx, num_envs, batch_size,
         horizon, n_joints, num_spheres, output_threads_per_batch,
         write_global_cumul, compute_com)
    return _unsupported("launch_kinematics_forward_spheres")


def launch_kinematics_forward_spheres_jacobian(
    link_pos: torch.Tensor, link_quat: torch.Tensor, batch_robot_spheres: torch.Tensor, batch_center_of_mass: torch.Tensor,
    batch_jacobian: torch.Tensor, global_cumul_mat: torch.Tensor, joint_vec: torch.Tensor, fixed_transform: torch.Tensor,
    robot_spheres: torch.Tensor, link_masses_com: torch.Tensor, joint_map_type: torch.Tensor, joint_map: torch.Tensor, link_map: torch.Tensor,
    tool_frame_map: torch.Tensor, link_sphere_map: torch.Tensor, link_chain_data: torch.Tensor, link_chain_offsets: torch.Tensor,
    joint_links_data: torch.Tensor, joint_links_offsets: torch.Tensor, joint_affects_endeffector: torch.Tensor,
    joint_offset_map: torch.Tensor, env_query_idx: torch.Tensor, num_envs: int, batch_size: int, horizon: int, n_joints: int,
    num_spheres: int, output_threads_per_batch: int, write_global_cumul: bool = True,
    compute_com: bool = False,
):
    del (link_pos, link_quat, batch_robot_spheres, batch_center_of_mass,
         batch_jacobian, global_cumul_mat, joint_vec, fixed_transform,
         robot_spheres, link_masses_com, joint_map_type, joint_map, link_map,
         tool_frame_map, link_sphere_map, link_chain_data, link_chain_offsets,
         joint_links_data, joint_links_offsets, joint_affects_endeffector,
         joint_offset_map, env_query_idx, num_envs, batch_size, horizon, n_joints,
         num_spheres, output_threads_per_batch, write_global_cumul, compute_com)
    return _unsupported("launch_kinematics_forward_spheres_jacobian")


def launch_kinematics_backward(
    grad_out: torch.Tensor, grad_nlinks_pos: torch.Tensor, grad_nlinks_quat: torch.Tensor, grad_spheres: torch.Tensor,
    grad_center_of_mass: torch.Tensor, batch_center_of_mass: torch.Tensor, grad_jacobian: torch.Tensor, global_cumul_mat: torch.Tensor,
    robot_spheres: torch.Tensor, link_masses_com: torch.Tensor, link_map: torch.Tensor, joint_map: torch.Tensor, joint_map_type: torch.Tensor,
    tool_frame_map: torch.Tensor, link_sphere_map: torch.Tensor, link_chain_data: torch.Tensor, link_chain_offsets: torch.Tensor,
    joint_links_data: torch.Tensor, joint_links_offsets: torch.Tensor, joint_affects_endeffector: torch.Tensor,
    joint_offset_map: torch.Tensor, env_query_idx: torch.Tensor, num_envs: int, batch_size: int, horizon: int, n_joints: int,
    num_spheres: int, compute_com: bool, compute_jacobian_grad: bool,
):
    del (grad_out, grad_nlinks_pos, grad_nlinks_quat, grad_spheres,
         grad_center_of_mass, batch_center_of_mass, grad_jacobian, global_cumul_mat,
         robot_spheres, link_masses_com, link_map, joint_map, joint_map_type,
         tool_frame_map, link_sphere_map, link_chain_data, link_chain_offsets,
         joint_links_data, joint_links_offsets, joint_affects_endeffector,
         joint_offset_map, env_query_idx, num_envs, batch_size, horizon, n_joints,
         num_spheres, compute_com, compute_jacobian_grad)
    return _unsupported("launch_kinematics_backward")
