"""Pinned CUDA kinematics call shapes with an explicit portable boundary."""

from .kinematics_config import KinematicsKernelCfg, KinematicsLaunchCfg


def _unsupported(name):
    raise NotImplementedError(
        f"{name} uses raw CUDA pointer buffers; use curobo.kinematics.Kinematics "
        "for portable CPU/MPS forward kinematics and Jacobians"
    )


def launch_kinematics_forward(
    link_pos, link_quat, batch_center_of_mass, global_cumul_mat, joint_vec,
    fixed_transform, link_masses_com, joint_map_type, joint_map, link_map,
    tool_frame_map, joint_offset_map, batch_size, horizon, n_joints,
    compute_com=False,
):
    del (link_pos, link_quat, batch_center_of_mass, global_cumul_mat, joint_vec,
         fixed_transform, link_masses_com, joint_map_type, joint_map, link_map,
         tool_frame_map, joint_offset_map, batch_size, horizon, n_joints, compute_com)
    return _unsupported("launch_kinematics_forward")


def launch_kinematics_forward_spheres(
    link_pos, link_quat, batch_robot_spheres, batch_center_of_mass,
    global_cumul_mat, joint_vec, fixed_transform, robot_spheres,
    link_masses_com, joint_map_type, joint_map, link_map, tool_frame_map,
    link_sphere_map, joint_offset_map, env_query_idx, num_envs, batch_size,
    horizon, n_joints, num_spheres, output_threads_per_batch,
    write_global_cumul=True, compute_com=False,
):
    del (link_pos, link_quat, batch_robot_spheres, batch_center_of_mass,
         global_cumul_mat, joint_vec, fixed_transform, robot_spheres,
         link_masses_com, joint_map_type, joint_map, link_map, tool_frame_map,
         link_sphere_map, joint_offset_map, env_query_idx, num_envs, batch_size,
         horizon, n_joints, num_spheres, output_threads_per_batch,
         write_global_cumul, compute_com)
    return _unsupported("launch_kinematics_forward_spheres")


def launch_kinematics_forward_spheres_jacobian(
    link_pos, link_quat, batch_robot_spheres, batch_center_of_mass,
    batch_jacobian, global_cumul_mat, joint_vec, fixed_transform,
    robot_spheres, link_masses_com, joint_map_type, joint_map, link_map,
    tool_frame_map, link_sphere_map, link_chain_data, link_chain_offsets,
    joint_links_data, joint_links_offsets, joint_affects_endeffector,
    joint_offset_map, env_query_idx, num_envs, batch_size, horizon, n_joints,
    num_spheres, output_threads_per_batch, write_global_cumul=True,
    compute_com=False,
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
    grad_out, grad_nlinks_pos, grad_nlinks_quat, grad_spheres,
    grad_center_of_mass, batch_center_of_mass, grad_jacobian, global_cumul_mat,
    robot_spheres, link_masses_com, link_map, joint_map, joint_map_type,
    tool_frame_map, link_sphere_map, link_chain_data, link_chain_offsets,
    joint_links_data, joint_links_offsets, joint_affects_endeffector,
    joint_offset_map, env_query_idx, num_envs, batch_size, horizon, n_joints,
    num_spheres, compute_com, compute_jacobian_grad,
):
    del (grad_out, grad_nlinks_pos, grad_nlinks_quat, grad_spheres,
         grad_center_of_mass, batch_center_of_mass, grad_jacobian, global_cumul_mat,
         robot_spheres, link_masses_com, link_map, joint_map, joint_map_type,
         tool_frame_map, link_sphere_map, link_chain_data, link_chain_offsets,
         joint_links_data, joint_links_offsets, joint_affects_endeffector,
         joint_offset_map, env_query_idx, num_envs, batch_size, horizon, n_joints,
         num_spheres, compute_com, compute_jacobian_grad)
    return _unsupported("launch_kinematics_backward")
