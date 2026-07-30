from .dynamics_config import DynamicsKernelCfg, DynamicsLaunchCfg


def _unsupported(name):
    raise NotImplementedError(
        f"{name} uses raw CUDA RNEA buffers; use curobo._src.robot.dynamics.Dynamics "
        "for portable CPU/MPS inverse dynamics"
    )


def launch_rnea_forward(tau, q, qd, qdd, fixed_transforms, link_masses_com, link_inertias, joint_map_type, joint_map, link_map, joint_offset_map, gravity, level_starts, level_links, forward_cache, batch_size, num_links, num_dof, n_levels, threads_per_batch=1, f_ext=None):
    return _unsupported("launch_rnea_forward")


def launch_rnea_backward(grad_q, grad_qd, grad_qdd, grad_tau, q, qd, fixed_transforms, link_masses_com, link_inertias, joint_map_type, joint_map, link_map, joint_offset_map, gravity, level_starts, level_links, forward_cache, batch_size, num_links, num_dof, n_levels, threads_per_batch=1, grad_f_ext=None):
    return _unsupported("launch_rnea_backward")
