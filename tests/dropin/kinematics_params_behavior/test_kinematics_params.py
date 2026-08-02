"""Portable behavioral coverage for the cuRoboV2 KinematicsParams record."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.kinematics import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg


def _params(device: torch.device | str = "cpu"):
    return KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=DeviceCfg(torch.device(device))
    ).kinematics_config


def test_tree_csr_inertial_metadata_and_topological_links_are_consistent():
    params = _params()
    assert params.all_link_names[0] == params.base_link
    assert params.link_map.shape == (params.num_links,)
    assert params.link_map[0].item() == -1
    assert torch.all(params.link_map[1:] >= 0)
    assert torch.all(params.link_map[1:] < torch.arange(1, params.num_links))

    assert params.fixed_transforms.shape == (params.num_links, 4, 4)
    assert params.link_masses_com.shape == (params.num_links, 4)
    assert params.link_inertias.shape == (params.num_links, 8)
    torch.testing.assert_close(params.link_inertias[:, -2:], torch.zeros_like(params.link_inertias[:, -2:]))

    assert params.link_chain_offsets.shape == (params.num_links + 1,)
    assert params.link_chain_offsets[0].item() == 0
    assert params.link_chain_offsets[-1].item() == params.link_chain_data.numel()
    assert params.joint_links_offsets.shape == (params.num_dof + 1,)
    assert params.joint_links_offsets[-1].item() == params.joint_links_data.numel()
    assert params.joint_affects_endeffector.shape == (params.num_dof * params.num_pose_links,)
    assert params.link_level_data.numel() == params.num_links
    assert params.n_tree_levels >= 1


def test_multi_environment_sphere_update_reset_copy_and_clone_are_independent():
    params = _params()
    link = "panda_link1"
    params.set_num_envs(3)
    assert params.link_spheres.shape[0] == 3
    original = params.get_link_spheres(link).clone()
    update = original.clone()
    update[..., 0] += 0.125
    params.update_link_spheres(link, update)
    torch.testing.assert_close(params.get_link_spheres(link, 0), update)
    torch.testing.assert_close(params.get_link_spheres(link, 2), update)

    one = original.clone()
    one[..., 1] -= 0.25
    params.update_link_spheres(link, one, config_idx=1)
    torch.testing.assert_close(params.get_link_spheres(link, 1), one)
    torch.testing.assert_close(params.get_link_spheres(link, 0), update)
    params.reset_link_spheres(link)
    for environment in range(params.num_envs):
        torch.testing.assert_close(params.get_link_spheres(link, environment), original)

    clone = params.clone()
    clone.update_link_mass(link, 2.5)
    assert clone.get_link_masses_com(link)[-1].item() == 2.5
    assert params.get_link_masses_com(link)[-1].item() != 2.5
    target = params.clone()
    sphere_buffer = target.link_spheres
    target.copy_(clone)
    assert target.link_spheres is sphere_buffer
    assert target.get_link_masses_com(link)[-1].item() == 2.5


def test_shape_validation_rejects_invalid_environment_or_nonfinite_sphere_updates():
    params = _params()
    with pytest.raises(ValueError, match="positive"):
        params.set_num_envs(0)
    with pytest.raises(ValueError, match="finite"):
        params.update_link_spheres("panda_link1", torch.tensor([[float("nan"), 0, 0, 1]]))
    with pytest.raises(ValueError, match="batched sphere values"):
        params.update_link_spheres(
            "panda_link1", torch.zeros(2, params.get_number_of_spheres("panda_link1"), 4)
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_to_mps_moves_cached_portable_metadata_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    params = _params()
    # Materialize every tensor cache before migrating the value model.
    _ = (params.link_spheres, params.fixed_transforms, params.link_masses_com, params.link_inertias)
    metal = params.to(DeviceCfg(torch.device("mps")))
    assert metal.device_cfg.device.type == "mps"
    for value in (
        metal.link_spheres,
        metal.reference_link_spheres,
        metal.fixed_transforms,
        metal.link_masses_com,
        metal.link_inertias,
        metal.link_chain_data,
        metal.joint_affects_endeffector,
    ):
        assert value.device.type == "mps"
    metal.validate_shapes()
