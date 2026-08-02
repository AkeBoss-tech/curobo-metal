"""Portable V2 self-collision parameter preprocessing behavior."""

from __future__ import annotations

import pytest
import torch

from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg


def test_distance_matrix_uses_v2_negative_infinity_disable_sentinel() -> None:
    distances = torch.tensor(
        [
            [-torch.inf, 0.25, torch.inf],
            [0.25, -torch.inf, -torch.inf],
            [torch.inf, -torch.inf, -torch.inf],
        ]
    )
    padding = torch.tensor([0.0, 0.1, 0.2])
    cfg = SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(distances, padding)

    # +inf remains enabled in the pinned implementation; -inf is the only
    # disabled-pair sentinel and only the upper triangle defines pair order.
    assert cfg.num_spheres == 3
    assert cfg.collision_pairs.dtype == torch.int64
    assert cfg.collision_pairs.tolist() == [[0, 1], [0, 2]]
    torch.testing.assert_close(cfg.sphere_padding, padding)


def test_link_pair_compilation_is_symmetric_and_does_not_mutate_inputs() -> None:
    links = ["arm", "wrist", "tool"]
    link_ids = {"arm": 0, "wrist": 1, "tool": 2}
    ignores = {"arm": ["wrist"]}
    link_padding = {"arm": 0.05, "tool": 0.02}
    spheres = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.4],
        ]
    )
    sphere_link_ids = torch.tensor([0, 0, 1, 2])
    distances, padding = (
        SelfCollisionKinematicsCfg.compute_sphere_pair_distance_with_link_pair_ignores(
            links,
            link_ids,
            ignores,
            link_padding,
            spheres,
            sphere_link_ids,
            DeviceCfg(),
        )
    )

    assert link_padding == {"arm": 0.05, "tool": 0.02}
    torch.testing.assert_close(padding, torch.tensor([0.05, 0.05, 0.0, 0.02]))
    assert torch.isneginf(distances[0, 1])  # same link
    assert torch.isneginf(distances[0, 2])  # ignored in just one direction
    assert torch.isneginf(distances[2, 1])
    torch.testing.assert_close(distances[0, 3], torch.tensor(0.57))
    torch.testing.assert_close(distances[1, 3], torch.tensor(0.67))
    torch.testing.assert_close(distances[2, 3], torch.tensor(0.72))
    torch.testing.assert_close(distances, distances.T)

    cfg = SelfCollisionKinematicsCfg.create_from_link_pairs(
        links, link_ids, ignores, link_padding, spheres, sphere_link_ids, DeviceCfg()
    )
    assert cfg.collision_pairs.tolist() == [[0, 3], [1, 3], [2, 3]]


def test_config_validation_and_v2_pair_launch_metadata() -> None:
    with pytest.raises(ValueError, match="shape"):
        SelfCollisionKinematicsCfg(num_spheres=2, sphere_padding=torch.zeros(1))
    with pytest.raises(ValueError, match="canonical"):
        SelfCollisionKinematicsCfg(num_spheres=2, collision_pairs=torch.tensor([[1, 0]]))
    with pytest.raises(ValueError, match="duplicate"):
        SelfCollisionKinematicsCfg(num_spheres=2, collision_pairs=torch.tensor([[0, 1], [0, 1]]))
    with pytest.raises(ValueError, match="same device"):
        SelfCollisionKinematicsCfg(
            num_spheres=2,
            sphere_padding=torch.zeros(2),
            collision_pairs=torch.empty((0, 2), device="meta", dtype=torch.long),
        )

    pairs = torch.stack((torch.arange(1001), torch.arange(1, 1002)), dim=-1)
    cfg = SelfCollisionKinematicsCfg(num_spheres=1002, collision_pairs=pairs)
    assert cfg.num_checks_per_thread == 256
    assert cfg.max_threads_per_block == 512
    assert cfg.num_blocks_per_batch == 1
    assert SelfCollisionKinematicsCfg(num_spheres=3).num_blocks_per_batch == 0


def test_rejects_device_and_link_mapping_errors_without_building_cpu_buffers() -> None:
    spheres = torch.zeros((2, 4))
    with pytest.raises(ValueError, match="device_cfg.device"):
        SelfCollisionKinematicsCfg.compute_sphere_pair_distance_with_link_pair_ignores(
            ["a"], {"a": 0}, {}, {}, spheres, torch.zeros(2, dtype=torch.long), DeviceCfg("meta")
        )
    with pytest.raises(ValueError, match="map every collision link"):
        SelfCollisionKinematicsCfg.compute_sphere_pair_distance_with_link_pair_ignores(
            ["a"], {"a": 0, "b": 1}, {}, {}, spheres, torch.zeros(2, dtype=torch.long), DeviceCfg()
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_mps_link_pair_configuration_remains_resident() -> None:
    device = torch.device("mps")
    cfg = SelfCollisionKinematicsCfg.create_from_link_pairs(
        ["a", "b"],
        {"a": 0, "b": 1},
        {},
        {"a": 0.01},
        torch.tensor([[0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 0.2]], device=device),
        torch.tensor([0, 1], device=device),
        DeviceCfg(device),
    )
    assert cfg.sphere_padding.device.type == cfg.collision_pairs.device.type == "mps"
    assert cfg.collision_pairs.tolist() == [[0, 1]]
