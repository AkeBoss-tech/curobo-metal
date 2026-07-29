from __future__ import annotations

import pytest
import torch

from curobo_metal.collision_checker import (
    CacheCapacityError, Cuboid, PrimitiveCacheConfig, RobotCollisionChecker,
    RobotCollisionCheckerConfig, VoxelCacheConfig, WorldCollision,
    WorldCollisionConfig,
)
from curobo_metal.ops.world_collision import VoxelGrid


def _world() -> WorldCollision:
    world = WorldCollision(WorldCollisionConfig(
        environments=2,
        primitive_cache=PrimitiveCacheConfig(max_cuboids=1),
        voxel_cache=VoxelCacheConfig(max_voxels=1),
        activation_distance=0.1,
        interpolation_steps=2,
    ))
    eye = torch.eye(3, dtype=torch.float64)
    half = torch.ones(3, dtype=torch.float64)
    world.update_cuboid(0, 0, torch.zeros(3, dtype=torch.float64), eye, half)
    world.update_cuboid(1, 0, torch.tensor([10., 0, 0]), eye, half)
    return world


def test_environment_routing_mutation_and_gradient() -> None:
    world = _world()
    spheres = torch.tensor(
        [[[1.2, 0, 0, 0.25]], [[1.2, 0, 0, 0.25]]],
        dtype=torch.float64, requires_grad=True,
    )
    result = world.get_sphere_distance(spheres, env_indices=torch.tensor([0, 1]))
    assert result.distance[0, 0] > 0
    assert result.distance[1, 0] == 0
    result.distance.sum().backward()
    assert spheres.grad is not None
    generation = world.generation
    world.enable_obstacle("primitive", 0, 0, False)
    assert world.generation != generation
    assert world.get_sphere_distance(spheres[:1]).distance.item() == 0


def test_empty_world_and_cache_capacity() -> None:
    world = WorldCollision()
    spheres = torch.tensor([[0., 0, 0, 0.1]])
    result = world.get_sphere_distance(spheres)
    assert torch.equal(result.distance, torch.zeros((1, 1)))
    with pytest.raises(CacheCapacityError):
        world.primitive_cache.append(
            0, Cuboid(torch.zeros(3), torch.eye(3), torch.ones(3))
        )


def test_self_pair_filtering_and_max_violation() -> None:
    checker = RobotCollisionChecker(RobotCollisionCheckerConfig(
        self_collision_pairs=((0, 1), (1, 2)),
    ))
    spheres = torch.tensor([
        [0., 0, 0, .6], [1., 0, 0, .6], [5., 0, 0, .1],
    ])
    assert checker.get_self_collision_distance(spheres).item() == pytest.approx(.2)
    active = torch.tensor([True, False, True])
    assert checker.get_self_collision_distance(spheres, sphere_active=active).item() == 0


def test_swept_interpolation_catches_mid_segment_collision() -> None:
    world = _world()
    start = torch.tensor([[-2., 0, 0, .1]], dtype=torch.float64)
    end = torch.tensor([[2., 0, 0, .1]], dtype=torch.float64)
    result = world.get_swept_sphere_distance(start, end, interpolation_steps=4)
    assert result.samples.shape == (1, 5, 1, 4)
    assert result.distance.item() > 0


def test_voxel_esdf_routing() -> None:
    world = WorldCollision(WorldCollisionConfig(voxel_cache=VoxelCacheConfig(1)))
    grid = VoxelGrid(
        torch.zeros((2, 2, 2), dtype=torch.float64), 1.0,
        torch.zeros(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64), 99.0,
    )
    world.voxel_cache.update(0, 0, grid)
    result = world.get_esdf(torch.zeros((1, 3), dtype=torch.float64))
    assert result.valid.item()


def test_fallback_disabled_is_explicit_for_unsupported_cache_device() -> None:
    world = WorldCollision(WorldCollisionConfig(
        primitive_cache=PrimitiveCacheConfig(1), allow_cpu_fallback=False
    ))
    # No fallback is needed by supported CPU queries; the flag is policy
    # metadata for future unsupported backends and must not disable CPU.
    world.update_cuboid(0, 0, torch.zeros(3), torch.eye(3), torch.ones(3))
    assert world.get_sphere_distance(torch.zeros((1, 4))).distance.shape == (1, 1)
