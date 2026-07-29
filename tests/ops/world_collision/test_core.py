from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from curobo_metal.ops.world_collision import (
    Mesh, VoxelGrid, mesh_distance, query_esdf, sample_voxel_sdf,
    sphere_world_collision,
)
from curobo_metal.reference.world_collision import (
    Mesh as RefMesh, VoxelGrid as RefGrid, mesh_distance as ref_mesh_distance,
    query_esdf as ref_query_esdf, sample_voxel_sdf as ref_sample_voxel_sdf,
)

FIXTURES = Path(__file__).parents[2] / "fixtures" / "world_collision"


def _meshes(dtype=torch.float64, device="cpu"):
    raw = json.loads((FIXTURES / "meshes.json").read_text())
    prod = {
        k: Mesh(torch.tensor(v["vertices"], dtype=dtype, device=device),
                torch.tensor(v["faces"], dtype=torch.int64, device=device), v["watertight"])
        for k, v in raw.items()
    }
    ref = {
        k: RefMesh(np.array(v["vertices"], float), np.array(v["faces"], int), v["watertight"])
        for k, v in raw.items()
    }
    return prod, ref


def _grid(dtype=torch.float64, device="cpu", translation=(0, 0, 0), rotation=None, oob=99):
    raw = json.loads((FIXTURES / "analytic_grid.json").read_text())
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    return (
        VoxelGrid(torch.tensor(raw["values"], dtype=dtype, device=device), raw["voxel_size"],
                  torch.tensor(translation, dtype=dtype, device=device),
                  torch.tensor(rotation, dtype=dtype, device=device), oob),
        RefGrid(np.array(raw["values"], float), raw["voxel_size"], np.array(translation, float), rotation, oob),
    )


def test_mesh_oracle_parity_transforms_masks_ties_and_autograd():
    meshes, refs = _meshes()
    points = torch.tensor([[2., .2, -.3], [0., 0., 0.], [1., .2, .1], [2., 2., 0.]],
                          dtype=torch.float64, requires_grad=True)
    t = torch.zeros((1, 1, 3), dtype=torch.float64)
    r = torch.eye(3, dtype=torch.float64)[None, None]
    actual = mesh_distance(points, [meshes["box"]], t, r)
    expected = ref_mesh_distance(points.detach().numpy(), [refs["box"]], t.numpy(), r.numpy())
    np.testing.assert_allclose(actual.reduced_distance.detach(), expected.reduced_distance, atol=2e-12)
    np.testing.assert_allclose(actual.reduced_gradient.detach(), expected.reduced_gradient, atol=2e-12)
    assert actual.winning_face[0, 1, 0].item() == 0
    actual.reduced_distance.sum().backward()
    np.testing.assert_allclose(points.grad, actual.reduced_gradient[0].detach(), atol=2e-12)

    inactive = mesh_distance(points.detach()[:1], [meshes["box"]], t, r,
                             env_mesh_active=torch.tensor([[False]]))
    assert inactive.winning_mesh.item() == -1 and torch.isinf(inactive.reduced_distance).all()
    with pytest.raises(ValueError, match="watertight"):
        mesh_distance(points.detach()[:1], [meshes["open_triangle"]], t, r)


def test_mesh_unsigned_batched_environment_transform():
    meshes, refs = _meshes()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    points = torch.tensor([[[.25, .25, 1.]], [[1.75, -.75, 1.]]], dtype=torch.float64)
    translations = torch.tensor([[[0., 0., 0.]], [[2., -1., 0.]]], dtype=torch.float64)
    rotations = torch.stack((torch.eye(3, dtype=torch.float64), rotation))[:, None]
    result = mesh_distance(points, [meshes["open_triangle"]], translations, rotations,
                           env_indices=torch.tensor([0, 1]), signed=False)
    torch.testing.assert_close(result.reduced_distance, torch.ones((2, 1), dtype=torch.float64))
    torch.testing.assert_close(result.reduced_gradient[..., 2], torch.ones((2, 1), dtype=torch.float64))


def test_voxel_oracle_parity_boundary_oob_batches_and_autograd():
    grid, ref = _grid()
    points = torch.tensor([[.1, -.2, .3], [-.75, -.75, -.75], [.75, .75, .75], [.751, 0., 0.]],
                          dtype=torch.float64, requires_grad=True)
    actual = sample_voxel_sdf(points, [[grid]])
    expected = ref_sample_voxel_sdf(points.detach().numpy(), [[ref]])
    np.testing.assert_allclose(actual.values.detach(), expected.values, atol=2e-12)
    np.testing.assert_allclose(actual.gradients.detach(), expected.gradients, atol=2e-12)
    np.testing.assert_array_equal(actual.valid.cpu(), expected.valid)
    actual.values[0, :3].sum().backward()
    np.testing.assert_allclose(points.grad[:3], actual.gradients[0, :3, 0].detach(), atol=2e-12)
    torch.testing.assert_close(points.grad[3], torch.zeros(3, dtype=torch.float64))


def test_esdf_reduction_padding_active_and_tie():
    g0, r0 = _grid()
    g1, r1 = _grid(oob=50)
    points = torch.tensor([[[.1, -.2, .3]], [[.2, -.1, .1]]], dtype=torch.float64)
    envs = [[g0, g1], [g0, g1]]
    refs = [[r0, r1], [r0, r1]]
    env_index = torch.tensor([0, 1])
    actual = query_esdf(points, envs, env_indices=env_index, padding=.1)
    expected = ref_query_esdf(points.numpy(), refs, env_indices=[0, 1], padding=.1)
    np.testing.assert_allclose(actual.distance, expected.distance, atol=2e-12)
    np.testing.assert_array_equal(actual.winning_grid, expected.winning_grid)
    assert torch.equal(actual.winning_grid, torch.zeros((2, 1), dtype=torch.int64))
    active = torch.tensor([[False, True], [True, True]])
    assert query_esdf(points, envs, env_indices=env_index, grid_active=active).winning_grid[0].item() == 1
    oob = query_esdf(torch.tensor([[10., 0., 0.]], dtype=torch.float64), envs)
    assert not oob.valid.item() and oob.winning_grid.item() == -1


def test_sphere_activation_and_autograd():
    spheres = torch.tensor([[0., 0., 0., .2], [0., 0., 0., .2], [0., 0., 0., .2]], dtype=torch.float64)
    sdf = torch.tensor([.35, .25, .1], dtype=torch.float64, requires_grad=True)
    grad = torch.tensor([[1., 0., 0.]] * 3, dtype=torch.float64)
    cost, gradient = sphere_world_collision(spheres, sdf, grad, activation_distance=.1, padding=.05, weight=2)
    torch.testing.assert_close(cost, torch.tensor([0., .1, .4], dtype=torch.float64))
    torch.testing.assert_close(gradient, torch.tensor([[0., 0., 0.], [-2., 0., 0.], [-2., 0., 0.]], dtype=torch.float64))
    cost.sum().backward()
    torch.testing.assert_close(sdf.grad, gradient[:, 0])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_float32_fallback_disabled_smoke():
    grid, _ = _grid(torch.float32, "mps")
    points = torch.tensor([[.1, -.2, .3]], device="mps")
    sampled = sample_voxel_sdf(points, [[grid]])
    torch.testing.assert_close(sampled.values.cpu(), torch.tensor([[[-.45]]]), rtol=3e-5, atol=3e-6)
    sampled.values.sum().backward() if points.requires_grad else None
