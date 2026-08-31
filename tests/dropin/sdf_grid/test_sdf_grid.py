import pytest
import torch

from curobo._src.geom.sdf_grid import SDFGrid, compute_sdf_gradient, lookup_distance


def _linear_grid() -> tuple[torch.Tensor, torch.Tensor]:
    shape = torch.tensor([3, 3, 3], dtype=torch.int64)
    x, y, z = torch.meshgrid(
        torch.arange(3), torch.arange(3), torch.arange(3), indexing="ij"
    )
    return (x + 2 * y + 3 * z).float().reshape(-1), shape


def test_integer_grid_lookup_and_finite_difference_gradient_match_v2() -> None:
    distances, shape = _linear_grid()
    point = torch.tensor([[1, 1, 1]], dtype=torch.int64)
    distance = lookup_distance(point, distances, shape)
    assert distance.tolist() == [6.0]
    gradient = compute_sdf_gradient(point, distances, shape, distance)
    torch.testing.assert_close(gradient, torch.tensor([[1.0, 2.0, 3.0]]))


def test_sdf_grid_custom_backward_uses_pinned_integer_gradient() -> None:
    distances, shape = _linear_grid()
    point = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    value = SDFGrid.apply(point, distances, shape)
    assert value.shape == (1, 1)
    value.sum().backward()
    torch.testing.assert_close(point.grad, torch.tensor([[1.0, 2.0, 3.0]]))


def test_sdf_grid_rejects_distance_matrix_gradients_like_v2() -> None:
    distances, shape = _linear_grid()
    distances.requires_grad_(True)
    point = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
    with pytest.raises(NotImplementedError, match="dist_matrix"):
        SDFGrid.apply(point, distances, shape).sum().backward()

