from __future__ import annotations

import pytest
import torch

from curobo_metal.ops.collision import (
    sphere_cuboid_signed_distance,
    sphere_sphere_signed_distance,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple MPS"
)


def _mps(value, *, dtype=torch.float32):
    return torch.tensor(value, dtype=dtype, device="mps")


def test_pair_fused_masks_first_tie_coincident_and_empty() -> None:
    spheres = _mps(
        [[[0, 0, 0, 0.2], [0, 0, 0, 0.3], [1, 0, 0, 0.1]]]
    ).requires_grad_()
    pairs = _mps([[0, 1], [0, 2], [2, 0]], dtype=torch.int64)
    result = sphere_sphere_signed_distance(
        spheres,
        pairs,
        pair_active=_mps([True, False, False], dtype=torch.bool),
    )
    assert result.winning_pair.item() == 0
    assert torch.isinf(result.distances[0, 1:]).all()
    torch.testing.assert_close(
        result.reduced_gradient.cpu(),
        torch.tensor(
            [[[1, 0, 0, -1], [-1, 0, 0, -1], [0, 0, 0, 0]]],
            dtype=torch.float32,
        ),
    )
    result.reduced_distance.sum().backward()
    torch.testing.assert_close(spheres.grad.cpu(), result.reduced_gradient.cpu())

    empty = sphere_sphere_signed_distance(
        _mps([]).reshape(0, 3, 4),
        _mps([], dtype=torch.int64).reshape(0, 2),
    )
    assert empty.distances.shape == (0, 0)
    assert empty.reduced_distance.shape == (0,)


def test_pair_fused_all_distance_and_reduced_autograd() -> None:
    spheres = _mps(
        [[[0.2, -0.1, 0.4, 0.15], [1.1, 0.3, -0.2, 0.25], [-0.4, 0.8, 0.5, 0.1]]]
    ).requires_grad_()
    pairs = _mps([[0, 1], [0, 2]], dtype=torch.int64)
    result = sphere_sphere_signed_distance(spheres, pairs)
    loss = (result.distances * _mps([[0.25, -0.5]])).sum() + 0.7 * result.reduced_distance.sum()
    loss.backward()
    expected = (
        0.25 * result.gradients[:, 0]
        - 0.5 * result.gradients[:, 1]
        + 0.7 * result.reduced_gradient
    )
    torch.testing.assert_close(spheres.grad.cpu(), expected.cpu(), rtol=2e-5, atol=2e-6)


def test_cuboid_fused_rotated_masks_boundaries_and_first_tie() -> None:
    spheres = _mps([[[0, 0, 0, 0], [1, 1, 0, 0], [3, 0, 0, 0.1]]]).requires_grad_()
    centers = _mps([[0, 0, 0], [0, 0, 0]])
    rotations = torch.eye(3, device="mps").expand(2, 3, 3).clone()
    half = _mps([[1, 1, 1], [1, 1, 1]])
    result = sphere_cuboid_signed_distance(
        spheres,
        centers,
        rotations,
        half,
        sphere_active=_mps([True, True, False], dtype=torch.bool),
    )
    torch.testing.assert_close(
        result.winning_cuboid.cpu(), torch.tensor([[0, 0, -1]])
    )
    torch.testing.assert_close(
        result.reduced_sphere_gradient.cpu(),
        torch.tensor(
            [[[1, 0, 0, -1], [1, 0, 0, -1], [0, 0, 0, 0]]],
            dtype=torch.float32,
        ),
    )
    result.reduced_distance[:, :2].sum().backward()
    torch.testing.assert_close(spheres.grad.cpu(), result.reduced_sphere_gradient.cpu())

    # Validation may be cached only until PyTorch observes an in-place mutation.
    half[0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        sphere_cuboid_signed_distance(spheres.detach(), centers, rotations, half)


@pytest.mark.parametrize("operation", ["pair", "cuboid"])
def test_fused_finite_difference(operation: str) -> None:
    source = torch.tensor([[0.37, -0.21, 0.73, 0.16]], dtype=torch.float64)
    spheres = source.float().to("mps").requires_grad_()
    if operation == "pair":
        other = _mps([[1.2, 0.4, -0.1, 0.22]])
        values = torch.cat((spheres, other), dim=0)
        result = sphere_sphere_signed_distance(
            values, _mps([[0, 1]], dtype=torch.int64)
        )
    else:
        result = sphere_cuboid_signed_distance(
            spheres,
            _mps([[0.1, -0.2, 0.3]]),
            _mps([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]]),
            _mps([[0.5, 0.7, 0.2]]),
        )
    result.reduced_distance.sum().backward()
    analytic = spheres.grad.detach().cpu().double()[0]

    step = 1e-4
    numeric = torch.empty(4, dtype=torch.float64)
    for component in range(4):
        observed = []
        for direction in (1, -1):
            point = source.clone()
            point[0, component] += direction * step
            if operation == "pair":
                value = sphere_sphere_signed_distance(
                    torch.cat((point.float().to("mps"), other), dim=0),
                    _mps([[0, 1]], dtype=torch.int64),
                ).reduced_distance
            else:
                value = sphere_cuboid_signed_distance(
                    point.float().to("mps"),
                    _mps([[0.1, -0.2, 0.3]]),
                    _mps([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]]),
                    _mps([[0.5, 0.7, 0.2]]),
                ).reduced_distance
            observed.append(value.item())
        numeric[component] = (observed[0] - observed[1]) / (2 * step)
    torch.testing.assert_close(analytic, numeric, rtol=5e-4, atol=5e-4)
