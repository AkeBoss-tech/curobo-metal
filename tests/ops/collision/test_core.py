from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from curobo_metal.ops.collision import (
    sphere_cuboid_signed_distance,
    sphere_sphere_signed_distance,
    transform_spheres,
)
from curobo_metal.reference.collision import load_collision_case

FIXTURES = Path(__file__).parents[2] / "fixtures" / "collision"


def devices() -> list[str]:
    result = ["cpu"]
    if torch.backends.mps.is_available():
        result.append("mps")
    return result


def tensor(value, device: str, *, dtype=torch.float32):
    return torch.tensor(value, device=device, dtype=dtype)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("fixture_name", ["minimal.json", "realistic.json"])
def test_float32_matches_independent_golden(device: str, fixture_name: str) -> None:
    replay = load_collision_case(FIXTURES / fixture_name)
    for case in replay["cases"]:
        inputs, expected = case["inputs"], case["expected"]
        if case["operation"] == "transform_spheres":
            actual = transform_spheres(
                tensor(inputs["transforms"], device),
                tensor(inputs["local_spheres"], device),
                tensor(inputs["link_indices"], device, dtype=torch.int64),
            )
            observed = actual.spheres
            target = expected["spheres"]
        elif case["operation"] == "sphere_sphere":
            actual = sphere_sphere_signed_distance(
                tensor(inputs["spheres"], device),
                tensor(inputs["pairs"], device, dtype=torch.int64),
                sphere_active=(
                    tensor(inputs["sphere_active"], device, dtype=torch.bool)
                    if "sphere_active" in inputs
                    else None
                ),
                pair_active=(
                    tensor(inputs["pair_active"], device, dtype=torch.bool)
                    if "pair_active" in inputs
                    else None
                ),
                padding=inputs.get("padding", 0.0),
            )
            torch.testing.assert_close(
                actual.winning_pair.cpu(),
                torch.tensor(expected["winner"], dtype=torch.int64),
                rtol=0,
                atol=0,
            )
            observed = actual.distances[:, : np.asarray(expected["distance"]).shape[1]]
            target = expected["distance"]
            if "gradient" in expected:
                torch.testing.assert_close(
                    actual.gradients.cpu(),
                    torch.tensor(expected["gradient"], dtype=torch.float32),
                    rtol=2e-5,
                    atol=2e-6,
                )
        else:
            actual = sphere_cuboid_signed_distance(
                tensor(inputs["spheres"], device),
                tensor(inputs["centers"], device),
                tensor(inputs["rotations"], device),
                tensor(inputs["half_extents"], device),
                cuboid_active=(
                    tensor(inputs["cuboid_active"], device, dtype=torch.bool)
                    if "cuboid_active" in inputs
                    else None
                ),
                padding=inputs.get("padding", 0.0),
            )
            torch.testing.assert_close(
                actual.winning_cuboid.cpu(),
                torch.tensor(expected["winner"], dtype=torch.int64),
                rtol=0,
                atol=0,
            )
            observed = actual.reduced_distance
            target = expected["distance"]
            if "gradient" in expected:
                torch.testing.assert_close(
                    actual.reduced_sphere_gradient.cpu(),
                    torch.tensor(expected["gradient"], dtype=torch.float32),
                    rtol=2e-5,
                    atol=2e-6,
                )
        torch.testing.assert_close(
            observed.cpu(),
            torch.tensor(target, dtype=torch.float32),
            rtol=2e-5,
            atol=2e-6,
        )
        assert observed.device.type == device


def test_cpu_float64_golden_and_transform_autograd() -> None:
    case = load_collision_case(FIXTURES / "minimal.json")["cases"][2]
    inputs = case["inputs"]
    transforms = tensor(
        inputs["transforms"], "cpu", dtype=torch.float64
    ).requires_grad_()
    result = transform_spheres(
        transforms,
        tensor(inputs["local_spheres"], "cpu", dtype=torch.float64),
        tensor(inputs["link_indices"], "cpu", dtype=torch.int64),
    )
    torch.testing.assert_close(
        result.spheres,
        tensor(case["expected"]["spheres"], "cpu", dtype=torch.float64),
        rtol=2e-12,
        atol=2e-12,
    )
    weights = torch.arange(
        result.spheres[..., :3].numel(), dtype=torch.float64
    ).reshape_as(result.spheres[..., :3])
    (result.spheres[..., :3] * weights).sum().backward()
    expected = torch.zeros_like(transforms)
    for sphere, link in enumerate(inputs["link_indices"]):
        expected[link, :3, :] += (
            weights[0, sphere, :, None]
            * torch.tensor(
                inputs["local_spheres"][sphere][:3] + [1.0], dtype=torch.float64
            )
        )
    torch.testing.assert_close(transforms.grad, expected)


@pytest.mark.parametrize("device", devices())
def test_pair_autograd_finite_difference_and_coincident_subgradient(device: str) -> None:
    spheres = tensor(
        [[0.2, -0.1, 0.4, 0.15], [1.1, 0.3, -0.2, 0.25]], device
    ).requires_grad_()
    pairs = tensor([[0, 1]], device, dtype=torch.int64)
    result = sphere_sphere_signed_distance(spheres, pairs)
    result.reduced_distance.sum().backward()
    analytic = spheres.grad.detach().cpu().double()
    source = spheres.detach().cpu().double()
    numeric = torch.empty_like(source)
    step = 1e-4 if device == "mps" else 1e-6
    for sphere in range(2):
        for component in range(4):
            plus, minus = source.clone(), source.clone()
            plus[sphere, component] += step
            minus[sphere, component] -= step
            numeric[sphere, component] = (
                sphere_sphere_signed_distance(plus, pairs.cpu()).reduced_distance[0]
                - sphere_sphere_signed_distance(minus, pairs.cpu()).reduced_distance[0]
            ) / (2 * step)
    torch.testing.assert_close(analytic, numeric, rtol=3e-4, atol=3e-5)

    coincident = tensor(
        [[0, 0, 0, 0.2], [0, 0, 0, 0.3]], device
    ).requires_grad_()
    collision = sphere_sphere_signed_distance(coincident, pairs)
    collision.reduced_distance.sum().backward()
    torch.testing.assert_close(
        coincident.grad.cpu(),
        torch.tensor([[1, 0, 0, -1], [-1, 0, 0, -1]], dtype=torch.float32),
    )


@pytest.mark.parametrize("device", devices())
def test_cuboid_autograd_finite_difference_and_boundary_rules(device: str) -> None:
    spheres = tensor([[1.2, 0.8, 0.6, 0.17]], device).requires_grad_()
    centers = tensor([[0.1, -0.2, 0.3]], device)
    rotations = tensor(
        [[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], device
    )
    half = tensor([[0.5, 0.7, 0.2]], device)
    result = sphere_cuboid_signed_distance(spheres, centers, rotations, half)
    result.reduced_distance.sum().backward()
    analytic = spheres.grad.detach().cpu().double()[0]
    source = spheres.detach().cpu().double()
    numeric = torch.empty(4, dtype=torch.float64)
    step = 1e-4 if device == "mps" else 1e-6
    for component in range(4):
        plus, minus = source.clone(), source.clone()
        plus[0, component] += step
        minus[0, component] -= step
        args = (
            centers.cpu().double(),
            rotations.cpu().double(),
            half.cpu().double(),
        )
        numeric[component] = (
            sphere_cuboid_signed_distance(plus, *args).reduced_distance[0, 0]
            - sphere_cuboid_signed_distance(minus, *args).reduced_distance[0, 0]
        ) / (2 * step)
    torch.testing.assert_close(analytic, numeric, rtol=3e-4, atol=3e-5)

    boundary = tensor([[0, 0, 0, 0], [1, 1, 0, 0]], device).requires_grad_()
    box = sphere_cuboid_signed_distance(
        boundary,
        tensor([[0, 0, 0]], device),
        torch.eye(3, device=device).unsqueeze(0),
        tensor([[1, 1, 1]], device),
    )
    box.reduced_distance.sum().backward()
    torch.testing.assert_close(
        boundary.grad.cpu(),
        torch.tensor([[1, 0, 0, -1], [1, 0, 0, -1]], dtype=torch.float32),
    )


@pytest.mark.parametrize("device", devices())
def test_masks_ties_empty_noncontiguous_and_padding(device: str) -> None:
    backing = torch.zeros((1, 3, 8), dtype=torch.float32, device=device)
    backing[..., ::2] = tensor(
        [[[0, 0, 0, 0.2], [1, 0, 0, 0.2], [-1, 0, 0, 0.2]]], device
    )
    spheres = backing[..., ::2]
    assert not spheres.is_contiguous()
    pairs = tensor([[0, 1], [0, 2]], device, dtype=torch.int64)
    pair = sphere_sphere_signed_distance(spheres, pairs, padding=0.1)
    assert pair.winning_pair.item() == 0
    torch.testing.assert_close(pair.reduced_distance.cpu(), torch.tensor([0.5]))

    inactive = sphere_sphere_signed_distance(
        spheres.requires_grad_(),
        pairs,
        pair_active=tensor([False, False], device, dtype=torch.bool),
    )
    assert torch.isinf(inactive.distances).all()
    assert inactive.winning_pair.item() == -1
    assert not inactive.reduced_gradient.any()
    inactive.reduced_distance.sum().backward()
    assert not spheres.grad.any()

    empty_pairs = sphere_sphere_signed_distance(
        torch.empty((0, 3, 4), device=device),
        torch.empty((0, 2), dtype=torch.int64, device=device),
    )
    assert empty_pairs.distances.shape == (0, 0)
    assert empty_pairs.reduced_distance.shape == (0,)

    empty_boxes = sphere_cuboid_signed_distance(
        spheres.detach().requires_grad_(),
        torch.empty((0, 3), device=device),
        torch.empty((0, 3, 3), device=device),
        torch.empty((0, 3), device=device),
    )
    assert empty_boxes.distances.shape == (1, 3, 0)
    assert torch.isinf(empty_boxes.reduced_distance).all()
    assert (empty_boxes.winning_cuboid == -1).all()
    empty_boxes.reduced_distance.sum().backward()


def test_validation_and_fallback_policy() -> None:
    spheres = torch.zeros((2, 4))
    pairs = torch.tensor([[0, 1]], dtype=torch.int64)
    invalid = [
        lambda: sphere_sphere_signed_distance(spheres.long(), pairs),
        lambda: sphere_sphere_signed_distance(spheres, pairs.float()),
        lambda: sphere_sphere_signed_distance(spheres, torch.tensor([[0, 0]])),
        lambda: sphere_sphere_signed_distance(spheres, pairs, padding=-1),
        lambda: sphere_cuboid_signed_distance(
            spheres, torch.zeros((1, 3)), torch.zeros((1, 3, 3)), torch.ones((1, 3))
        ),
    ]
    for call in invalid:
        with pytest.raises((TypeError, ValueError)):
            call()
    if torch.backends.mps.is_available():
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
        with pytest.raises(ValueError, match="expected a tensor"):
            sphere_sphere_signed_distance(spheres.to("mps"), pairs)
