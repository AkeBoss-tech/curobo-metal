from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from curobo_metal.backend import resolve_device, validate_tensor_device
from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics
from curobo_metal.reference import SerialRobot, load_case

FIXTURES = Path(__file__).parents[2] / "fixtures"


def fixture(name: str = "panda_serial.json"):
    case = load_case(FIXTURES / name)
    return case, SerialRobot.from_dict(case["robot"])


def devices() -> list[str]:
    values = ["cpu"]
    if torch.backends.mps.is_available():
        values.append("mps")
    return values


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("name", ["two_link_planar.json", "panda_serial.json"])
def test_float32_matches_independent_golden(device: str, name: str) -> None:
    case, robot = fixture(name)
    q = torch.tensor(case["inputs"]["q"], dtype=torch.float32, device=device)
    chain = KinematicChain(robot, device=device, dtype=q.dtype)
    actual = forward_kinematics(chain, q)
    expected = case["expected"]
    assert actual.transforms.device.type == device
    assert actual.transform_jacobian.device.type == device
    assert actual.geometric_jacobian.device.type == device
    np.testing.assert_allclose(
        actual.transforms.detach().cpu(), expected["transforms"], rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        actual.transform_jacobian.detach().cpu(),
        expected["transform_jacobian"],
        rtol=8e-5,
        atol=8e-5,
    )
    np.testing.assert_allclose(
        actual.geometric_jacobian.detach().cpu(),
        expected["geometric_jacobian"],
        rtol=8e-5,
        atol=8e-5,
    )


def test_cpu_float64_matches_independent_golden() -> None:
    case, robot = fixture()
    q = torch.tensor(case["inputs"]["q"], dtype=torch.float64)
    actual = forward_kinematics(
        KinematicChain(robot, dtype=torch.float64), q
    )
    np.testing.assert_allclose(
        actual.transforms.detach(), case["expected"]["transforms"], rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        actual.transform_jacobian.detach(),
        case["expected"]["transform_jacobian"],
        rtol=5e-11,
        atol=5e-11,
    )
    np.testing.assert_allclose(
        actual.geometric_jacobian.detach(),
        case["expected"]["geometric_jacobian"],
        rtol=5e-11,
        atol=5e-11,
    )


@pytest.mark.parametrize("device", devices())
def test_autograd_vjp_matches_independent_jacobian(device: str) -> None:
    case, robot = fixture()
    q = torch.tensor(
        case["inputs"]["q"][1], dtype=torch.float32, device=device, requires_grad=True
    )
    result = forward_kinematics(
        KinematicChain(robot, device=device, dtype=q.dtype), q
    )
    weights = torch.arange(
        result.transforms.numel(),
        dtype=q.dtype,
        device=device,
    ).reshape_as(result.transforms) / result.transforms.numel()
    (result.transforms * weights).sum().backward()
    expected_jacobian = torch.tensor(
        case["expected"]["transform_jacobian"][1],
        dtype=q.dtype,
        device=device,
    )
    expected = torch.einsum("blrc,lrcj->j", weights, expected_jacobian)
    torch.testing.assert_close(q.grad, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("device", devices())
def test_non_contiguous_unbatched_and_empty(device: str) -> None:
    case, robot = fixture()
    chain = KinematicChain(robot, device=device, dtype=torch.float32)
    backing = torch.zeros((robot.dof, 2), dtype=torch.float32, device=device)
    backing[:, 0] = torch.tensor(
        case["inputs"]["q"][1], dtype=torch.float32, device=device
    )
    q = backing[:, 0]
    assert not q.is_contiguous()
    result = forward_kinematics(chain, q)
    assert result.input_was_batched is False
    assert result.transforms.shape == (1, 8, 4, 4)
    assert not result.transforms.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()

    empty = forward_kinematics(
        chain, torch.empty((0, robot.dof), dtype=torch.float32, device=device)
    )
    assert empty.transforms.shape == (0, 8, 4, 4)
    assert empty.transform_jacobian.shape == (0, 8, 4, 4, 7)
    assert empty.geometric_jacobian.shape == (0, 8, 6, 7)


def test_prismatic_chain() -> None:
    robot = SerialRobot.from_dict(
        {
            "name": "slider",
            "joints": [
                {
                    "name": "slide",
                    "type": "prismatic",
                    "axis": [2.0, 0.0, 0.0],
                    "origin": {"xyz": [0.0, 1.0, 0.0]},
                }
            ],
        }
    )
    q = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    actual = forward_kinematics(
        KinematicChain(robot, dtype=torch.float64), q
    )
    torch.testing.assert_close(
        actual.transforms[0, 0, :3, 3],
        torch.tensor([0.5, 1.0, 0.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        actual.geometric_jacobian[0, 0, :, 0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    )
    actual.transforms.sum().backward()
    torch.testing.assert_close(q.grad, torch.tensor([1.0], dtype=torch.float64))


@pytest.mark.parametrize(
    ("q", "error"),
    [
        (torch.zeros(6), ValueError),
        (torch.zeros(1, 8), ValueError),
        (torch.zeros(7, dtype=torch.int64), TypeError),
        (torch.full((7,), torch.nan), ValueError),
        (torch.full((7,), torch.inf), ValueError),
    ],
)
def test_invalid_inputs(q: torch.Tensor, error: type[Exception]) -> None:
    _, robot = fixture()
    chain = KinematicChain(robot)
    with pytest.raises(error):
        forward_kinematics(chain, q)


def test_explicit_device_validation() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="supports only"):
        resolve_device("meta")
    if torch.backends.mps.is_available():
        with pytest.raises(ValueError, match="expected"):
            validate_tensor_device(torch.zeros(1), expected="mps")
        _, robot = fixture()
        with pytest.raises(TypeError, match="only float32"):
            KinematicChain(robot, device="mps", dtype=torch.float64)
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0"
