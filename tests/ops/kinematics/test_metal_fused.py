from __future__ import annotations

from pathlib import Path

import pytest
import torch

from curobo_metal.ops.kinematics import KinematicChain, forward_kinematics
from curobo_metal.ops.kinematics.metal import fused_forward_kinematics
from curobo_metal.reference import SerialRobot, load_case

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
FIXTURE = Path(__file__).parents[2] / "fixtures" / "panda_serial.json"


def panda():
    case = load_case(FIXTURE)
    robot = SerialRobot.from_dict(case["robot"])
    return case, KinematicChain(robot, device="mps", dtype=torch.float32)


def test_fused_outputs_match_public_dispatch_and_golden() -> None:
    case, chain = panda()
    q = torch.tensor(case["inputs"]["q"], dtype=torch.float32, device="mps")
    direct = fused_forward_kinematics(chain, q)
    public = forward_kinematics(chain, q)
    for observed, dispatched, name, atol in zip(
        direct,
        (
            public.transforms,
            public.transform_jacobian,
            public.geometric_jacobian,
        ),
        ("transforms", "transform_jacobian", "geometric_jacobian"),
        (2e-5, 8e-5, 8e-5),
        strict=True,
    ):
        torch.testing.assert_close(observed, dispatched, rtol=0, atol=0)
        expected = torch.tensor(
            case["expected"][name], dtype=torch.float32, device="mps"
        )
        torch.testing.assert_close(observed, expected, rtol=atol, atol=atol)


def test_fused_vjp_and_finite_difference() -> None:
    case, chain = panda()
    q = torch.tensor(
        case["inputs"]["q"][1],
        dtype=torch.float32,
        device="mps",
        requires_grad=True,
    )
    transforms, jacobian, _ = fused_forward_kinematics(chain, q.unsqueeze(0))
    weights = torch.linspace(
        -0.5, 0.75, transforms.numel(), device="mps"
    ).reshape_as(transforms)
    loss = (transforms * weights).sum()
    (analytic,) = torch.autograd.grad(loss, q)
    expected = torch.einsum("blrc,blrcj->j", weights, jacobian)
    torch.testing.assert_close(analytic, expected, rtol=2e-4, atol=2e-4)

    step = 3e-3
    finite_difference = torch.empty_like(q)
    with torch.no_grad():
        for j in range(q.numel()):
            offset = torch.zeros_like(q)
            offset[j] = step
            plus = fused_forward_kinematics(chain, (q + offset).unsqueeze(0))[0]
            minus = fused_forward_kinematics(chain, (q - offset).unsqueeze(0))[0]
            finite_difference[j] = ((plus - minus) * weights).sum() / (2 * step)
    torch.testing.assert_close(
        analytic, finite_difference, rtol=3e-3, atol=3e-4
    )


def test_fused_empty_and_noncontiguous_input() -> None:
    case, chain = panda()
    empty = fused_forward_kinematics(
        chain, torch.empty((0, chain.dof), dtype=torch.float32, device="mps")
    )
    assert [value.shape for value in empty] == [
        (0, 8, 4, 4),
        (0, 8, 4, 4, 7),
        (0, 8, 6, 7),
    ]

    backing = torch.zeros((1, chain.dof, 2), dtype=torch.float32, device="mps")
    backing[0, :, 0] = torch.tensor(case["inputs"]["q"][0], device="mps")
    noncontiguous = backing[:, :, 0]
    assert not noncontiguous.is_contiguous()
    result = fused_forward_kinematics(chain, noncontiguous)
    assert all(value.device.type == "mps" for value in result)
