import pytest
import torch

from curobo_metal_op_spike import metal_square


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS-capable Mac"
)


def test_forward_runs_on_mps_and_matches_reference():
    x = torch.linspace(-3.0, 3.0, 257, device="mps")
    actual = metal_square(x)
    torch.mps.synchronize()

    assert x.device.type == "mps"
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), x.cpu().square())


def test_custom_backward_runs_on_mps_and_matches_derivative():
    x = torch.tensor([-2.5, -0.5, 0.0, 1.5, 4.0], device="mps", requires_grad=True)
    weights = torch.tensor([0.2, -1.0, 3.0, 2.0, -0.5], device="mps")

    metal_square(x).backward(weights)
    torch.mps.synchronize()

    assert x.grad is not None
    assert x.grad.device.type == "mps"
    torch.testing.assert_close(x.grad.cpu(), (2.0 * x.detach() * weights).cpu())


def test_backward_matches_central_finite_difference():
    x = torch.tensor([-1.25, -0.1, 0.75, 2.0], device="mps", requires_grad=True)
    metal_square(x).sum().backward()
    analytic = x.grad.detach().cpu()

    epsilon = 1e-3
    samples = x.detach().cpu()
    finite_difference = ((samples + epsilon).square() - (samples - epsilon).square())
    finite_difference /= 2.0 * epsilon
    torch.testing.assert_close(analytic, finite_difference, rtol=2e-3, atol=2e-3)


def test_empty_tensor():
    x = torch.empty(0, device="mps", requires_grad=True)
    metal_square(x).sum().backward()
    assert x.grad is not None
    assert x.grad.device.type == "mps"
    assert x.grad.numel() == 0


def test_cpu_input_is_rejected_instead_of_falling_back():
    with pytest.raises(ValueError, match="CPU fallback is forbidden"):
        metal_square(torch.ones(4))


def test_unsupported_layout_and_dtype_are_explicit():
    with pytest.raises(TypeError, match="float32"):
        metal_square(torch.ones(4, device="mps", dtype=torch.float16))
    with pytest.raises(ValueError, match="contiguous"):
        metal_square(torch.ones((3, 4), device="mps").transpose(0, 1))

