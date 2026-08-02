"""Portable GraphPlannerResult lifecycle coverage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.graph_planner.result import GraphPlannerResult
from curobo._src.types.device_cfg import DeviceCfg


def _result(device: str = "cpu") -> GraphPlannerResult:
    values = torch.arange(18, dtype=torch.float32, device=device).reshape(3, 2, 3)
    return GraphPlannerResult(
        success=torch.tensor([True, False, True], dtype=torch.bool, device=device),
        plan_waypoints=[values[0], None, values[2, :1]],
        interpolated_waypoints=torch.arange(36, dtype=torch.float32, device=device).reshape(3, 4, 3),
        joint_names=["a", "b", "c"],
        path_length=torch.tensor([1.0, float("inf"), 0.0], device=device),
        solve_time=0.25,
        valid_query=False,
        debug_info={
            "per_problem": torch.tensor([2.0, 3.0, 5.0], device=device),
            "labels": ["first", "second", "third"],
        },
    )


def test_result_lifecycle_pads_selects_and_preserves_metadata() -> None:
    result = _result()
    assert result.batch_size == 3
    assert result.device.type == "cpu"
    assert result.num_success == 2
    assert result.success_ratio == pytest.approx(2 / 3)
    assert result.any_success() and not result.all_success()
    assert [path.shape[0] for path in result.successful_paths()] == [2, 1]

    padded, valid = result.as_padded_waypoints(-1)
    assert padded.shape == (3, 2, 3)
    assert valid.tolist() == [[True, True], [False, False], [True, False]]
    assert padded[1].eq(-1).all()
    torch.testing.assert_close(padded[0], result.plan_waypoints[0])

    selected = result[2]
    assert selected.success.tolist() == [True]
    assert selected.plan_waypoints is not None and selected.plan_waypoints[0].shape == (1, 3)
    assert selected.interpolated_waypoints.shape == (1, 4, 3)
    assert selected.path_length.tolist() == [0.0]
    assert selected.debug_info["per_problem"].item() == 5.0
    assert selected.debug_info["labels"] == ["third"]

    clone = result.clone()
    clone.plan_waypoints[0][0, 0] = -1
    clone.debug_info["per_problem"][0] = -1
    assert result.plan_waypoints[0][0, 0].item() == 0.0
    assert result.debug_info["per_problem"][0].item() == 2.0


def test_result_detach_to_and_validation() -> None:
    result = _result()
    result.plan_waypoints[0] = result.plan_waypoints[0].requires_grad_()
    detached = result.detach()
    assert not detached.plan_waypoints[0].requires_grad
    moved = result.to(DeviceCfg(torch.device("cpu"), torch.float64))
    assert moved.success.dtype is torch.bool
    assert moved.path_length.dtype is torch.float64
    assert moved.plan_waypoints[0].dtype is torch.float64

    with pytest.raises(ValueError, match="one-dimensional bool"):
        GraphPlannerResult(torch.ones(1))
    with pytest.raises(ValueError, match="one entry per batch"):
        GraphPlannerResult(torch.ones(2, dtype=torch.bool), plan_waypoints=[None])
    with pytest.raises(ValueError, match="same batch size"):
        GraphPlannerResult(torch.ones(2, dtype=torch.bool), interpolated_waypoints=torch.zeros(1, 2, 3))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_result_mps_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    result = _result("mps")
    padded, valid = result.as_padded_waypoints()
    assert padded.device.type == "mps" and valid.device.type == "mps"
    assert result[0].interpolated_waypoints.device.type == "mps"
    moved = result.to(DeviceCfg(torch.device("mps"), torch.float32))
    assert moved.path_length.device.type == "mps"
