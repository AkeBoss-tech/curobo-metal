"""Direct behavioural coverage for the portable PRM linear connector."""

from __future__ import annotations

import pytest
import torch

from curobo._src.graph_planner.graph.connector_linear import LinearConnector
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg


def _connector(device: str = "cpu", *, step: float = 0.25, capacity: int = 16):
    device_cfg = DeviceCfg(device=torch.device(device))
    config = PRMGraphPlannerCfg(
        cspace_similarity_threshold=step,
        steer_buffer_size=capacity,
        device_cfg=device_cfg,
    )
    connector = LinearConnector(config, device_cfg=config.device_cfg)
    connector.set_dependencies(
        2,
        torch.tensor([2.0, 1.0], **device_cfg.as_torch_dict()),
        lambda rows: rows[:, 0] <= 0.60,
        torch.arange(capacity, device=device_cfg.device, dtype=torch.int64),
    )
    return connector


def test_batched_weighted_sampling_first_collision_and_index_padding() -> None:
    connector = _connector()
    start = torch.tensor([[0.0, 0.0, 4.0], [-1.0, 0.0, 9.0]])
    goal = torch.tensor([[1.0, 0.0, 7.0], [0.0, 0.0, 8.0]])
    line = connector._compute_steering_line_points(start, goal)
    # Weight two on q0 yields nine endpoint-inclusive samples, independent of
    # each input graph index.
    assert line.shape == (2, 9, 2)
    torch.testing.assert_close(line[0, :, 0], torch.linspace(0.0, 1.0, 9))
    output = connector.steer_until_infeasible(start, goal)
    # The second segment remains feasible through its endpoint at zero.
    torch.testing.assert_close(output, torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))


def test_all_feasible_first_infeasible_empty_and_buffer_contract() -> None:
    connector = _connector(step=0.5, capacity=5)
    connector.set_dependencies(
        2, torch.ones(2), lambda rows: torch.ones(rows.shape[0], dtype=torch.bool), torch.arange(5)
    )
    start = torch.tensor([[0.0, 0.0, 2.0]])
    goal = torch.tensor([[1.0, 0.0, 3.0]])
    torch.testing.assert_close(connector.steer_until_infeasible(start, goal), torch.tensor([[1.0, 0.0, 0.0]]))
    connector.set_dependencies(
        2, torch.ones(2), lambda rows: torch.zeros(rows.shape[0], dtype=torch.bool), torch.arange(5)
    )
    torch.testing.assert_close(connector.steer_until_infeasible(start, goal), torch.tensor([[0.0, 0.0, 0.0]]))
    empty = torch.empty((0, 3))
    assert connector.steer_until_infeasible(empty, empty).shape == (0, 3)
    tiny = _connector(capacity=2)
    with pytest.raises(ValueError, match="steer_buffer_size"):
        tiny.steer_until_infeasible(start, goal)


def test_dependency_and_callback_contracts_are_explicit() -> None:
    config = PRMGraphPlannerCfg(cspace_similarity_threshold=0.1, steer_buffer_size=8, device_cfg=DeviceCfg("cpu"))
    connector = LinearConnector(config, device_cfg=config.device_cfg)
    with pytest.raises(RuntimeError, match="set_dependencies"):
        connector.steer_until_infeasible(torch.zeros(1, 3), torch.zeros(1, 3))
    with pytest.raises(ValueError, match="finite positive"):
        LinearConnector(PRMGraphPlannerCfg(cspace_similarity_threshold=0.0, steer_buffer_size=8, device_cfg=DeviceCfg("cpu")), device_cfg=DeviceCfg("cpu"))
    with pytest.raises(ValueError, match="nonnegative"):
        connector.set_dependencies(2, torch.tensor([-1.0, 1.0]), lambda x: x[:, 0].bool(), torch.arange(8))
    connector.set_dependencies(2, torch.ones(2), lambda rows: torch.ones(rows.shape[0], dtype=torch.int64), torch.arange(8))
    with pytest.raises(ValueError, match="bool shape"):
        connector.steer_until_infeasible(torch.zeros(1, 3), torch.full((1, 3), 0.5))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_connector_stays_on_mps_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    connector = _connector("mps")
    start = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 2.0]], device="mps")
    goal = torch.tensor([[1.0, 0.0, 3.0], [0.0, 0.0, 4.0]], device="mps")
    output = connector.steer_until_infeasible(start, goal)
    assert output.device.type == "mps"
    torch.testing.assert_close(output.cpu(), torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))
