"""Additional rollout-metrics aggregation and result-lifecycle coverage."""

from __future__ import annotations

import pytest
import torch

from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.state.state_joint import JointState


def test_batch_scalar_terms_keep_the_batch_axis_and_autograd() -> None:
    scalar = torch.tensor([1.5, -2.0, 3.5], requires_grad=True)
    collection = CostCollection([scalar], ["terminal"])

    per_horizon = collection.get_sum(sum_horizon=False)
    total = collection.get_sum(sum_horizon=True)
    assert per_horizon.shape == total.shape == (3, 1)
    torch.testing.assert_close(total[:, 0], scalar)
    total.sum().backward()
    torch.testing.assert_close(scalar.grad, torch.ones_like(scalar))


def test_aggregation_rejects_ambiguous_batch_seed_or_time_layouts() -> None:
    collection = CostCollection(
        [torch.ones(2, 3, 1), torch.ones(2, 4, 1)], ["short", "long"]
    )
    with pytest.raises(ValueError, match="horizon"):
        collection.get_sum()

    # A nested seed layout cannot silently be combined with a different batch
    # prefix, which previously produced opaque torch.cat failures.
    collection = CostCollection(
        [torch.ones(2, 3, 4, 1), torch.ones(2, 4, 1)], ["seeded", "plain"]
    )
    with pytest.raises(ValueError, match="batch/seed"):
        collection.get_sum()


def _metrics(device: str = "cpu") -> RolloutMetrics:
    actions = torch.arange(48.0, device=device).reshape(2, 3, 4, 2)
    trace = torch.arange(24.0, device=device).reshape(2, 3, 4, 1)
    return RolloutMetrics(
        actions=actions,
        costs_and_constraints=CostsAndConstraints(
            costs=CostCollection([trace], ["trace"]),
        ),
        state=JointState.from_position(actions.clone()),
        debug={"trace": trace, "nested": {"mask": trace > 3}, "label": "retained"},
        feasible=(trace >= 0),
        convergence=CostCollection([trace + 1], ["convergence"]),
    )


def test_selection_indexes_nested_debug_tensors_but_preserves_metadata() -> None:
    metrics = _metrics()
    indexed = metrics[1]
    assert indexed.debug["trace"].shape == (3, 4, 1)
    assert indexed.debug["nested"]["mask"].shape == (3, 4, 1)
    assert indexed.debug["label"] == "retained"

    selected = metrics.get_only_batch_seed_indices(torch.tensor([0, 1]), torch.tensor([2, 0]))
    assert selected.debug["trace"].shape == (2, 4, 1)
    torch.testing.assert_close(selected.debug["trace"], metrics.debug["trace"][[0, 1], [2, 0]])
    assert selected.debug["nested"]["mask"].dtype is torch.bool
    # Selection is a value view like regular tensor indexing; it does not
    # modify the original debug payload or unrelated rollout channels.
    assert metrics.debug["trace"].shape == (2, 3, 4, 1)
    assert selected.actions.shape == selected.state.position.shape == (2, 4, 2)

    result = RolloutResult(actions=metrics.actions, debug=metrics.debug)
    assert result[0].debug["trace"].shape == (3, 4, 1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_scalar_aggregation_and_nested_selection_stay_on_mps_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    scalar = torch.tensor([1.0, 2.0], device="mps", requires_grad=True)
    total = CostCollection([scalar], ["terminal"]).get_sum()
    assert total.device.type == "mps"
    total.sum().backward()
    assert scalar.grad is not None and scalar.grad.device.type == "mps"

    selected = _metrics("mps").get_only_batch_seed_indices(
        torch.tensor([0], device="mps"), torch.tensor([1], device="mps")
    )
    assert selected.debug["trace"].device.type == "mps"
    assert selected.debug["nested"]["mask"].device.type == "mps"
