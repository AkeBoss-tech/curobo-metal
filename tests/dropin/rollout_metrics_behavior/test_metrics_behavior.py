"""Behavioral coverage for portable rollout metric value objects."""

import pytest
import torch

from curobo._src.rollout.metrics import (
    CostCollection,
    CostCollectionSum,
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
)
from curobo._src.state.state_joint import JointState


def _metrics(device: str = "cpu") -> RolloutMetrics:
    # [batch, seed, horizon, component] verifies the seed-safe reduction path
    # used by particle and trajectory solvers.
    actions = torch.arange(48.0, device=device).reshape(2, 3, 4, 2)
    values = CostCollection()
    values.add(actions[..., :1].clone(), "value", weight=torch.ones(2, 3, 1, 1, device=device))
    constraints = CostCollection()
    constraints.add(-torch.ones(2, 3, 4, 1, device=device), "safe")
    hybrid = CostCollection()
    hybrid.add(torch.full((2, 3, 4, 1), 2.0, device=device), "hybrid")
    convergence = CostCollection()
    convergence.add(torch.ones(2, 3, 4, 1, device=device), "goal")
    return RolloutMetrics(
        actions=actions,
        costs_and_constraints=CostsAndConstraints(values, constraints, hybrid),
        state=JointState.from_position(actions.clone()),
        feasible=torch.ones(2, 3, 4, 1, device=device, dtype=torch.bool),
        convergence=convergence,
        debug={"portable": True},
    )


def test_collection_aggregates_component_and_horizon_for_batch_seed_layouts():
    metrics = _metrics()
    costs = metrics.costs_and_constraints
    assert costs.get_sum_cost(False).shape == (2, 3, 4)
    # Each horizon sample combines the first action component and hybrid=2.
    torch.testing.assert_close(costs.get_sum_cost(True), torch.tensor([[[20.], [52.], [84.]], [[116.], [148.], [180.]]]))
    assert costs.get_feasible(True).shape == (2, 3, 1)
    assert not bool(costs.get_feasible(True).all())
    assert bool(costs.get_feasible(True, include_all_hybrid=False).all())
    selected = costs.get_sum_cost(False, include_all_hybrid=False, include_from_hybrid=("hybrid",))
    torch.testing.assert_close(selected, costs.get_sum_cost(False))


def test_cost_metadata_stays_term_aligned_and_clone_is_independent():
    collection = CostCollection()
    collection.add(torch.ones(2, 3, 1), "unweighted")
    weight = torch.full((2, 3, 1), 3.0)
    collection.add(torch.ones(2, 3, 1), "weighted", weight=weight)
    clone = collection.clone()
    assert clone.weights[0] is None
    torch.testing.assert_close(clone.weights[1], weight)
    weights, squares = CostsAndConstraints(constraints=collection).get_constraint_weights()
    assert weights == [weight]
    torch.testing.assert_close(squares[0], torch.ones_like(weight))
    collection.values[0].zero_()
    assert bool(clone.values[0].all())


def test_result_and_metric_selection_preserve_all_rollout_channels():
    metrics = _metrics()
    indexed = metrics[1]
    assert indexed.actions.shape == (3, 4, 2)
    assert indexed.state.position.shape == (3, 4, 2)
    assert indexed.costs_and_constraints.costs.values[0].shape == (3, 4, 1)
    assert indexed.convergence.values[0].shape == (3, 4, 1)

    selected = metrics.get_only_batch_seed_indices(torch.tensor([0, 1]), torch.tensor([2, 0]))
    assert selected.actions.shape == (2, 4, 2)
    assert selected.state.position.shape == (2, 4, 2)
    assert selected.costs_and_constraints.constraints.values[0].shape == (2, 4, 1)
    assert selected.convergence.values[0].shape == (2, 4, 1)

    result = RolloutResult(actions=metrics.actions, costs_and_constraints=metrics.costs_and_constraints)
    assert result[0].costs_and_constraints.costs.values[0].shape == (3, 4, 1)


def test_batch_seed_copy_updates_actions_state_feasibility_and_collections():
    target = _metrics()
    source = _metrics()
    source.actions.fill_(-9.0)
    source.state.position.fill_(-7.0)
    source.feasible.fill_(False)
    source.costs_and_constraints.costs.values[0].fill_(99.0)
    source.convergence.values[0].fill_(17.0)
    batch = torch.tensor([0, 1])
    seed = torch.tensor([1, 2])
    target.copy_at_batch_seed_indices(source, batch, seed)
    assert target.actions[0, 1, 0, 0] == -9
    assert target.state.position[1, 2, 0, 0] == -7
    assert not target.feasible[0, 1, 0, 0]
    assert target.costs_and_constraints.costs.values[1 - 1][0, 1, 0, 0] == 99
    assert target.convergence.values[0][1, 2, 0, 0] == 17
    # A non-selected seed must retain the original source-independent value.
    assert target.actions[0, 0, 0, 0] != -9


def test_explicit_vjp_helper_preserves_provided_gradient_without_cuda_abi():
    value = torch.ones(2, 3, 1, requires_grad=True)
    vjp = torch.full_like(value, 2.5)
    CostCollectionSum.apply(value, vjp).sum().backward()
    torch.testing.assert_close(value.grad, vjp)
    with pytest.raises(ValueError, match="pairs"):
        CostCollectionSum.apply(value)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_metric_aggregation_and_vjp_remain_on_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    metrics = _metrics("mps")
    total = metrics.costs_and_constraints.get_sum_cost(True)
    assert total.device.type == "mps"
    selected = metrics.get_only_batch_seed_indices(torch.tensor([0], device="mps"), torch.tensor([1], device="mps"))
    assert selected.actions.device.type == selected.state.position.device.type == "mps"

    value = torch.ones(1, 2, 1, device="mps", requires_grad=True)
    CostCollectionSum.apply(value, torch.ones_like(value)).sum().backward()
    assert value.grad is not None and value.grad.device.type == "mps"
