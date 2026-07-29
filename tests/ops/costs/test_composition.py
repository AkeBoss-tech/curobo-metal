import torch

from curobo_metal.ops.costs import CostManager, CostTerm, waypoint_cost


def test_waypoint_offset_run_weight_and_named_composition() -> None:
    q = torch.tensor([[[0.0, 0.0], [1.0, -1.0], [2.0, 2.0]]])
    target = torch.tensor([0.0, 0.0])
    manager = CostManager({
        "middle": CostTerm(
            lambda value: waypoint_cost(
                value, target, offset=1, run_weight=torch.tensor([2.0]),
            )
        ),
        "last": CostTerm(lambda value: waypoint_cost(value, target, offset=-1)),
    })
    total, components = manager.evaluate(q)
    torch.testing.assert_close(total, components["middle"] + components["last"])
    torch.testing.assert_close(components["middle"], torch.tensor([2.0]))
    torch.testing.assert_close(components["last"], torch.tensor([4.0]))
