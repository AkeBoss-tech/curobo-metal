"""Adversarial indexed-copy coverage for shared JointState metadata."""

import torch

from curobo._src.state.state_joint import JointState


def _seeded_state(value: float) -> JointState:
    position = torch.full((2, 3, 4, 1), value)
    return JointState(
        position=position,
        velocity=position + 1,
        acceleration=position + 2,
        jerk=position + 3,
        joint_names=["joint"],
        dt=torch.full((2, 3), value + 4),
        knot=torch.full((5, 1), value + 5),
        knot_dt=torch.tensor(value + 6),
    )


def test_batch_seed_copy_skips_global_knot_schedule_and_scalar_timing() -> None:
    target = _seeded_state(0.0)
    source = _seeded_state(10.0)
    original_knot = target.knot.clone()
    original_knot_dt = target.knot_dt.clone()

    target.copy_at_batch_seed_indices(
        source, torch.tensor([0, 1]), torch.tensor([2, 0])
    )

    torch.testing.assert_close(target.position[0, 2], source.position[0, 2])
    torch.testing.assert_close(target.position[1, 0], source.position[1, 0])
    torch.testing.assert_close(target.dt[0, 2], source.dt[0, 2])
    torch.testing.assert_close(target.dt[1, 0], source.dt[1, 0])
    torch.testing.assert_close(target.knot, original_knot)
    torch.testing.assert_close(target.knot_dt, original_knot_dt)


def test_copy_only_index_skips_scalar_metadata_but_copies_batched_data() -> None:
    target = JointState.from_position(torch.zeros(3, 1), ["joint"])
    source = JointState.from_position(torch.arange(3.0).unsqueeze(-1), ["joint"])
    target.knot_dt = torch.tensor(0.25)
    source.knot_dt = torch.tensor(0.5)

    target.copy_only_index(source, torch.tensor([1, 2]))

    torch.testing.assert_close(target.position[1:], source.position[1:])
    torch.testing.assert_close(target.knot_dt, torch.tensor(0.25))
