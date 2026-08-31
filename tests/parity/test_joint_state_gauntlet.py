from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.parity.joint_state_probe import run
from tools.parity.replay_corpus import load
from tools.parity.replay_registry import BY_ID


CORPUS = Path("artifacts/parity/replay/corpus")


def _probe() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw, _ = load(CORPUS, BY_ID["types.joint_state"])
    return raw, run(raw, "cpu")


def test_joint_state_probe_has_broad_observable_schema_and_finite_outputs() -> None:
    _, output = _probe()
    assert set(output) == {
        "packed",
        "reordered_position",
        "reordered_velocity",
        "repeated_position",
        "repeated_dt",
        "kernel_position",
        "kernel_velocity",
        "kernel_dt",
        "scaled_velocity",
        "scaled_acceleration",
        "scaled_jerk",
        "concatenated_position",
        "input_gradient",
        "noncontiguous_packed",
        "derivative_channels_independent",
        "clone_independent",
    }
    assert all(np.isfinite(value).all() for value in output.values())
    assert output["input_gradient"].shape == (2, 2)
    assert output["repeated_position"].shape == (6, 2)
    assert output["kernel_position"].shape == (3, 2)


def test_joint_state_probe_kills_layout_reorder_and_constant_output_mutants() -> None:
    raw, output = _probe()
    expected_packed = np.concatenate(
        [
            raw["joint_position"],
            raw["joint_velocity"],
            raw["joint_acceleration"],
            raw["joint_jerk"],
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(output["packed"], expected_packed)
    np.testing.assert_array_equal(
        output["reordered_position"], raw["joint_position"][:, ::-1]
    )
    np.testing.assert_array_equal(
        output["reordered_velocity"], raw["joint_velocity"][:, ::-1]
    )
    np.testing.assert_array_equal(
        output["concatenated_position"],
        np.concatenate([raw["joint_position"], raw["joint_position"]], axis=-1),
    )


def test_joint_state_probe_kills_scaling_kernel_and_detached_gradient_mutants() -> None:
    raw, output = _probe()
    np.testing.assert_allclose(output["scaled_velocity"], raw["joint_velocity"] * 0.5)
    np.testing.assert_allclose(
        output["scaled_acceleration"], raw["joint_acceleration"] * 0.25
    )
    np.testing.assert_allclose(output["scaled_jerk"], raw["joint_jerk"] * 0.125)
    np.testing.assert_allclose(
        output["kernel_position"], raw["joint_kernel"] @ raw["joint_position"]
    )
    np.testing.assert_allclose(
        output["kernel_velocity"], raw["joint_kernel"] @ raw["joint_velocity"]
    )
    np.testing.assert_allclose(output["kernel_dt"], raw["joint_kernel"] @ raw["joint_dt"])
    np.testing.assert_allclose(
        output["input_gradient"],
        np.asarray([[-0.5, 1.25], [9.25, 9.75]], dtype=np.float32),
    )


def test_joint_state_probe_preserves_noncontiguous_and_aliasing_contracts() -> None:
    raw, output = _probe()
    noncontiguous_position = raw["joint_noncontiguous_source"][:, ::2]
    expected = np.concatenate(
        [noncontiguous_position, np.zeros_like(noncontiguous_position).repeat(3, axis=-1)],
        axis=-1,
    )
    np.testing.assert_array_equal(output["noncontiguous_packed"], expected)
    np.testing.assert_array_equal(output["derivative_channels_independent"], [1])
    np.testing.assert_array_equal(output["clone_independent"], [1])
