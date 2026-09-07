"""Import and command-line smoke tests for the packaged examples."""
from __future__ import annotations

import importlib

import pytest


MODULES = [
    "curobo.examples.getting_started.feature_mapping",
    "curobo.examples.getting_started.humanoid_retargeting",
    "curobo.examples.getting_started.volumetric_mapping",
    "curobo.examples.guides.custom_optimization",
    "curobo.examples.reference.lidar_volumetric_mapping",
    "curobo.examples.reference.live_volumetric_mapping_mpc",
    "curobo.examples.reference.robot_pose_calibration",
    "curobo.examples.reference.sphere_fit_comparison",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_example_imports_are_platform_safe(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert callable(module.main)


def test_examples_expose_portable_helpers() -> None:
    from curobo.examples.getting_started.humanoid_retargeting import validate_pose_sequence
    from curobo.examples.guides.custom_optimization import quadratic_cost

    import numpy as np
    import torch

    assert validate_pose_sequence({"hand": np.zeros((2, 7))}) == 2
    assert quadratic_cost(torch.zeros(1, 2), torch.ones(1, 2)).item() == 2
