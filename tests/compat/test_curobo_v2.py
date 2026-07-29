from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from curobo_metal.compat import (
    CUROBO_V2_REVISION,
    UnsupportedCuroboConfig,
    convert_kinematics_config,
)

FIXTURE = Path(__file__).parent / "fixtures" / "two_link_curobo_v2.json"
FORBIDDEN_ROOTS = {"cuda", "isaacsim", "omni", "warp"}


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_mapping_conversion_has_deterministic_shapes_and_values() -> None:
    result = convert_kinematics_config(load_fixture())
    assert result.upstream_revision == CUROBO_V2_REVISION
    assert result.fixed_transforms.shape == (3, 4, 4)
    assert result.fixed_transforms.dtype == np.float64
    np.testing.assert_array_equal(result.fixed_transforms[:, 3], [[0, 0, 0, 1]] * 3)
    np.testing.assert_array_equal(result.fixed_transforms[:, :3, 3], [[0, 0, 0], [1, 0, 0], [.75, 0, 0]])
    np.testing.assert_array_equal(result.parent_link, [0, 0, 1])
    np.testing.assert_array_equal(result.joint_index, [-1, 0, 1])
    np.testing.assert_array_equal(result.joint_type, [-1, 5, 5])
    np.testing.assert_array_equal(result.joint_offset, [[1, 0], [1, 0], [-1, .25]])
    assert result.tool_link.shape == (1,)
    assert result.link_spheres is not None
    assert result.link_spheres.shape == (2, 3, 4)
    assert result.sphere_link is not None
    np.testing.assert_array_equal(result.sphere_link, [0, 1, 2])
    assert result.link_names == ("base_link", "upper_arm", "tool")
    assert result.joint_names == ("shoulder", "elbow")
    assert result.tool_frames == ("tool",)


class FakeTensor:
    """Enough of torch.Tensor's public conversion protocol for an import-free test."""

    def __init__(self, value):
        self.value = np.asarray(value)
        self.detached = False
        self.moved_to_cpu = False

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        self.moved_to_cpu = True
        return self

    def numpy(self):
        assert self.detached and self.moved_to_cpu
        return self.value


class Params:
    pass


class KinematicsCfg:
    pass


def test_duck_typed_kinematics_cfg_uses_tensor_protocol_without_torch() -> None:
    raw = load_fixture()
    params = Params()
    tensors = {}
    for key, value in raw.items():
        converted = FakeTensor(value) if key in {
            "fixed_transforms", "joint_map", "joint_map_type", "joint_offset_map",
            "link_map", "link_sphere_idx_map", "link_spheres", "tool_frame_map",
        } else value
        setattr(params, key, converted)
        if isinstance(converted, FakeTensor):
            tensors[key] = converted
    config = KinematicsCfg()
    config.kinematics_config = params
    result = convert_kinematics_config(config)
    assert result.num_dof == 2
    assert all(item.detached and item.moved_to_cpu for item in tensors.values())


def test_import_does_not_load_gpu_or_simulator_modules() -> None:
    code = """
import json
import sys
before = set(sys.modules)
import curobo_metal.compat
roots = {name.partition('.')[0] for name in set(sys.modules) - before}
print(json.dumps(sorted(roots)))
"""
    loaded = set(json.loads(subprocess.check_output([sys.executable, "-c", code], text=True)))
    assert loaded.isdisjoint(FORBIDDEN_ROOTS | {"torch"})


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"link_map": [0, 0, 0]}, "serial chain"),
        ({"joint_map_type": [-1, 5, 99]}, "JointType"),
        ({"joint_map": [0, 0, 1]}, "fixed links"),
        ({"joint_map": [-1, 0, 0]}, "cover"),
        ({"tool_frame_map": [9]}, "out-of-range"),
        ({"link_sphere_idx_map": None}, "appear together"),
    ],
)
def test_unsupported_or_inconsistent_configs_fail_explicitly(update, message) -> None:
    raw = load_fixture()
    raw.update(update)
    with pytest.raises(UnsupportedCuroboConfig, match=message):
        convert_kinematics_config(raw)


def test_input_arrays_are_copied() -> None:
    raw = load_fixture()
    original = np.asarray(raw["fixed_transforms"], dtype=np.float32)
    raw["fixed_transforms"] = original
    result = convert_kinematics_config(raw)
    original[1, 0, 3] = 999
    assert result.fixed_transforms[1, 0, 3] == 1
    assert result.fixed_transforms.flags.c_contiguous
