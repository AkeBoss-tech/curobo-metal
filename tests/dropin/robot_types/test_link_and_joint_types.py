"""Pinned V2 link-record and joint-enum behavior on portable CPU/MPS paths."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.link_params import DeviceCfg, LinkParams, Pose, log_and_raise


def test_joint_type_values_and_round_trip_by_name_and_value():
    expected = {
        "FIXED": -1,
        "X_PRISM": 0, "Y_PRISM": 1, "Z_PRISM": 2,
        "X_ROT": 3, "Y_ROT": 4, "Z_ROT": 5,
        "X_PRISM_NEG": 6, "Y_PRISM_NEG": 7, "Z_PRISM_NEG": 8,
        "X_ROT_NEG": 9, "Y_ROT_NEG": 10, "Z_ROT_NEG": 11,
    }
    assert {member.name: member.value for member in JointType} == expected
    for name, value in expected.items():
        assert JointType[name] is JointType(value)


def test_link_params_create_converts_pose_without_mutating_source():
    source = {
        "link_name": "tool",
        "joint_name": "tool_joint",
        "joint_type": "Z_ROT",
        "fixed_transform": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        "parent_link_name": "base",
    }
    params = LinkParams.create(source)
    assert params.joint_type is JointType.Z_ROT
    np.testing.assert_allclose(params.fixed_transform, np.array([
        [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 1.0, 3.0],
    ]))
    assert source["joint_type"] == "Z_ROT"
    assert len(source["fixed_transform"]) == 7


def test_link_params_create_accepts_native_affine_and_enum():
    affine = np.concatenate((np.eye(3), np.array([[0.5], [0.0], [-0.25]])), axis=1)
    params = LinkParams.create({
        "link_name": "tool", "joint_name": "tool_joint", "joint_type": JointType.X_PRISM,
        "fixed_transform": affine,
    })
    assert params.joint_type is JointType.X_PRISM
    np.testing.assert_allclose(params.fixed_transform, affine)
    affine[0, 3] = 99.0
    assert params.fixed_transform[0, 3] == pytest.approx(0.5)


@pytest.mark.parametrize("transform", [np.eye(4), [0.0] * 6, np.zeros((2, 6))])
def test_link_params_rejects_invalid_transform_shape(transform):
    with pytest.raises(ValueError, match="fixed_transform"):
        LinkParams.create({
            "link_name": "tool", "joint_name": "tool_joint", "joint_type": "FIXED",
            "fixed_transform": transform,
        })


def test_link_params_validates_com_only_when_queried_like_pinned_v2():
    params = LinkParams("tool", "joint", JointType.FIXED, np.eye(4)[:3], link_com=np.zeros(2))
    with pytest.raises(ValueError, match="link_com"):
        params.get_link_com_and_mass()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple Metal")
def test_public_pose_and_device_cfg_reexports_support_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    device_cfg = DeviceCfg(torch.device("mps"))
    pose = Pose.from_list([0.25, -0.5, 1.0, 1.0, 0.0, 0.0, 0.0], device_cfg=device_cfg)
    assert pose.device.type == "mps"
    np.testing.assert_allclose(pose.get_numpy_affine_matrix()[0, :3, 3], [0.25, -0.5, 1.0])


def test_public_log_and_raise_reexport_is_callable():
    with pytest.raises(RuntimeError, match="portable error"):
        log_and_raise("portable error", exception_type=RuntimeError)
