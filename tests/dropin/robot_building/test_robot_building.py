import importlib

import pytest
import torch

from curobo.content import get_assets_path
from curobo.kinematics import KinematicsCfg
from curobo.robot_builder import RobotBuilder
from curobo.robot_parser import UrdfRobotParser
from curobo._src.robot.dynamics import Dynamics, DynamicsCfg
from curobo._src.robot.kinematics.kinematics_reducer import KinematicsReducer
from curobo._src.robot.loader import KinematicsLoader
from curobo._src.robot.types import CSpaceParams, JointLimits, JointType
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.content_path import ContentPath
from curobo._src.util.xrdf_util import convert_xrdf_to_curobo
from curobo._src.util_file import load_yaml


URDF = get_assets_path() / "robot/franka_description/franka_panda.urdf"


def test_owned_pinned_modules_import():
    names = [
        "curobo.robot_builder", "curobo.robot_parser",
        "curobo._src.robot.builder.builder_robot",
        "curobo._src.robot.builder.debugger_robot",
        "curobo._src.robot.dynamics.dynamics",
        "curobo._src.robot.dynamics.dynamics_cfg",
        "curobo._src.robot.loader.kinematics_loader",
        "curobo._src.robot.loader.kinematics_loader_cfg",
        "curobo._src.robot.loader.util",
        "curobo._src.robot.parser.parser_base",
        "curobo._src.robot.parser.parser_urdf",
        "curobo._src.robot.types.collision_geometry",
        "curobo._src.robot.types.cspace_params",
        "curobo._src.robot.types.joint_limits",
        "curobo._src.robot.types.joint_types",
        "curobo._src.robot.types.link_params",
        "curobo._src.robot.types.self_collision_params",
        "curobo._src.robot.kinematics.kinematics_reducer",
    ]
    for name in names:
        importlib.import_module(name)


def test_urdf_parser_preserves_tree_mimic_and_inertials():
    parser = UrdfRobotParser(str(URDF))
    assert parser.root_link == "base_link"
    assert parser.get_chain("base_link", "panda_hand")[-1] == "panda_hand"
    assert "panda_joint1" in parser.get_actuated_joint_names()
    assert parser.get_mimic_joint_map() == {}
    params = parser.get_link_parameters("panda_link1")
    assert params.joint_type is JointType.Z_ROT
    assert params.fixed_transform.shape == (3, 4)
    assert params.link_mass > 0
    assert "robot" in parser.get_urdf_string()
    assert parser.get_link_mesh("panda_link1") is None


def test_builder_and_loader_compile_full_urdf():
    builder = RobotBuilder(str(URDF), tool_frames=["panda_hand"])
    config = builder.build()
    loader = KinematicsLoader(config)
    assert loader.kinematics_config.num_dof == 9
    assert loader.kinematics_parser.root_link == "base_link"
    limits = loader.get_joint_limits()
    assert limits.position.shape == (2, 9)
    assert builder._create_neighbor_ignore_matrix()["panda_link1"] == [
        "panda_link0", "panda_link2"
    ]
    spheres = builder.fit_collision_spheres()
    assert sum(len(values) for values in spheres.values()) > 0


def test_builder_collects_exact_primitive_spheres_and_round_trips_yaml(tmp_path):
    urdf = tmp_path / "primitive.urdf"
    urdf.write_text(
        """<robot name="primitive">
  <link name="base"><collision><origin xyz="0.25 0 0"/>
    <geometry><sphere radius="0.1"/></geometry></collision></link>
  <link name="tool"/>
  <joint name="tool_joint" type="fixed"><parent link="base"/><child link="tool"/>
    <origin xyz="0 0 1" rpy="0 0 0"/></joint>
</robot>""",
        encoding="utf-8",
    )
    builder = RobotBuilder(str(urdf), tool_frames=["tool"])
    spheres = builder.fit_collision_spheres(
        use_collision_mesh=True, compute_metrics=True,
        clip_links={"base": ("x", 0.2)},
    )
    assert spheres == {"base": [{"center": [0.25, 0.0, 0.0], "radius": 0.1}]}
    assert builder.link_metrics["base"].coverage == 1.0
    assert builder.num_spheres == 1
    matrix = builder.compute_collision_matrix()
    assert matrix["base"] == ["tool"]
    builder.add_collision_ignore("tool", ["base"])
    config = builder.build()
    assert config.collision_link_names == ["base"]
    saved = tmp_path / "primitive.yml"
    builder.save(config, str(saved))
    loaded = RobotBuilder.from_config(str(saved))
    assert loaded.collision_spheres == {"base": [{"center": [0.25, 0.0, 0.0], "radius": 0.1}]}
    assert len(builder.refit_link_spheres("base", num_spheres=2, use_collision_mesh=True)) == 2


def test_builder_save_serializes_runtime_cspace_and_xrdf(tmp_path):
    """Loaded configs materialize cspace tensors but remain editable/savable."""
    builder = RobotBuilder.from_config(str(get_assets_path().parent / "configs/robot/franka.yml"))
    config = builder.build()
    yaml_path = tmp_path / "franka-edited.yml"
    builder.save(config, str(yaml_path))
    saved = load_yaml(str(yaml_path))
    cspace = saved["kinematics"]["cspace"]
    assert isinstance(cspace["default_joint_position"], list)
    assert isinstance(cspace["max_acceleration"], list)
    reloaded = RobotBuilder.from_config(str(yaml_path))
    assert reloaded._cspace_config["joint_names"] == builder._cspace_config["joint_names"]

    xrdf_path = tmp_path / "franka.xrdf"
    builder.save_xrdf(config, str(xrdf_path), geometry_name="portable_collision")
    xrdf = load_yaml(str(xrdf_path))
    assert xrdf["format"] == "xrdf"
    assert "portable_collision" in xrdf["geometry"]
    converted = convert_xrdf_to_curobo(
        content_path=ContentPath(
            robot_urdf_absolute_path=str(URDF),
            robot_asset_absolute_path=str(get_assets_path()),
        ),
        input_xrdf_dict=xrdf,
    )
    assert converted["robot_cfg"]["kinematics"]["cspace"]["joint_names"] == cspace["joint_names"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_builder_save_materializes_mps_cspace_as_yaml(tmp_path):
    builder = RobotBuilder.from_config(
        str(get_assets_path().parent / "configs/robot/franka.yml"),
        device_cfg=DeviceCfg(torch.device("mps")),
    )
    config = builder.build()
    assert config.cspace.default_joint_position.device.type == "mps"
    output = tmp_path / "franka-mps.yml"
    builder.save(config, str(output))
    assert isinstance(load_yaml(str(output))["kinematics"]["cspace"]["max_jerk"], list)


def test_joint_limits_and_cspace_reindex_scale_clone():
    cfg = CSpaceParams(
        ["a", "b"], [0.0, 1.0], [1.0, 2.0], [1.0, 1.0],
        velocity_scale=[0.5, 0.25],
    )
    limits = JointLimits.from_data_dict({
        "joint_names": ["a", "b"],
        "position": [[-1, -2], [1, 2]],
        "velocity": [[-4, -8], [4, 8]],
        "acceleration": [[-2, -2], [2, 2]],
        "jerk": [[-10, -10], [10, 10]],
    })
    scaled = cfg.scale_joint_limits(limits)
    torch.testing.assert_close(scaled.velocity[1], torch.tensor([2.0, 2.0]))
    cfg.inplace_reindex(["b", "a"])
    assert cfg.joint_names == ["b", "a"]
    torch.testing.assert_close(cfg.default_joint_position, torch.tensor([1.0, 0.0]))


def test_franka_dynamics_batches_and_autograd():
    cfg = KinematicsCfg.from_robot_yaml_file("franka.yml")
    dynamics = Dynamics(DynamicsCfg(cfg.kinematics_config, cfg.device_cfg))
    q = torch.zeros((2, 3, cfg.dof), requires_grad=True)
    state = JointState(
        q, torch.zeros_like(q), torch.zeros_like(q),
        joint_names=cfg.kinematics_config.joint_names,
    )
    torque = dynamics.compute_inverse_dynamics(state)
    assert torque.shape == q.shape and torch.isfinite(torque).all()
    torch.autograd.grad(torque.sum(), q)
    dynamics.update_link_mass("panda_link1", 3.0)
    with pytest.raises(ValueError, match="f_ext must end"):
        dynamics.compute_inverse_dynamics(
            state, f_ext=torch.ones((2, 3, 1, 6))
        )


def test_kinematics_reducer_and_state_reconstruction():
    cfg = KinematicsCfg.from_robot_yaml_file("franka.yml")
    reduced = KinematicsReducer.reduce_dof(
        cfg.kinematics_config, ["panda_link4"], remove_collision_spheres=True
    )
    assert reduced.tool_frames == ["panda_link4"]
    assert reduced.num_dof == 4
    state = JointState.from_position(
        torch.arange(4.0), joint_names=reduced.joint_names
    )
    locked = JointState.from_position(torch.tensor([9.0]), joint_names=["extra"])
    rebuilt = KinematicsReducer.reconstruct_joint_state(
        state, locked, reduced.joint_names + ["extra"]
    )
    assert rebuilt.joint_names[-1] == "extra"
    assert rebuilt.position[-1].item() == 9.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_dynamics_mps_without_fallback(monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from curobo._src.types.device_cfg import DeviceCfg
    cfg = KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=DeviceCfg(torch.device("mps"))
    )
    dynamics = Dynamics(DynamicsCfg(cfg.kinematics_config, cfg.device_cfg))
    q = torch.zeros((1, cfg.dof), device="mps")
    result = dynamics.compute_inverse_dynamics(
        JointState(
            q, torch.zeros_like(q), torch.zeros_like(q),
            joint_names=cfg.kinematics_config.joint_names,
        )
    )
    assert result.device.type == "mps"
