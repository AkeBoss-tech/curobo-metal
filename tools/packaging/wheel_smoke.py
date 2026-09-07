#!/usr/bin/env python3
"""Fail unless an installed curobo-metal wheel exposes its drop-in foundation."""

from __future__ import annotations

from importlib.metadata import packages_distributions, version
from pathlib import Path

import torch

import curobo
from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.curobolib.cuda_ops.tensor_checks import check_float32_tensors
from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper
from curobo._src.optim.gradient.gradient_descent import GradientDescentOpt
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegratorCfg,
)
from curobo._src.robot.loader import KinematicsLoader
from curobo._src.util.cuda_graph_util import create_graph_executor
from curobo._src.util.sampling.sequencer_halton import HaltonSequencer
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.content import get_assets_path, get_robot_path, get_task_configs_path
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.optim import LBFGSOptCfg, MPPICfg
from curobo.perception import Mapper, MapperCfg
from curobo.robot_builder import RobotBuilder
from curobo.robot_parser import UrdfRobotParser
from curobo.rollout import RosenbrockCfg, RosenbrockRollout
from curobo.scene import Cuboid, Scene
from curobo.sphere_fit import SphereFitType
from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
from curobo.types import DeviceCfg, JointState, Pose
from curobo.util_file import load_yaml


def main() -> None:
    package_root = Path(curobo.__file__).resolve().parent
    assert "site-packages" in package_root.as_posix(), package_root
    owners = set(packages_distributions().get("curobo", []))
    assert owners == {"curobo-metal"}, (
        "the curobo namespace must be owned only by curobo-metal; "
        f"found {sorted(owners)}"
    )
    assert curobo.__version__ == version("curobo-metal")

    robot_config = get_robot_path("franka")
    config = load_yaml(str(robot_config))["robot_cfg"]["kinematics"]
    urdf = get_assets_path() / config["urdf_path"]
    assert robot_config.is_file(), robot_config
    assert urdf.is_file(), urdf

    # Solver and rollout constructors resolve these files lazily.  Checking only
    # the robot assets lets a wheel import successfully and then fail when a
    # consumer first creates an IK/MPC/trajectory rollout.
    task_configs = get_task_configs_path()
    required_task_configs = (
        "metrics_base.yml",
        "graph_planner/exact_graph_planner.yml",
        "graph_planner/transition_graph_planner.yml",
        "ik/lbfgs_ik.yml",
        "ik/lbfgs_retarget_ik.yml",
        "ik/particle_ik.yml",
        "ik/transition_ik.yml",
        "mpc/lbfgs_mpc.yml",
        "mpc/lbfgs_retarget_mpc.yml",
        "mpc/transition_bspline_mpc.yml",
        "trajopt/lbfgs_bspline_trajopt.yml",
        "trajopt/particle_trajopt.yml",
        "trajopt/transition_bspline_trajopt.yml",
    )
    for relative in required_task_configs:
        task_config = task_configs / relative
        assert task_config.is_file(), task_config
        assert isinstance(load_yaml(str(task_config)), dict), task_config

    device_cfg = DeviceCfg(torch.device("cpu"), torch.float32)
    pose = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device_cfg)
    state = JointState.from_position(torch.zeros(1, 7))
    assert pose.get_matrix().shape == (1, 4, 4)
    assert state.position.shape == (1, 7)

    kinematics_cfg = KinematicsCfg.from_robot_yaml_file(
        "franka.yml", device_cfg=device_cfg
    )
    kinematics = Kinematics(kinematics_cfg)
    robot_state = kinematics.compute_kinematics(
        JointState.from_position(
            kinematics.default_joint_position,
            joint_names=kinematics.joint_names,
        )
    )
    assert robot_state.tool_poses.position.shape == (1, 1, 1, 3)
    # The upstream-compatible layout retains four disabled attachment slots.
    assert robot_state.robot_spheres.shape[-2:] == (65, 4)

    # These imports are part of the public drop-in contract. Construction is
    # covered by the source suite because it requires a complete scene config.
    assert RobotCollisionChecker.__name__ == "RobotSceneCollision"
    assert RobotCollisionCheckerCfg.__name__ == "RobotSceneCollisionCfg"

    rollout = RosenbrockRollout(RosenbrockCfg(device_cfg))
    rollout_cost = rollout.evaluate_action(torch.zeros(1, 1, 2))
    assert rollout_cost.costs_and_constraints.get_sum_cost().shape == (1, 1)
    assert LBFGSOptCfg().solver_name == "lbfgs"
    assert MPPICfg(num_particles=8).num_particles == 8

    ik_cfg = InverseKinematicsCfg.create("franka.yml", num_seeds=1)
    inverse_kinematics = InverseKinematics(ik_cfg)
    assert inverse_kinematics.joint_names == kinematics.joint_names
    assert TrajectoryOptimizer.__name__ == "TrajOptSolver"
    assert TrajectoryOptimizerCfg.__name__ == "TrajOptSolverCfg"

    parser = UrdfRobotParser(str(urdf))
    assert parser.root_link == "base_link"
    builder_cfg = RobotBuilder(str(urdf), tool_frames=["panda_hand"]).build()
    assert KinematicsLoader(builder_cfg).kinematics_config.num_dof == 9

    planner_cfg = MotionPlannerCfg.create(
        "franka.yml", num_ik_seeds=1, num_trajopt_seeds=1
    )
    assert MotionPlanner(planner_cfg).joint_names == kinematics.joint_names
    assert MapperCfg((0.2, 0.2, 0.2), voxel_size=0.05).grid_shape == (4, 4, 4)
    assert Mapper.__name__ == "Mapper"
    assert len(Scene(cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])])) == 1
    assert SphereFitType.VOXEL.value == "voxel"

    check_float32_tensors(torch.device("cpu"), state=state.position)
    graph_executor = create_graph_executor(
        lambda value: value.square(), "cpu", use_cuda_graph=True
    )
    assert graph_executor(torch.tensor([2.0])).item() == 4.0
    sequence = HaltonSequencer(2, seed=7)
    first = sequence.random(3)
    sequence.reset()
    assert (first == sequence.random(3)).all()
    assert AttachmentManager.__name__ == "AttachmentManager"
    assert ConvexPolygon2DHelper.__name__ == "ConvexPolygon2DHelper"
    assert GradientDescentOpt.__name__ == "GradientDescentOpt"
    assert BlockSparseESDFIntegratorCfg(voxel_size=0.05).voxel_size == 0.05


if __name__ == "__main__":
    main()
