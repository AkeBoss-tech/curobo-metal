"""Run dependency-free equivalents of cuRobo V2 getting-started workflows.

The pinned V2 tutorials assume CUDA.  This example exercises the corresponding
public cuRobo-compatible interfaces on CPU or Apple Metal instead: Franka FK
with autograd, pose IK, a pose-to-pose motion-plan route, a primitive-URDF
robot-model round trip, and dense RGB-D/ESDF mapping.  It deliberately avoids
external datasets, mesh assets, Viser, and CUDA/Warp kernels.

Examples
--------

.. code-block:: bash

   PYTORCH_ENABLE_MPS_FALLBACK=0 python examples/getting_started_portable.py --device mps
   python examples/getting_started_portable.py --device cpu --json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.perception import Mapper, MapperCfg
from curobo.robot_builder import RobotBuilder
from curobo.scene import Cuboid, Scene
from curobo.types import CameraObservation, DeviceCfg, GoalToolPose, JointState, Pose


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps was requested but MPS is unavailable")
    if device.type not in {"cpu", "mps"}:
        raise ValueError("portable getting-started flows support only cpu or mps")
    return device


def run_forward_kinematics(device: torch.device) -> dict[str, Any]:
    """Run batched, differentiable Franka forward kinematics."""
    robot = Kinematics(
        KinematicsCfg.from_robot_yaml_file("franka.yml", device_cfg=DeviceCfg(device)),
        compute_jacobian=True,
    )
    joint_position = torch.zeros((8, robot.dof), device=device, requires_grad=True)
    state = robot.compute_kinematics(
        JointState.from_position(joint_position, joint_names=robot.joint_names)
    )
    loss = state.tool_poses.position.square().sum()
    loss.backward()
    if joint_position.grad is None or not torch.isfinite(joint_position.grad).all():
        raise RuntimeError("FK did not produce finite joint gradients")
    return {
        "dof": robot.dof,
        "tool_position_shape": list(state.tool_poses.position.shape),
        "sphere_count": robot.total_spheres,
        "gradient_norm": float(joint_position.grad.norm().item()),
    }


def run_inverse_kinematics(device: torch.device) -> dict[str, Any]:
    """Solve a reachable single-pose Franka IK problem."""
    solver = InverseKinematics(
        InverseKinematicsCfg.create("franka.yml", device_cfg=DeviceCfg(device), num_seeds=4)
    )
    target = Pose(
        position=torch.tensor([[0.4, 0.0, 0.4]], device=device),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
    )
    result = solver.solve_pose(
        GoalToolPose.from_poses({solver.tool_frames[0]: target}, num_goalset=1)
    )
    if not bool(result.success.all().item()):
        raise RuntimeError("portable IK did not solve the reachable target")
    return {
        "success": True,
        "position_error": float(result.position_error.max().item()),
        "solution_device": result.solution.device.type,
    }


def run_motion_planning(device: torch.device) -> dict[str, Any]:
    """Plan to the current tool pose through the public pose-planning route."""
    config = MotionPlannerCfg.create(
        "franka.yml",
        device_cfg=DeviceCfg(device),
        graph_planner_config=None,
        num_ik_seeds=2,
        num_trajopt_seeds=1,
        use_cuda_graph=False,
    )
    # A small deterministic smoke workload; not a performance benchmark.
    config.trajopt_solver_config.max_iterations = 2
    planner = MotionPlanner(config)
    current = planner.default_joint_state
    pose = planner.compute_kinematics(current).tool_poses.get_link_pose(planner.tool_frames[0])
    goal = GoalToolPose.from_poses(
        {planner.tool_frames[0]: pose}, ordered_tool_frames=planner.tool_frames
    )
    result = planner.plan_pose(goal, current, max_attempts=1)
    if result is None or not bool(result.success.all().item()):
        raise RuntimeError("portable pose motion plan failed")
    trajectory = result.get_interpolated_plan()
    return {
        "success": True,
        "waypoints": int(trajectory.position.shape[-2]),
        "solution_device": result.solution.device.type,
    }


def run_robot_builder() -> dict[str, Any]:
    """Build, save, and reload a primitive-sphere URDF configuration.

    This is the exact portable subset of the V2 robot-builder tutorial.  The
    bundled Franka URDF references mesh geometry, whose MorphIt fitting path is
    intentionally outside the CPU/MPS implementation boundary.
    """
    urdf = """<robot name=\"portable_sphere\">
  <link name=\"base\"><collision><origin xyz=\"0.1 0 0\"/>
    <geometry><sphere radius=\"0.05\"/></geometry></collision></link>
  <link name=\"tool\"/>
  <joint name=\"tool_joint\" type=\"fixed\"><parent link=\"base\"/><child link=\"tool\"/>
    <origin xyz=\"0 0 0.1\" rpy=\"0 0 0\"/></joint>
</robot>"""
    with tempfile.TemporaryDirectory(prefix="curobo-metal-builder-") as directory:
        directory_path = Path(directory)
        urdf_path = directory_path / "sphere.urdf"
        output_path = directory_path / "sphere.yml"
        urdf_path.write_text(urdf, encoding="utf-8")
        builder = RobotBuilder(str(urdf_path), tool_frames=["tool"])
        # The V2 builder's collision model is derived from ``<collision>``
        # geometry.  The primitive fixture uses that same URDF channel.
        spheres = builder.fit_collision_spheres(
            use_collision_mesh=True, compute_metrics=True
        )
        collision_matrix = builder.compute_collision_matrix()
        builder.save(builder.build(), str(output_path))
        reloaded = RobotBuilder.from_config(str(output_path))
        if not output_path.exists() or reloaded.num_spheres != 1:
            raise RuntimeError("portable primitive robot builder round trip failed")
        return {
            "sphere_count": builder.num_spheres,
            "collision_ignore_entries": sum(len(value) for value in collision_matrix.values()),
            "reloaded_sphere_count": reloaded.num_spheres,
        }


def run_volumetric_mapping(device: torch.device) -> dict[str, Any]:
    """Fuse a synthetic depth image, stamp geometry, and derive an ESDF."""
    mapper = Mapper(
        MapperCfg(
            extent_meters_xyz=(0.4, 0.4, 0.4),
            voxel_size=0.1,
            truncation_distance=0.2,
            enable_static=True,
            device=str(device),
            image_height=8,
            image_width=8,
        )
    )
    observation = CameraObservation(
        depth_image=torch.full((8, 8), 100.0, device=device),
        intrinsics=torch.tensor(
            ((8.0, 0.0, 3.5), (0.0, 8.0, 3.5), (0.0, 0.0, 1.0)), device=device
        ),
        pose=Pose(
            torch.zeros(3, device=device), torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)
        ),
    )
    mapper.integrate(observation)
    stamped = mapper.update_static_obstacles(
        Scene(cuboid=[Cuboid("portable_box", pose=[0, 0, 0.1, 1, 0, 0, 0], dims=[0.1] * 3)])
    )
    esdf = mapper.compute_esdf()
    mesh = mapper.extract_mesh()
    if (
        stamped < 1
        or esdf.feature_tensor is None
        or esdf.feature_tensor.device.type != device.type
    ):
        raise RuntimeError("portable volumetric mapping did not retain a device-resident ESDF")
    return {
        "stamped_voxels": int(stamped),
        "esdf_shape": list(esdf.feature_tensor.shape),
        "mesh_vertices": len(mesh.vertices),
        "esdf_device": esdf.feature_tensor.device.type,
    }


def run(device_name: str) -> dict[str, Any]:
    """Run every portable getting-started flow and return JSON-safe evidence."""
    device = _device(device_name)
    return {
        "device": device.type,
        "forward_kinematics": run_forward_kinematics(device),
        "inverse_kinematics": run_inverse_kinematics(device),
        "motion_planning": run_motion_planning(device),
        "robot_builder": run_robot_builder(),
        "volumetric_mapping": run_volumetric_mapping(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--json", action="store_true", help="emit only the JSON evidence record")
    args = parser.parse_args()
    result = run(args.device)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("Portable getting-started flows: PASS")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
