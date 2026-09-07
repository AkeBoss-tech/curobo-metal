"""Optional Viser visualization adapter with the pinned public shape."""

from __future__ import annotations

import tempfile
from typing import Any as trimesh
from typing import Any as yourdfpy
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
try:
    import trimesh as _trimesh
except ImportError:  # Optional visualization dependency.
    _trimesh = None
else:
    trimesh = _trimesh
try:
    import yourdfpy as _yourdfpy
except ImportError:  # Optional visualization dependency.
    _yourdfpy = None
else:
    yourdfpy = _yourdfpy
try:
    import viser
    from viser.extras import ViserUrdf
except ImportError:  # Optional visualization dependency.
    viser = None
    ViserUrdf = None

from curobo._src.geom.types import SceneCfg, Sphere
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.robot.loader.util import load_robot_yaml
from curobo._src.state.state_joint import JointState
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise
from curobo._src.util_file import join_path


class ViserVisualizer:
    def __init__(
        self,
        content_path: Optional[ContentPath] = None,
        device_cfg: DeviceCfg = DeviceCfg(device=torch.device("cuda"), dtype=torch.float32),
        add_robot_to_scene: bool = False,
        connect_ip: str = "0.0.0.0",
        connect_port: int = 8080,
        initialize_viser: bool = True,
        add_control_frames: bool = True,
        visualize_robot_spheres: bool = False,
        visualize_collision_meshes: bool = False,
    ):
        del initialize_viser
        raise NotImplementedError(
            "Viser external integration is unavailable in the portable Metal package"
        )
        # Match the pinned CUDA-labelled declaration default while keeping the
        # actual portable visualization/kinematics path on MPS or CPU.
        if device_cfg.device.type == "cuda":
            device_cfg = DeviceCfg(
                device=torch.device("mps", 0)
                if torch.backends.mps.is_available()
                else torch.device("cpu"),
                dtype=device_cfg.dtype,
                collision_geometry_dtype=device_cfg.collision_geometry_dtype,
                collision_gradient_dtype=device_cfg.collision_gradient_dtype,
                collision_distance_dtype=device_cfg.collision_distance_dtype,
            )
        self._visualize_robot_spheres = visualize_robot_spheres
        self._server = viser.ViserServer(host=connect_ip, port=connect_port)
        self._server.scene.add_grid("/ground_plane", width=20, height=20)
        self._robot_model = None
        self._mesh_root = None
        self._kinematics = None
        self._viser_joint_names = []
        self._vis_frames = []
        self._control_frames = {}
        self._robot_spheres = []
        if add_robot_to_scene and content_path is None:
            log_and_raise("Content path is required to add robot to scene")
        if content_path is None:
            return
        robot_data = load_robot_yaml(content_path) if not isinstance(content_path, dict) else content_path
        if "robot_cfg" not in robot_data:
            robot_data = {"robot_cfg": robot_data}
        source_kinematics = robot_data["robot_cfg"]["kinematics"]
        source_kinematics["load_tool_frames_with_mesh"] = True
        self._vis_frames = source_kinematics["tool_frames"].copy()
        self._robot_model = KinematicsCfg.from_data_dict(
            robot_data["robot_cfg"]["kinematics"], device_cfg=device_cfg
        )
        self._kinematics = Kinematics(self._robot_model)
        generator_config = self._robot_model.generator_config
        if generator_config is None:
            self._mesh_root = join_path(
                content_path.robot_asset_root_path, source_kinematics["asset_root_path"]
            )
            extra_links = source_kinematics.get("extra_links")
        else:
            self._mesh_root = generator_config.asset_root_path
            extra_links = generator_config.extra_links
        if extra_links and generator_config is not None and self._robot_model.kinematics_parser is not None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".urdf", delete=False, mode="w", prefix="curobo_viz_"
            )
            self._kinematics.config.kinematics_config.export_to_urdf(
                output_path=tmp.name,
                kinematics_parser=self._robot_model.kinematics_parser,
            )
            urdf_path = tmp.name
        else:
            urdf_path = (
                join_path(content_path.robot_urdf_root_path, source_kinematics["urdf_path"])
                if generator_config is None
                else generator_config.urdf_path
            )
        self._urdf = yourdfpy.URDF.load(
            urdf_path,
            load_meshes=True,
            build_scene_graph=True,
            filename_handler=self._file_name_handler,
            build_collision_scene_graph=visualize_collision_meshes,
            load_collision_meshes=visualize_collision_meshes,
        )
        self._viser_urdf = ViserUrdf(
            self._server,
            self._urdf,
            root_node_name="/" + (
                source_kinematics["base_link"]
                if generator_config is None
                else generator_config.base_link
            ),
            load_collision_meshes=visualize_collision_meshes,
        )
        self._viser_update_joint_names = list(self._viser_urdf.get_actuated_joint_names())
        active_joint_names = set(self._kinematics.joint_names)
        self._viser_joint_names = [
            name
            for name in self._viser_update_joint_names
            if name in active_joint_names
        ]
        self.reset_robot()
        if add_control_frames:
            kin_state = self._kinematics.compute_kinematics(self._kinematics.default_joint_state)
            if kin_state.tool_poses is None:
                log_and_raise("KinematicsState.tool_poses is None; check tool_frames / kinematics config.")
            for frame_name in self._vis_frames:
                self._control_frames[frame_name] = self.add_control_frame(
                    "/target_" + frame_name, kin_state.tool_poses[frame_name], scale=0.1
                )

    def _file_name_handler(self, fname: str) -> str:
        return join_path(self._mesh_root, fname.replace("package://", ""))

    @property
    def joint_names(self) -> List[str]:
        return self._viser_joint_names

    def update_robot_spheres(self, joint_state: JointState):
        self._robot_spheres = []
        joint_state = self._kinematics.get_active_js(joint_state.clone())
        spheres = self._kinematics.get_robot_as_spheres(
            joint_state.position.contiguous().view(1, -1), filter_valid=True
        )[0]
        self.add_batched_spheres(spheres)

    def add_frame(self, frame_name: str, frame_pose: Pose, scale: float = 0.2) -> viser.FrameHandle:
        del scale
        return self._server.scene.add_frame(
            frame_name,
            position=frame_pose.position.cpu().squeeze().numpy(),
            wxyz=frame_pose.quaternion.cpu().squeeze().numpy(),
        )

    def add_batched_frames(self, frame_name: str, frame_poses: Pose):
        self._batched_frames_handle = self._server.scene.add_batched_axes(
            name=frame_name,
            batched_positions=frame_poses.position.cpu().squeeze().numpy(),
            batched_wxyzs=frame_poses.quaternion.cpu().squeeze().numpy(),
            axes_length=0.02,
            axes_radius=0.005,
        )

    def add_control_frame(
        self, frame_name: str, frame_pose: Pose, scale: float = 0.2
    ) -> viser.TransformControlsHandle:
        return self._server.scene.add_transform_controls(
            frame_name,
            scale=scale,
            position=frame_pose.position.cpu().squeeze().numpy(),
            wxyz=frame_pose.quaternion.cpu().squeeze().numpy(),
        )

    def reset_robot(self):
        joint_state = self._kinematics.get_full_js(self._kinematics.default_joint_state)
        self.set_joint_state(joint_state)

    def set_joint_positions(
        self, joint_positions: Sequence[float], joint_names: List[str], **kwargs
    ) -> None:
        self.set_joint_state(JointState.from_position(joint_positions, joint_names=joint_names))

    def get_control_frame_pose(self) -> Dict[str, Pose]:
        return {
            frame_name: Pose.from_numpy(
                self._control_frames[frame_name].position,
                self._control_frames[frame_name].wxyz,
            )
            for frame_name in self._vis_frames
        }

    def set_joint_state(self, joint_state: JointState) -> None:
        joint_state = joint_state.to(self._kinematics.device_cfg)
        viser_set = set(self._viser_update_joint_names)
        js_set = set(joint_state.joint_names or [])
        if viser_set != js_set and viser_set - js_set:
            joint_state = self._kinematics.get_full_js(joint_state)
        if self._visualize_robot_spheres:
            self.update_robot_spheres(joint_state)
        joint_state.reindex(self._viser_update_joint_names)
        self._viser_urdf.update_cfg(joint_state.position.cpu().squeeze().numpy())

    def add_batched_spheres(self, spheres: List[Sphere]):
        sphere_mesh = spheres[0].get_trimesh_mesh()
        mesh_radius = spheres[0].radius
        self._batched_spheres_handle = self._server.scene.add_batched_meshes_simple(
            name="curobo_spheres",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_scales=[sphere.radius / mesh_radius for sphere in spheres],
            batched_positions=[sphere.pose[0:3] for sphere in spheres],
            batched_wxyzs=[sphere.pose[3:7] for sphere in spheres],
        )

    def add_batched_spheres_from_position(
        self,
        position: np.ndarray,
        radius: np.ndarray,
        color=None,
        name: str = "curobo_spheres",
    ):
        radius = np.asarray(radius).copy()
        radius[radius < 0.001] = 0.001
        sphere_mesh = Sphere(
            name="curobo_spheres", pose=[0, 0, 0, 1, 0, 0, 0], radius=radius[0]
        ).get_trimesh_mesh()
        quaternion = np.zeros((len(position), 4))
        quaternion[:, 0] = 1.0
        colors = np.zeros((len(position), 3))
        if color is not None:
            colors[:, :] = color
        self._batched_spheres_handle = self._server.scene.add_batched_meshes_simple(
            name=name,
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            batched_scales=radius / radius[0],
            batched_positions=position,
            batched_wxyzs=quaternion,
            batched_colors=colors,
        )

    def add_sphere(self, sphere: Sphere):
        if not isinstance(sphere, Sphere):
            log_and_raise("Sphere is not a valid Sphere object")
        return self._server.scene.add_icosphere(
            name=sphere.name,
            position=np.ravel(sphere.position),
            radius=sphere.radius,
            color=sphere.color if sphere.color is not None else (0, 200, 0),
        )

    def add_line_segments(self, line_segments: np.ndarray, color: np.ndarray):
        self._server.scene.add_line_segments(
            "/line_segments", points=line_segments, colors=color, line_width=3.0
        )

    def add_mesh(self, mesh_trimesh: trimesh.Trimesh, name: str = "mesh"):
        if not isinstance(mesh_trimesh, trimesh.Trimesh):
            if not hasattr(mesh_trimesh, "vertices") or not hasattr(mesh_trimesh, "faces"):
                log_and_raise("Mesh is not a valid Mesh object")
            mesh_trimesh = trimesh.Trimesh(
                vertices=np.asarray(mesh_trimesh.vertices),
                faces=np.asarray(mesh_trimesh.faces),
                process=False,
            )
        return self._server.scene.add_mesh_trimesh(name=name, mesh=mesh_trimesh)

    def add_scene(self, scene_cfg: SceneCfg, add_control_frames: bool = False):
        obstacle_frames = {}
        mesh_scene = SceneCfg.create_mesh_scene(scene_cfg)
        for mesh in mesh_scene.mesh:
            self.add_mesh(
                mesh.get_trimesh_mesh(transform_with_pose=not add_control_frames),
                name="/obstacles/" + mesh.name + "/mesh",
            )
            if add_control_frames:
                obstacle_frames[mesh.name] = self.add_control_frame(
                    "/obstacles/" + mesh.name, Pose.from_list(mesh.pose), scale=0.2
                )
        return obstacle_frames

    def add_point_cloud(
        self,
        pointcloud: np.ndarray,
        colors=[200, 200, 200],
        point_size: float = 0.005,
        name: str = "pointcloud",
    ):
        return self._server.scene.add_point_cloud(
            name=name, points=pointcloud, colors=colors, point_size=point_size
        )

    def add_image(
        self,
        image: np.ndarray,
        render_width: float,
        render_height: float,
        pose: Pose,
        name: str = "image",
    ):
        return self._server.scene.add_image(
            name=name,
            image=image,
            render_width=render_width,
            render_height=render_height,
            position=pose.position.cpu().squeeze().numpy(),
            wxyz=pose.quaternion.cpu().squeeze().numpy(),
        )
