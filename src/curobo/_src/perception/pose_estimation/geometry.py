"""Dependency-light geometry models for portable pose estimation."""

from __future__ import annotations

import torch

from curobo._src.types.device_cfg import DeviceCfg


class RigidObjectGeometry:
    def __init__(self, mesh, device_cfg=DeviceCfg()):
        self.mesh = mesh
        self._tensor_args = device_cfg

    def update(self, joint_angles):
        return None

    def sample_surface_points(self, n_points):
        vertices = torch.as_tensor(
            getattr(self.mesh, "vertices"), **self._tensor_args.as_torch_dict()
        )
        faces = torch.as_tensor(getattr(self.mesh, "faces"), device=vertices.device).long()
        if len(faces) == 0:
            raise ValueError("mesh requires at least one triangle")
        selected = faces[torch.arange(n_points, device=faces.device) % len(faces)]
        points = vertices[selected].mean(1)
        normals = torch.linalg.cross(
            vertices[selected[:, 1]] - vertices[selected[:, 0]],
            vertices[selected[:, 2]] - vertices[selected[:, 0]],
        )
        normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True).clamp_min(1e-12)
        return points, normals

    def get_dof(self):
        return 0

    @property
    def device_cfg(self):
        return self._tensor_args


class ArticulatedRobotGeometry:
    def __init__(self, robot_model, device_cfg=DeviceCfg(), **kwargs):
        self.robot_model = robot_model
        self._tensor_args = device_cfg
        self._n_dof = len(robot_model.joint_names)
        if not hasattr(robot_model, "get_robot_link_meshes"):
            raise NotImplementedError("robot model does not expose link meshes for surface sampling")
        self._rigid = [
            RigidObjectGeometry(m.get_trimesh_mesh() if hasattr(m, "get_trimesh_mesh") else m, device_cfg)
            for m in robot_model.get_robot_link_meshes()
        ]
        self._joint_angles = None

    def update(self, joint_angles):
        self._joint_angles = joint_angles

    def sample_surface_points(self, n_points):
        if self._joint_angles is None:
            raise RuntimeError("call update(joint_angles) before sampling articulated geometry")
        raise NotImplementedError(
            "articulated mesh point transforms require the optional upstream mesh/FK bridge"
        )

    def get_dof(self):
        return self._n_dof

    @property
    def device_cfg(self):
        return self._tensor_args
