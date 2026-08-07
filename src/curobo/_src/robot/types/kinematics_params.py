"""Portable representation of the pinned kinematics parameter object."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET
import math

import torch

from .collision_geometry import RobotCollisionGeometry
from .cspace_params import CSpaceParams
from .joint_limits import JointLimits
from .joint_types import JointType
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class KinematicsParams:
    """Tensor/model metadata consumed by :class:`curobo.kinematics.Kinematics`.

    The CUDA implementation exposes a very large tensor record.  This portable
    implementation preserves its commonly consumed public attributes while the
    production tree-kinematics backend owns the compiled representation.
    """

    robot_cfg: Any
    reference_link_spheres: torch.Tensor | None = field(default=None, init=False)
    _link_spheres: torch.Tensor | None = field(default=None, init=False, repr=False)
    # These records are intentionally derived from the portable tree rather
    # than copied from a CUDA loader.  Keeping them cached gives callers the
    # same stable tensor identity between reads while still allowing
    # ``copy_``/``clone`` to preserve caller-owned buffers.
    _tree_cache: Any = field(default=None, init=False, repr=False)
    _fixed_transforms: torch.Tensor | None = field(default=None, init=False, repr=False)
    _link_masses_com: torch.Tensor | None = field(default=None, init=False, repr=False)
    _link_inertias: torch.Tensor | None = field(default=None, init=False, repr=False)
    _inertial_overrides: dict[str, dict[str, object]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._validate_robot_cfg()
        self.validate_shapes()

    def _validate_robot_cfg(self) -> None:
        """Validate the portable model before lazily materializing tensors.

        cuRobo's original record is made by a CUDA-side loader.  Here the
        source of truth is the mutable :class:`RobotCfg`, so rejecting malformed
        names or disconnected trees at the value-model boundary is preferable
        to failing later in a planner or collision query.
        """
        required = ("links", "joints", "joint_names", "base_link", "device_cfg")
        if any(not hasattr(self.robot_cfg, name) for name in required):
            raise TypeError("robot_cfg must be a portable RobotCfg-style model")
        names = [link.name for link in self.robot_cfg.links]
        if not names or self.robot_cfg.base_link not in names:
            raise ValueError("robot_cfg must contain its base_link")
        if len(set(names)) != len(names):
            raise ValueError("robot_cfg link names must be unique")
        joint_names = list(self.robot_cfg.joint_names)
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("robot_cfg active joint names must be non-empty and unique")
        all_joint_names = [joint.name for joint in self.robot_cfg.joints]
        if len(set(all_joint_names)) != len(all_joint_names):
            raise ValueError("robot_cfg joint names must be unique")
        links = set(names)
        for joint in self.robot_cfg.joints:
            if joint.parent not in links or joint.child not in links:
                raise ValueError(f"joint {joint.name!r} references an unknown link")
            if joint.kind not in {"fixed", "revolute", "prismatic"}:
                raise ValueError(f"unsupported portable joint kind: {joint.kind!r}")
            if joint.mimic_joint is not None and joint.mimic_joint not in joint_names:
                raise ValueError(f"mimic joint {joint.name!r} has an unknown source")

    def _tree(self):
        """Return the tree in the topological order used by production FK."""
        if self._tree_cache is None:
            from curobo_metal.reference.tree_kinematics import TreeRobot

            mapping = self.robot_cfg._tree_mapping()
            # Kinematics metadata must remain available for URDFs carrying
            # visual/approximate inertias that are not positive semidefinite.
            # Production FK does not consume inertia; dynamics validates its
            # own physical model.  Keep the original values for
            # ``link_inertias`` below, but use zero inertias to build this
            # topology-only tree exactly as portable FK does.
            for link in mapping["links"]:
                link["inertial"]["inertia"] = [0.0] * 6
            self._tree_cache = TreeRobot.from_dict(mapping)
        return self._tree_cache

    @staticmethod
    def _copy_or_replace(current: torch.Tensor | None, source: torch.Tensor | None) -> torch.Tensor | None:
        if source is None:
            return None
        if (
            current is not None
            and current.shape == source.shape
            and current.device == source.device
            and current.dtype == source.dtype
        ):
            current.copy_(source)
            return current
        return source.clone()

    def _clear_derived(self) -> None:
        self._tree_cache = None
        self._fixed_transforms = None
        self._link_masses_com = None
        self._link_inertias = None

    def _v2_urdf_inertials(self) -> dict[str, tuple[tuple[float, float, float], float, tuple[float, float, float, float, float, float]]] | None:
        """Return the pinned loader's inertial representation for a URDF.

        This is intentionally not the authored URDF inertia.  The pinned V2
        loader composes an inertial origin pose with a position pose (thereby
        applying the origin translation twice) and retains its small default
        inertia buffer when reading URDF links.  Dynamics consumes that packed
        representation, so matching it here is necessary for a compatible
        public ``KinematicsParams``/``Dynamics`` lifecycle.
        """
        path = getattr(self.robot_cfg, "urdf_path", None)
        if not path:
            return None
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            return None

        def rotation(values: tuple[float, float, float]) -> torch.Tensor:
            roll, pitch, yaw = values
            cr, sr = math.cos(roll), math.sin(roll)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cy, sy = math.cos(yaw), math.sin(yaw)
            return torch.tensor(((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                                 (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                                 (-sp, cp * sr, cp * cr)), dtype=torch.float64)

        result: dict[str, tuple[tuple[float, float, float], float, tuple[float, float, float, float, float, float]]] = {}
        for link in root.findall("link"):
            name = link.get("name")
            if not name:
                continue
            inertial = link.find("inertial")
            mass = 0.01
            com = torch.zeros(3, dtype=torch.float64)
            if inertial is not None:
                mass_node = inertial.find("mass")
                if mass_node is not None:
                    try:
                        mass = max(float(mass_node.get("value", "0")), 0.01)
                    except ValueError:
                        return None
                origin = inertial.find("origin")
                if origin is not None:
                    try:
                        xyz = torch.tensor(
                            [float(value) for value in origin.get("xyz", "0 0 0").split()],
                            dtype=torch.float64,
                        )
                        rpy = tuple(float(value) for value in origin.get("rpy", "0 0 0").split())
                    except ValueError:
                        return None
                    if xyz.shape != (3,) or len(rpy) != 3:
                        return None
                    # ``Pose.from_matrix(origin).multiply(Pose(xyz, I))`` in
                    # pinned V2 yields origin.xyz + R(origin) @ origin.xyz.
                    com = xyz + rotation(rpy) @ xyz
            result[name] = (
                tuple(float(value) for value in com.tolist()), mass,
                (1.0e-4, 1.0e-4, 1.0e-4, 0.0, 0.0, 0.0),
            )
        for name, values in self._inertial_overrides.items():
            if name not in result:
                continue
            com, mass, inertia = result[name]
            result[name] = (
                values.get("com", com), values.get("mass", mass),
                values.get("inertia", inertia),
            )
        return result

    @property
    def num_dof(self) -> int:
        return len(self.robot_cfg.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return list(self.robot_cfg.joint_names)

    @property
    def non_fixed_joint_names(self) -> list[str]:
        return [
            joint.name for joint in self.robot_cfg.joints if joint.kind != "fixed"
        ]

    @property
    def base_link(self) -> str:
        return self.robot_cfg.base_link

    @property
    def device_cfg(self):
        """Device policy shared by every tensor exposed by this record."""
        return self.robot_cfg.device_cfg

    @property
    def cspace(self) -> Any:
        return self.robot_cfg.cspace

    @property
    def debug(self) -> Any:
        return self.robot_cfg.metadata.get("debug")

    @property
    def joint_limits(self):
        """Return portable tensor joint limits in the active-joint order.

        The config model stores scalar limits on the URDF joints; consumers of
        the historical ``KinematicsParams`` record expect the tensor-valued
        ``JointLimits`` view.  Constructing it here keeps mutations of the
        source robot visible and keeps CPU/MPS placement deterministic.
        """
        names = self.joint_names
        by_name = {joint.name: joint for joint in self.robot_cfg.joints}
        joints = [by_name[name] for name in names]

        def bounds(values: list[tuple[float, float]]) -> torch.Tensor:
            return self.device_cfg.to_device(values).transpose(0, 1).contiguous()

        return JointLimits(
            names,
            bounds([(joint.limits.lower, joint.limits.upper) for joint in joints]),
            bounds([(-joint.limits.velocity, joint.limits.velocity) for joint in joints]),
            bounds([(-10.0, 10.0) for _ in joints]),
            bounds([(-500.0, 500.0) for _ in joints]),
            bounds([(-joint.limits.effort, joint.limits.effort) for joint in joints]),
            self.device_cfg,
        )

    @property
    def total_spheres(self) -> int:
        return len(self.robot_cfg.collision_spheres)

    @property
    def link_spheres(self) -> torch.Tensor:
        if self._link_spheres is None:
            self._link_spheres = self.robot_cfg.device_cfg.to_device([
                [*sphere.center, sphere.radius]
                for sphere in self.robot_cfg.collision_spheres
            ]).reshape(1, -1, 4)
            self.reference_link_spheres = self._link_spheres.clone()
        return self._link_spheres

    @property
    def link_sphere_idx_map(self) -> torch.Tensor:
        mapping = self.link_name_to_idx_map
        return torch.tensor(
            [mapping[sphere.link_name] for sphere in self.robot_cfg.collision_spheres],
            dtype=torch.int64,
            device=self.robot_cfg.device_cfg.device,
        )

    @property
    def link_name_to_idx_map(self) -> dict[str, int]:
        return {link.name: index for index, link in enumerate(self._tree().links)}

    @property
    def link_map(self) -> torch.Tensor:
        """Parent-link index for each topologically ordered link.

        Root is ``-1``.  This is a genuine tree map, not an identity map: it
        is safe for branching robots and is suitable for portable metadata
        consumers even though raw CUDA packed-FK launchers remain unavailable.
        """
        return torch.tensor(
            [link.parent for link in self._tree().links],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )

    @property
    def joint_map(self) -> torch.Tensor:
        """Active-joint index map in the source joint-record order.

        The public portable loader historically exposed this one entry per
        source joint (rather than including the synthetic root link), so that
        rank is retained here.  Mimics resolve to their source active joint.
        """
        active = {name: index for index, name in enumerate(self.joint_names)}
        return torch.tensor(
            [
                active.get(
                    joint.mimic_joint if joint.mimic_joint is not None else joint.name,
                    -1,
                )
                for joint in self.robot_cfg.joints
            ],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )

    @property
    def joint_map_type(self) -> torch.Tensor:
        kinds = {"fixed": -1, "prismatic": 0, "revolute": 1}
        return torch.tensor(
            [kinds.get(joint.kind, -2) for joint in self.robot_cfg.joints],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )

    @property
    def joint_offset_map(self) -> torch.Tensor:
        return self.device_cfg.to_device(
            [[joint.mimic_multiplier, joint.mimic_offset] for joint in self.robot_cfg.joints]
        ).reshape(-1, 2)

    @property
    def mimic_joints(self) -> dict[str, tuple[str, float, float]]:
        return {
            joint.name: (joint.mimic_joint, joint.mimic_multiplier, joint.mimic_offset)
            for joint in self.robot_cfg.joints
            if joint.mimic_joint is not None
        }

    @property
    def tool_frame_map(self) -> torch.Tensor:
        indices = self.link_name_to_idx_map
        return torch.tensor(
            [indices[name] for name in self.tool_frames],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )

    @property
    def fixed_transforms(self) -> torch.Tensor:
        """[link, 4, 4] rest transforms, useful for portable introspection."""
        if self._fixed_transforms is None:
            self._fixed_transforms = torch.stack([
                torch.as_tensor(link.origin, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
                for link in self._tree().links
            ]).contiguous()
        return self._fixed_transforms

    @property
    def link_masses_com(self) -> torch.Tensor:
        """Packed ``[link, xyz-com, mass]`` inertial metadata."""
        if self._link_masses_com is None:
            raw = {link.name: link for link in self.robot_cfg.links}
            v2 = self._v2_urdf_inertials()
            self._link_masses_com = self.device_cfg.to_device(
                [
                    [*(v2[link.name][0] if v2 is not None and link.name in v2 else raw[link.name].com),
                     v2[link.name][1] if v2 is not None and link.name in v2 else raw[link.name].mass]
                    for link in self._tree().links
                ]
            ).reshape(-1, 4).contiguous()
        return self._link_masses_com

    @property
    def link_inertias(self) -> torch.Tensor:
        """CUDA-shaped ``[link, 8]`` inertia packing for portable consumers.

        The final two values are zero padding, retained for source
        compatibility.  Calling a raw CUDA RNEA/FK ABI with this record is
        still intentionally unsupported.
        """
        if self._link_inertias is None:
            raw = {link.name: link for link in self.robot_cfg.links}
            v2 = self._v2_urdf_inertials()
            values = []
            for link in self._tree().links:
                packed = (
                    v2[link.name][2]
                    if v2 is not None and link.name in v2
                    else raw[link.name].inertia
                )
                values.append([
                    packed[0], packed[1], packed[2], packed[3], packed[4], packed[5], 0.0, 0.0,
                ])
            self._link_inertias = self.device_cfg.to_device(values).reshape(-1, 8).contiguous()
        return self._link_inertias

    @property
    def grasp_contact_link_names(self) -> list[str]:
        value = self.robot_cfg.metadata.get("grasp_contact_link_names", [])
        if value is None:
            return []
        result = list(value)
        unknown = sorted(set(result) - set(self.link_name_to_idx_map))
        if unknown:
            raise ValueError(f"grasp_contact_link_names contain unknown links: {unknown}")
        return result

    @property
    def tool_frames(self) -> list[str]:
        return list(self.robot_cfg.tool_frames)

    @property
    def all_link_names(self) -> list[str]:
        """All tree links in the same topological order as production FK."""
        return [link.name for link in self._tree().links]

    @property
    def link_chain_data(self) -> torch.Tensor:
        """CSR payload of ancestor link indices for every output link.

        This is useful for Jacobian/introspection code.  It deliberately does
        not claim compatibility with CUDA packed-buffer launch semantics.
        """
        data: list[int] = []
        for index, link in enumerate(self._tree().links):
            chain: list[int] = []
            current = index
            while current >= 0:
                chain.append(current)
                current = self._tree().links[current].parent
            data.extend(reversed(chain))
        return torch.tensor(data, dtype=torch.int64, device=self.device_cfg.device)

    @property
    def link_chain_offsets(self) -> torch.Tensor:
        offsets = [0]
        for index, link in enumerate(self._tree().links):
            length = 1
            current = link.parent
            while current >= 0:
                length += 1
                current = self._tree().links[current].parent
            offsets.append(offsets[-1] + length)
        return torch.tensor(offsets, dtype=torch.int64, device=self.device_cfg.device)

    @property
    def joint_links_data(self) -> torch.Tensor:
        """CSR payload of output links affected by each active joint."""
        data: list[int] = []
        for joint_index in range(self.num_dof):
            for index, link in enumerate(self._tree().links):
                current = index
                while current >= 0:
                    ancestor = self._tree().links[current]
                    if ancestor.q_index == joint_index:
                        data.append(index)
                        break
                    current = ancestor.parent
        return torch.tensor(data, dtype=torch.int64, device=self.device_cfg.device)

    @property
    def joint_links_offsets(self) -> torch.Tensor:
        offsets = [0]
        for joint_index in range(self.num_dof):
            count = 0
            for index, link in enumerate(self._tree().links):
                current = index
                while current >= 0:
                    ancestor = self._tree().links[current]
                    if ancestor.q_index == joint_index:
                        count += 1
                        break
                    current = ancestor.parent
            offsets.append(offsets[-1] + count)
        return torch.tensor(offsets, dtype=torch.int64, device=self.device_cfg.device)

    @property
    def joint_affects_endeffector(self) -> torch.Tensor:
        """Flattened active-joint × tool-frame reachability matrix."""
        values = []
        indices = self.link_name_to_idx_map
        for joint_index in range(self.num_dof):
            for name in self.tool_frames:
                current = indices[name]
                affects = False
                while current >= 0:
                    link = self._tree().links[current]
                    if link.q_index == joint_index:
                        affects = True
                        break
                    current = link.parent
                values.append(affects)
        return torch.tensor(values, dtype=torch.bool, device=self.device_cfg.device)

    @property
    def mesh_link_names(self) -> list[str]:
        return list(self.robot_cfg.metadata.get("mesh_link_names", []))

    @property
    def lock_jointstate(self) -> JointState:
        locked = self.robot_cfg.metadata.get("lock_joints", {})
        return JointState.from_position(
            self.robot_cfg.device_cfg.to_device(list(locked.values())),
            joint_names=list(locked),
        )

    def make_contiguous(self) -> None:
        """Make materialized portable tensor metadata contiguous in place."""
        for name in (
            "_link_spheres", "reference_link_spheres", "_fixed_transforms",
            "_link_masses_com", "_link_inertias",
        ):
            value = getattr(self, name)
            if value is not None and not value.is_contiguous():
                setattr(self, name, value.contiguous())

    def copy_(self, other: "KinematicsParams") -> "KinematicsParams":
        if not isinstance(other, KinematicsParams):
            raise TypeError("other must be KinematicsParams")
        if self.device_cfg != other.device_cfg:
            raise ValueError("copy_ requires matching device_cfg values")
        # The mutable source model does not expose an in-place record-copy API.
        # Deep-copying it prevents a clone/update of one KinematicsParams from
        # altering another, while cached tensor buffers retain identity whenever
        # their shapes remain compatible.
        self.robot_cfg = deepcopy(other.robot_cfg)
        self._tree_cache = None
        self._link_spheres = self._copy_or_replace(self._link_spheres, other._link_spheres)
        self.reference_link_spheres = self._copy_or_replace(
            self.reference_link_spheres, other.reference_link_spheres
        )
        self._fixed_transforms = self._copy_or_replace(
            self._fixed_transforms, other._fixed_transforms
        )
        self._link_masses_com = self._copy_or_replace(
            self._link_masses_com, other._link_masses_com
        )
        self._link_inertias = self._copy_or_replace(
            self._link_inertias, other._link_inertias
        )
        self.validate_shapes()
        return self

    def clone(self) -> "KinematicsParams":
        result = KinematicsParams(deepcopy(self.robot_cfg))
        result._link_spheres = self._copy_or_replace(None, self._link_spheres)
        result.reference_link_spheres = self._copy_or_replace(None, self.reference_link_spheres)
        result._fixed_transforms = self._copy_or_replace(None, self._fixed_transforms)
        result._link_masses_com = self._copy_or_replace(None, self._link_masses_com)
        result._link_inertias = self._copy_or_replace(None, self._link_inertias)
        return result

    def to(self, device_cfg: Any = None, *, device: torch.device | str | None = None,
           dtype: torch.dtype | None = None) -> "KinematicsParams":
        """Return an independent copy on a new portable device policy.

        The operation moves value-model tensors only.  It does not expose the
        CUDA FK/RNEA ABI or transform this record into a raw CUDA buffer.
        """
        from curobo._src.types.device_cfg import DeviceCfg

        if isinstance(device_cfg, DeviceCfg):
            target = device_cfg
        else:
            target_device = device if device is not None else device_cfg
            if target_device is None:
                target_device = self.device_cfg.device
            target = DeviceCfg(torch.device(target_device), self.device_cfg.dtype if dtype is None else dtype)
        result = self.clone()
        result.robot_cfg.device_cfg = target
        for name in (
            "_link_spheres", "reference_link_spheres", "_fixed_transforms",
            "_link_masses_com", "_link_inertias",
        ):
            value = getattr(result, name)
            if value is not None:
                setattr(result, name, value.to(**target.as_torch_dict()))
        return result

    def validate_shapes(self) -> None:
        if self.num_dof != len(self.joint_names) or self.num_dof <= 0:
            raise ValueError("num_dof and joint_names disagree")
        if len(set(self.tool_frames)) != len(self.tool_frames):
            raise ValueError("tool_frames must be unique")
        unknown_tools = sorted(set(self.tool_frames) - set(self.all_link_names))
        if unknown_tools:
            raise ValueError(f"tool_frames contain unknown links: {unknown_tools}")
        if self.link_spheres.ndim != 3 or self.link_spheres.shape[-1] != 4:
            raise ValueError("link_spheres must have shape [env, sphere, 4]")
        if self.link_spheres.shape[0] < 1:
            raise ValueError("link_spheres must contain at least one environment")
        if self.link_spheres.shape[1] != self.total_spheres:
            raise ValueError("link_spheres sphere count does not match collision geometry")
        if not self.device_cfg.is_same_torch_device(self.link_spheres.device):
            raise ValueError("link_spheres must be on device_cfg.device")
        if not bool(torch.isfinite(self.link_spheres).all().item()):
            raise ValueError("link_spheres must contain finite values")
        if self.link_sphere_idx_map.numel() != self.total_spheres:
            raise ValueError("link_sphere_idx_map must have one index per sphere")
        if not self.device_cfg.is_same_torch_device(self.link_sphere_idx_map.device):
            raise ValueError("link_sphere_idx_map must be on device_cfg.device")
        if self.link_sphere_idx_map.numel() and bool(
            ((self.link_sphere_idx_map < 0) | (self.link_sphere_idx_map >= self.num_links)).any().item()
        ):
            raise ValueError("link_sphere_idx_map contains an invalid link index")
        if self.fixed_transforms.shape != (self.num_links, 4, 4):
            raise ValueError("fixed_transforms must have shape [num_links, 4, 4]")
        if self.link_masses_com.shape != (self.num_links, 4):
            raise ValueError("link_masses_com must have shape [num_links, 4]")
        if self.link_inertias.shape != (self.num_links, 8):
            raise ValueError("link_inertias must have shape [num_links, 8]")

    def load_cspace_cfg_from_kinematics(self) -> None:
        """Complete an omitted portable C-space record from joint limits.

        The source loader creates a centered retract configuration and unit
        cost weights when a robot file contains only URDF joint limits.  The
        portable value model stores those values as Python lists so it can be
        serialized without CUDA; derive the same values before any solver
        reads the configuration.  Existing user-authored values are never
        replaced.
        """
        cspace = self.robot_cfg.cspace
        names = self.joint_names
        if not cspace.joint_names:
            cspace.joint_names = names.copy()
        elif list(cspace.joint_names) != names:
            raise ValueError("cspace joint_names must match active joint_names")

        limits = self.joint_limits
        lower, upper = limits.position[0], limits.position[1]
        # Unbounded URDF joints cannot have a meaningful midpoint.  Zero is
        # the conservative portable default and agrees with robot loaders
        # that omit a retract configuration.
        midpoint = torch.where(
            torch.isfinite(lower) & torch.isfinite(upper),
            (lower + upper) / 2,
            torch.zeros_like(lower),
        ).detach().cpu().tolist()
        if not cspace.default_joint_position:
            cspace.default_joint_position = midpoint
        if cspace.cspace_distance_weight is None:
            cspace.cspace_distance_weight = [1.0] * self.num_dof
        if cspace.null_space_weight is None:
            cspace.null_space_weight = [1.0] * self.num_dof
        if cspace.max_acceleration is None:
            cspace.max_acceleration = [10.0] * self.num_dof
        if cspace.max_jerk is None:
            cspace.max_jerk = [500.0] * self.num_dof

    def get_sphere_index_from_link_name(self, link_name: str) -> torch.Tensor:
        if link_name not in self.link_name_to_idx_map:
            raise ValueError(f"unknown link: {link_name}")
        values = [
            index for index, sphere in enumerate(self.robot_cfg.collision_spheres)
            if sphere.link_name == link_name
        ]
        if not values:
            return torch.empty(0, dtype=torch.int64, device=self.robot_cfg.device_cfg.device)
        return torch.tensor(values, dtype=torch.int64, device=self.robot_cfg.device_cfg.device)

    def update_link_spheres(
        self,
        link_name: str,
        sphere_position_radius: torch.Tensor,
        start_sph_idx: int = 0,
        config_idx: int | None = None,
    ) -> None:
        indices = self.get_sphere_index_from_link_name(link_name)
        if indices.numel() == 0:
            raise ValueError(f"link {link_name!r} has no collision spheres")
        if start_sph_idx < 0:
            raise ValueError("start_sph_idx must be non-negative")
        values = torch.as_tensor(
            sphere_position_radius,
            device=self.link_spheres.device,
            dtype=self.link_spheres.dtype,
        )
        if values.ndim not in (2, 3) or values.shape[-1] != 4:
            raise ValueError("sphere_position_radius must have shape [sphere, 4] or [env, sphere, 4]")
        count = values.shape[-2]
        target = indices[start_sph_idx:start_sph_idx + count]
        if target.numel() != count:
            raise ValueError("too many sphere values for link")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("sphere_position_radius must contain finite values")
        if config_idx is None:
            if values.ndim == 2:
                self.link_spheres[:, target] = values
            elif values.shape[0] == self.num_envs:
                self.link_spheres[:, target] = values
            else:
                raise ValueError("batched sphere values must match num_envs")
        else:
            if not 0 <= config_idx < self.num_envs:
                raise IndexError("config_idx is out of range")
            if values.ndim == 3:
                if values.shape[0] != 1:
                    raise ValueError("config_idx accepts one sphere configuration")
                values = values[0]
            self.link_spheres[config_idx, target] = values

    def set_num_envs(self, num_envs: int) -> None:
        """Resize the collision-sphere configuration bank deterministically.

        Expanding repeats the reference configuration.  Shrinking keeps the
        leading configurations, matching the environment-index convention used
        by portable FK and collision dispatch.
        """
        if not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        current = self.link_spheres
        if num_envs == current.shape[0]:
            return
        reference = self.reference_link_spheres
        if num_envs < current.shape[0]:
            self._link_spheres = current[:num_envs].clone()
            self.reference_link_spheres = reference[:num_envs].clone()
            return
        extra = num_envs - current.shape[0]
        self._link_spheres = torch.cat((current, reference[:1].expand(extra, -1, -1).clone()), dim=0)
        self.reference_link_spheres = torch.cat(
            (reference, reference[:1].expand(extra, -1, -1).clone()), dim=0
        )

    def get_link_spheres(self, link_name: str, config_idx: int = 0) -> torch.Tensor:
        return self.link_spheres[config_idx, self.get_sphere_index_from_link_name(link_name)]

    def get_reference_link_spheres(self, link_name: str, config_idx: int = 0) -> torch.Tensor:
        self.link_spheres
        return self.reference_link_spheres[
            config_idx, self.get_sphere_index_from_link_name(link_name)
        ]

    def get_number_of_spheres(self, link_name: str) -> int:
        return int(self.get_sphere_index_from_link_name(link_name).numel())

    def disable_link_spheres(self, link_name: str) -> None:
        indices = self.get_sphere_index_from_link_name(link_name)
        # V2 uses a fixed negative sentinel rather than merely flipping the
        # current radius.  Retaining it makes repeated disable/update/enable
        # cycles deterministic and allows existing collision filters to test
        # the same convention on CPU and MPS.
        self.link_spheres[:, indices, 3] = -100.0

    def enable_link_spheres(self, link_name: str) -> None:
        indices = self.get_sphere_index_from_link_name(link_name)
        self.link_spheres[:, indices, 3] = self.reference_link_spheres[:, indices, 3]

    def reset_link_spheres(self, link_name: str) -> None:
        indices = self.get_sphere_index_from_link_name(link_name)
        self.link_spheres[:, indices] = self.reference_link_spheres[:, indices]

    def get_link_masses_com(self, link_name: str) -> torch.Tensor:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        return self.robot_cfg.device_cfg.to_device([*link.com, link.mass])

    def update_link_mass(self, link_name: str, mass: float) -> None:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        if not torch.isfinite(torch.tensor(mass)) or mass < 0:
            raise ValueError("mass must be finite and non-negative")
        link.mass = float(mass)
        self._inertial_overrides.setdefault(link_name, {})["mass"] = float(mass)
        self._link_masses_com = None
        self._tree_cache = None

    def update_link_com(self, link_name: str, com: torch.Tensor) -> None:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        values = torch.as_tensor(com).reshape(-1)
        if values.numel() != 3:
            raise ValueError("com must have three values")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("com must contain finite values")
        link.com = tuple(values.cpu().tolist())
        self._inertial_overrides.setdefault(link_name, {})["com"] = link.com
        self._link_masses_com = None
        self._tree_cache = None

    def update_link_inertia(self, link_name: str, inertia: torch.Tensor) -> None:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        values = torch.as_tensor(inertia).reshape(-1)
        if values.numel() not in (6, 9):
            raise ValueError("inertia must have six values or a 3x3 matrix")
        if values.numel() == 9:
            matrix = values.reshape(3, 3)
            values = matrix[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("inertia must contain finite values")
        link.inertia = tuple(values.cpu().tolist())
        self._inertial_overrides.setdefault(link_name, {})["inertia"] = link.inertia
        self._link_inertias = None
        self._tree_cache = None

    def get_link_inertia(self, link_name: str) -> torch.Tensor:
        link = next((x for x in self.robot_cfg.links if x.name == link_name), None)
        if link is None:
            raise ValueError(f"unknown link: {link_name}")
        return self.robot_cfg.device_cfg.to_device(link.inertia)

    def get_robot_collision_geometry(self):
        from .collision_geometry import RobotCollisionGeometry
        return RobotCollisionGeometry(self.link_sphere_idx_map.clone(), self.num_links)

    @property
    def num_pose_links(self) -> int:
        return len(self.tool_frames)

    @property
    def num_links(self) -> int:
        return len(self.robot_cfg.links)

    @property
    def num_spheres(self) -> int:
        return self.total_spheres

    @property
    def num_envs(self) -> int:
        return self.link_spheres.shape[0]

    @property
    def n_tree_levels(self) -> int:
        return int(self.link_level_offsets.numel() - 1)

    @property
    def max_level_width(self) -> int:
        offsets = self.link_level_offsets
        return int((offsets[1:] - offsets[:-1]).max().item()) if offsets.numel() > 1 else 0

    @property
    def link_level_data(self) -> torch.Tensor:
        levels: list[list[int]] = []
        for index, link in enumerate(self._tree().links):
            depth = 0
            current = link.parent
            while current >= 0:
                depth += 1
                current = self._tree().links[current].parent
            while len(levels) <= depth:
                levels.append([])
            levels[depth].append(index)
        return torch.tensor(
            [index for level in levels for index in level],
            dtype=torch.int64,
            device=self.device_cfg.device,
        )

    @property
    def link_level_offsets(self) -> torch.Tensor:
        offsets = [0]
        count = 0
        for depth in range(self.num_links):
            level_count = 0
            for link in self._tree().links:
                current, actual = link.parent, 0
                while current >= 0:
                    actual += 1
                    current = self._tree().links[current].parent
                if actual == depth:
                    level_count += 1
            if level_count:
                count += level_count
                offsets.append(count)
        return torch.tensor(offsets, dtype=torch.int64, device=self.device_cfg.device)

    def export_to_urdf(
        self,
        robot_name: str = "robot",
        output_path: str | None = None,
        include_spheres: bool = False,
        kinematics_parser=None,
    ) -> str:
        """Serialize the current portable robot value model as URDF XML.

        This deliberately exports mutable mass, inertia, joint, and sphere
        state instead of copying ``robot_cfg.urdf_path``.  It therefore works
        for YAML-created models and for models changed after loading.  The
        return value is XML text (and the same text is written when
        ``output_path`` is supplied); construction of a ``yourdfpy`` object,
        visual meshes, USD, and Isaac geometry remain optional external
        integrations rather than hidden dependencies of the Metal package.
        ``kinematics_parser`` is accepted for the pinned signature but is not
        needed for portable structural export.
        """
        del kinematics_parser
        if not isinstance(robot_name, str) or not robot_name:
            raise ValueError("robot_name must be a non-empty string")

        def number(value: object, label: str) -> str:
            scalar = float(value)
            if not torch.isfinite(torch.tensor(scalar)):
                raise ValueError(f"{label} must be finite for URDF export")
            return format(scalar, ".17g")

        def vector(values: object, size: int, label: str) -> str:
            items = list(values)
            if len(items) != size:
                raise ValueError(f"{label} must contain {size} values")
            return " ".join(number(value, label) for value in items)

        root = ET.Element("robot", {"name": robot_name})
        spheres_by_link: dict[str, list[torch.Tensor]] = {}
        if include_spheres:
            spheres = self.link_spheres[0]
            for index, sphere in enumerate(self.robot_cfg.collision_spheres):
                value = spheres[index]
                # Negative radii are the canonical disabled-sphere sentinel.
                # A URDF sphere cannot represent that state, so omit it.
                if float(value[3].detach().cpu()) > 0.0:
                    spheres_by_link.setdefault(sphere.link_name, []).append(value)

        for link in self.robot_cfg.links:
            element = ET.SubElement(root, "link", {"name": link.name})
            inertial = ET.SubElement(element, "inertial")
            ET.SubElement(inertial, "origin", {"xyz": vector(link.com, 3, "link com"), "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", {"value": number(link.mass, "link mass")})
            ixx, iyy, izz, ixy, ixz, iyz = link.inertia
            ET.SubElement(inertial, "inertia", {
                "ixx": number(ixx, "link inertia"), "iyy": number(iyy, "link inertia"),
                "izz": number(izz, "link inertia"), "ixy": number(ixy, "link inertia"),
                "ixz": number(ixz, "link inertia"), "iyz": number(iyz, "link inertia"),
            })
            for sphere in spheres_by_link.get(link.name, []):
                collision = ET.SubElement(element, "collision")
                ET.SubElement(collision, "origin", {
                    "xyz": vector(sphere[:3].detach().cpu().tolist(), 3, "sphere center"),
                    "rpy": "0 0 0",
                })
                geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(geometry, "sphere", {"radius": number(sphere[3].detach().cpu(), "sphere radius")})

        for joint in self.robot_cfg.joints:
            joint_type = joint.kind
            if joint_type not in {"fixed", "revolute", "prismatic"}:
                raise NotImplementedError(f"URDF export does not support joint kind {joint_type!r}")
            element = ET.SubElement(root, "joint", {"name": joint.name, "type": joint_type})
            ET.SubElement(element, "parent", {"link": joint.parent})
            ET.SubElement(element, "child", {"link": joint.child})
            ET.SubElement(element, "origin", {
                "xyz": vector(joint.xyz, 3, "joint origin xyz"),
                "rpy": vector(joint.rpy, 3, "joint origin rpy"),
            })
            if joint_type != "fixed":
                ET.SubElement(element, "axis", {"xyz": vector(joint.axis, 3, "joint axis")})
                values = (joint.limits.lower, joint.limits.upper, joint.limits.effort, joint.limits.velocity)
                if all(torch.isfinite(torch.tensor(float(value))) for value in values):
                    ET.SubElement(element, "limit", {
                        "lower": number(values[0], "joint lower limit"),
                        "upper": number(values[1], "joint upper limit"),
                        "effort": number(values[2], "joint effort limit"),
                        "velocity": number(values[3], "joint velocity limit"),
                    })
            if joint.mimic_joint is not None:
                ET.SubElement(element, "mimic", {
                    "joint": joint.mimic_joint,
                    "multiplier": number(joint.mimic_multiplier, "mimic multiplier"),
                    "offset": number(joint.mimic_offset, "mimic offset"),
                })

        value = ET.tostring(root, encoding="unicode", xml_declaration=True)
        if output_path is not None:
            Path(output_path).write_text(value, encoding="utf-8")
        return value


__all__ = [
    "Any", "CSpaceParams", "DeviceCfg", "Dict", "JointLimits", "JointState",
    "JointType", "KinematicsParams", "List", "Optional", "RobotCollisionGeometry",
]
