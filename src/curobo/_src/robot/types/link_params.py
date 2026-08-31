"""Kinematic link records compatible with the pinned cuRoboV2 surface.

``LinkParams`` is deliberately a NumPy value record: it describes static
robot topology and is consumed by the loader before tensor kernels run.  The
``Pose`` and ``DeviceCfg`` imports below are public upstream re-exports.  In
particular, importing this module must not initialize CUDA, Warp, or Isaac;
the supplied pose conversion works on both CPU and Apple Metal before the
result is materialized as host-side kinematic metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .joint_types import JointType
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


@dataclass
class LinkParams:
    link_name: str
    joint_name: str
    joint_type: JointType
    fixed_transform: np.ndarray
    parent_link_name: Optional[str] = None
    child_link_name: Optional[str] = None
    joint_limits: Optional[List[float]] = None
    joint_axis: Optional[np.ndarray] = None
    joint_id: Optional[int] = None
    joint_velocity_limits: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    joint_offset: List[float] = field(default_factory=lambda: [1.0, 0.0])
    mimic_joint_name: Optional[str] = None
    joint_effort_limit: List[float] = field(default_factory=lambda: [10000.0])
    link_mass: float = 0.01
    link_com: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    link_inertia: np.ndarray = field(
        default_factory=lambda: np.array([1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0])
    )

    @staticmethod
    def create(dict_data: Dict[str, Any]) -> LinkParams:
        """Create a link record from the pinned YAML-style representation.

        Upstream configurations encode ``fixed_transform`` as a seven-value
        ``[x, y, z, qw, qx, qy, qz]`` pose.  We retain that exact conversion
        and additionally accept an already-materialized ``[3, 4]`` affine
        transform, which is the native result of this method and is useful
        for portable YAML round trips.  The source mapping is copied, so a
        loader can safely reuse its parsed configuration after compilation.
        """
        if not isinstance(dict_data, dict):
            raise TypeError("LinkParams.create expects a dictionary")
        data = dict(dict_data)
        try:
            joint_type = data["joint_type"]
            data["joint_type"] = (
                joint_type if isinstance(joint_type, JointType) else JointType[joint_type]
            )
        except KeyError as error:
            raise KeyError("LinkParams data must contain a valid joint_type") from error
        if "fixed_transform" not in data:
            raise KeyError("LinkParams data is missing fixed_transform")
        transform = np.asarray(data["fixed_transform"], dtype=float)
        if transform.size == 7:
            # The pinned implementation reshapes its historical pose helper
            # through a ``4x4`` matrix.  Portable ``Pose`` exposes the same
            # affine information directly as ``[batch, 3, 4]``; selecting the
            # singleton batch avoids manufacturing a homogeneous row while
            # preserving the public LinkParams ``[3, 4]`` layout.
            transform = Pose.from_list(transform.tolist()).get_numpy_affine_matrix().reshape(
                -1, 3, 4
            )
            if transform.shape[0] != 1:
                log_and_raise("LinkParams.create expects exactly one fixed-transform pose")
            transform = transform[0]
        if transform.shape != (3, 4):
            log_and_raise(
                "fixed_transform must be a seven-value pose or a (3, 4) affine matrix; "
                f"got shape {transform.shape}"
            )
        data["fixed_transform"] = transform.copy()
        return LinkParams(**data)

    def __post_init__(self) -> None:
        self.fixed_transform = np.asarray(self.fixed_transform, dtype=float)
        self.link_com = np.asarray(self.link_com, dtype=float)
        self.link_inertia = np.asarray(self.link_inertia, dtype=float)
        if self.fixed_transform.shape != (3, 4):
            log_and_raise(
                f"fixed_transform shape does not match: {self.fixed_transform.shape} != (3, 4)"
            )

    def get_link_com_and_mass(self) -> np.ndarray:
        """Return ``[com_x, com_y, com_z, mass]`` in the pinned order."""
        if self.link_com.shape != (3,):
            log_and_raise(f"link_com shape does not match: {self.link_com.shape} != (3,)")
        return np.concatenate([self.link_com, [self.link_mass]])


__all__ = ["DeviceCfg", "JointType", "LinkParams", "Pose", "log_and_raise"]
