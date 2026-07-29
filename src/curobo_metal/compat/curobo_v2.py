"""Convert the tensor portion of a cuRoboV2 kinematics config.

This module intentionally uses structural typing instead of importing cuRobo or
PyTorch.  It is the boundary between cuRobo's generated ``KinematicsCfg`` /
``KinematicsParams`` objects and future portable FK and collision operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

CUROBO_V2_REVISION = "8e734f3ced1df898990bcd92de40abce475907db"

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


class UnsupportedCuroboConfig(ValueError):
    """The supplied cuRobo configuration is outside the supported seam."""


@dataclass(frozen=True)
class BackendRobotConfig:
    """Backend-neutral FK and sphere-collision inputs.

    Arrays own CPU memory and are C-contiguous.  A backend is expected to move
    them to its device once, outside the timed execution path.
    """

    fixed_transforms: FloatArray  # [L, 4, 4], parent-to-link
    parent_link: IntArray  # [L]
    joint_index: IntArray  # [L], -1 for fixed links
    joint_type: IntArray  # [L], pinned cuRobo JointType integer values
    joint_offset: FloatArray  # [L, 2], multiplier and additive offset
    tool_link: IntArray  # [T]
    link_spheres: FloatArray | None  # [E, S, 4], xyz and radius
    sphere_link: IntArray | None  # [S]
    joint_names: tuple[str, ...]
    tool_frames: tuple[str, ...]
    link_names: tuple[str, ...]
    base_link: str
    num_dof: int
    upstream_revision: str = CUROBO_V2_REVISION


def _field(value: Any, name: str, default: Any = ...) -> Any:
    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is not ...:
        return default
    raise UnsupportedCuroboConfig(f"missing cuRoboV2 field: {name}")


def _numpy(value: Any, name: str, dtype: np.dtype[Any]) -> NDArray[Any]:
    """Copy an ndarray or CPU-convertible tensor without importing torch."""
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    try:
        array = np.asarray(candidate, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise UnsupportedCuroboConfig(f"{name} is not array-convertible") from error
    return np.array(array, dtype=dtype, order="C", copy=True)


def _link_names(params: Any, links: int, base_link: str) -> tuple[str, ...]:
    raw = _field(params, "link_name_to_idx_map", None)
    if raw is None:
        return tuple(base_link if index == 0 else f"link_{index}" for index in range(links))
    names: list[str | None] = [None] * links
    for name, index_value in raw.items():
        index = int(index_value)
        if index < 0 or index >= links or names[index] is not None:
            raise UnsupportedCuroboConfig("link_name_to_idx_map is not one-to-one")
        names[index] = str(name)
    if any(name is None for name in names):
        raise UnsupportedCuroboConfig("link_name_to_idx_map does not name every link")
    return tuple(name for name in names if name is not None)


def convert_kinematics_config(config: Any) -> BackendRobotConfig:
    """Convert a pinned cuRoboV2 ``KinematicsCfg`` or ``KinematicsParams``.

    A JSON-like mapping with the same public field names is also accepted,
    which permits deterministic tests and serialized CUDA-to-Mac handoff.
    Only a single serial chain is accepted in this initial seam.
    """
    params = _field(config, "kinematics_config", config)
    fixed = _numpy(_field(params, "fixed_transforms"), "fixed_transforms", np.dtype("<f8"))
    if fixed.ndim != 3 or fixed.shape[1:] not in {(3, 4), (4, 4)}:
        raise UnsupportedCuroboConfig("fixed_transforms must have shape [L,3,4] or [L,4,4]")
    if fixed.shape[1:] == (3, 4):
        homogeneous = np.broadcast_to(np.eye(4), (fixed.shape[0], 4, 4)).copy()
        homogeneous[:, :3] = fixed
        fixed = homogeneous
    if not np.all(np.isfinite(fixed)):
        raise UnsupportedCuroboConfig("fixed_transforms must be finite")

    links = fixed.shape[0]
    parent = _numpy(_field(params, "link_map"), "link_map", np.dtype("<i8")).reshape(-1)
    joint_index = _numpy(_field(params, "joint_map"), "joint_map", np.dtype("<i8")).reshape(-1)
    joint_type = _numpy(
        _field(params, "joint_map_type"), "joint_map_type", np.dtype("i1")
    ).reshape(-1)
    if any(array.shape != (links,) for array in (parent, joint_index, joint_type)):
        raise UnsupportedCuroboConfig("link_map, joint_map, and joint_map_type must have shape [L]")
    if links < 1 or parent[0] not in (-1, 0) or not np.array_equal(
        parent[1:], np.arange(links - 1)
    ):
        raise UnsupportedCuroboConfig("only a topologically ordered serial chain is supported")
    if np.any((joint_type < -1) | (joint_type > 11)):
        raise UnsupportedCuroboConfig("joint_map_type contains an unknown pinned JointType value")

    num_dof = int(_field(params, "num_dof"))
    movable = joint_type != -1
    if np.any(joint_index[~movable] != -1):
        raise UnsupportedCuroboConfig("fixed links must use joint index -1")
    if np.any(joint_index[movable] < 0) or not np.array_equal(
        np.sort(joint_index[movable]), np.arange(num_dof)
    ):
        raise UnsupportedCuroboConfig("movable joint indices must cover [0, num_dof)")

    offset_flat = _numpy(
        _field(params, "joint_offset_map"), "joint_offset_map", np.dtype("<f8")
    ).reshape(-1)
    if offset_flat.size != links * 2 or not np.all(np.isfinite(offset_flat)):
        raise UnsupportedCuroboConfig("joint_offset_map must contain two finite values per link")
    offset = offset_flat.reshape(links, 2)

    tool_link = _numpy(
        _field(params, "tool_frame_map"), "tool_frame_map", np.dtype("<i8")
    ).reshape(-1)
    if np.any((tool_link < 0) | (tool_link >= links)):
        raise UnsupportedCuroboConfig("tool_frame_map contains an out-of-range link")
    tool_frames = tuple(str(item) for item in _field(params, "tool_frames"))
    if len(tool_frames) != tool_link.size:
        raise UnsupportedCuroboConfig("tool_frames and tool_frame_map lengths differ")

    spheres_raw = _field(params, "link_spheres", None)
    sphere_map_raw = _field(params, "link_sphere_idx_map", None)
    if (spheres_raw is None) != (sphere_map_raw is None):
        raise UnsupportedCuroboConfig("link_spheres and link_sphere_idx_map must appear together")
    spheres: FloatArray | None = None
    sphere_link: IntArray | None = None
    if spheres_raw is not None:
        spheres = _numpy(spheres_raw, "link_spheres", np.dtype("<f8"))
        if spheres.ndim == 2:
            spheres = spheres[None, ...]
        sphere_link = _numpy(sphere_map_raw, "link_sphere_idx_map", np.dtype("<i8")).reshape(-1)
        if spheres.ndim != 3 or spheres.shape[2] != 4 or sphere_link.shape != (
            spheres.shape[1],
        ):
            raise UnsupportedCuroboConfig(
                "link_spheres must be [E,S,4] and link_sphere_idx_map must be [S]"
            )
        if not np.all(np.isfinite(spheres)) or np.any(
            (sphere_link < 0) | (sphere_link >= links)
        ):
            raise UnsupportedCuroboConfig("collision sphere data is invalid")

    base_link = str(_field(params, "base_link", "base_link"))
    joint_names = tuple(str(item) for item in _field(params, "joint_names", ()))
    if len(joint_names) != num_dof:
        raise UnsupportedCuroboConfig("joint_names length must equal num_dof")

    return BackendRobotConfig(
        fixed_transforms=fixed,
        parent_link=parent,
        joint_index=joint_index,
        joint_type=joint_type,
        joint_offset=offset,
        tool_link=tool_link,
        link_spheres=spheres,
        sphere_link=sphere_link,
        joint_names=joint_names,
        tool_frames=tool_frames,
        link_names=_link_names(params, links, base_link),
        base_link=base_link,
        num_dof=num_dof,
    )
