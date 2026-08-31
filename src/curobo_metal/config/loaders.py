"""YAML, URDF and XRDF loaders for the portable configuration model."""

from __future__ import annotations

from copy import deepcopy
import ast
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from curobo_metal.types import DeviceCfg

from .robot import (
    CSpaceConfig,
    CollisionSphere,
    JointConfig,
    JointLimits,
    LinkConfig,
    RobotCfg,
    UnsupportedConfigError,
    _topological_links,
)
from .world import WorldConfig


def _yaml_module() -> Any:
    try:
        import yaml
    except ImportError as error:
        raise UnsupportedConfigError(
            "YAML loading requires PyYAML; install pyyaml or pass an already parsed mapping"
        ) from error
    return yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        value = _yaml_module().safe_load(text)
    except UnsupportedConfigError:
        value = _parse_yaml(text)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: YAML root must be a mapping")
    return dict(value)


def dump_yaml(value: Mapping[str, Any]) -> str:
    try:
        return _yaml_module().safe_dump(
            dict(value), sort_keys=False, default_flow_style=False
        )
    except UnsupportedConfigError:
        return _dump_yaml(dict(value)) + "\n"


def _commentless(line: str) -> str:
    quote: str | None = None
    for index, character in enumerate(line):
        if character in {"'", '"'} and (index == 0 or line[index - 1] != "\\"):
            quote = None if quote == character else character if quote is None else quote
        elif character == "#" and quote is None:
            return line[:index]
    return line


def _split_inline(value: str) -> list[str]:
    parts: list[str] = []
    start = depth = 0
    quote: str | None = None
    for index, character in enumerate(value):
        if character in {"'", '"'} and (index == 0 or value[index - 1] != "\\"):
            quote = None if quote == character else character if quote is None else quote
        elif quote is None:
            if character in "[{":
                depth += 1
            elif character in "]}":
                depth -= 1
            elif character == "," and depth == 0:
                parts.append(value[start:index].strip())
                start = index + 1
    parts.append(value[start:].strip())
    return parts


def _scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_scalar(item) for item in _split_inline(inner)]
    if text.startswith("{") and text.endswith("}"):
        result: dict[str, Any] = {}
        for item in _split_inline(text[1:-1]):
            key, raw = item.split(":", 1)
            result[str(_scalar(key))] = _scalar(raw)
        return result
    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return float(text) if any(token in text.lower() for token in (".", "e")) else int(text)
        except ValueError:
            return text


def _key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"invalid YAML mapping entry: {text!r}")
    key, value = text.split(":", 1)
    return str(_scalar(key.strip())), value.strip()


def _parse_yaml(text: str) -> Any:
    """Parse the conservative YAML subset used by cuRobo robot/XRDF files."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        clean = _commentless(raw).rstrip()
        if not clean.strip() or clean.lstrip().startswith("---"):
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        if "\t" in clean[:indent]:
            raise ValueError("YAML indentation must use spaces")
        lines.append((indent, clean.lstrip()))

    def block(index: int, indent: int) -> tuple[Any, int]:
        is_list = lines[index][1].startswith("-")
        result: Any = [] if is_list else {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ValueError(f"unexpected YAML indentation near {content!r}")
            if is_list:
                if not content.startswith("-"):
                    break
                item = content[1:].strip()
                index += 1
                if not item:
                    if index >= len(lines) or lines[index][0] <= indent:
                        result.append(None)
                    else:
                        child, index = block(index, lines[index][0])
                        result.append(child)
                elif ":" in item:
                    key, raw_value = _key_value(item)
                    row: dict[str, Any] = {key: _scalar(raw_value)}
                    if not raw_value and index < len(lines) and lines[index][0] > indent:
                        row[key], index = block(index, lines[index][0])
                    if index < len(lines) and lines[index][0] > indent:
                        continuation, index = block(index, lines[index][0])
                        if not isinstance(continuation, dict):
                            raise ValueError("YAML list mapping continuation must be a mapping")
                        row.update(continuation)
                    result.append(row)
                else:
                    result.append(_scalar(item))
            else:
                if content.startswith("-"):
                    break
                key, raw_value = _key_value(content)
                index += 1
                if raw_value:
                    result[key] = _scalar(raw_value)
                elif index < len(lines) and (
                    lines[index][0] > indent
                    or (
                        lines[index][0] == indent
                        and lines[index][1].startswith("-")
                    )
                ):
                    # YAML permits an "indentless sequence" directly below a
                    # mapping key. cuRobo's own robot configs use this form:
                    #
                    #   tool_frames:
                    #   - panda_hand
                    #
                    # The recursive list parser stops when the next mapping
                    # entry at this indentation is reached.
                    result[key], index = block(index, lines[index][0])
                else:
                    result[key] = None
        return result, index

    if not lines:
        return {}
    value, end = block(0, lines[0][0])
    if end != len(lines):
        raise ValueError("could not parse complete YAML document")
    return value


def _dump_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, Mapping):
        rows = []
        for key, item in value.items():
            if isinstance(item, (Mapping, list)) and item:
                rows.append(f"{prefix}{key}:")
                rows.append(_dump_yaml(item, indent + 2))
            elif item == []:
                rows.append(f"{prefix}{key}: []")
            else:
                rows.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return "\n".join(rows)
    if isinstance(value, list):
        rows = []
        for item in value:
            if isinstance(item, Mapping):
                rendered = _dump_yaml(item, indent + 2).splitlines()
                rows.append(f"{prefix}- {rendered[0].lstrip()}")
                rows.extend(rendered[1:])
            elif isinstance(item, list):
                rows.append(f"{prefix}-")
                rows.append(_dump_yaml(item, indent + 2))
            else:
                rows.append(f"{prefix}- {_yaml_scalar(item)}")
        return "\n".join(rows)
    return f"{prefix}{_yaml_scalar(value)}"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def _numbers(value: str | None, count: int, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        return tuple(default)
    result = tuple(float(item) for item in value.split())
    if len(result) != count:
        raise ValueError(f"expected {count} numeric values, found {len(result)}")
    return result


def load_urdf(
    path: str | Path,
    *,
    base_link: str | None = None,
    tool_frames: Sequence[str] | None = None,
) -> RobotCfg:
    source = Path(path).resolve()
    if source.suffix.lower() in {".usd", ".usda", ".usdc"}:
        raise UnsupportedConfigError(
            "USD/Isaac robot loading is unavailable; provide a URDF path or parsed mapping"
        )
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as error:
        raise ValueError(f"{source}: invalid URDF XML: {error}") from error
    if root.tag != "robot":
        raise ValueError(f"{source}: URDF root element must be <robot>")
    links: list[LinkConfig] = []
    for element in root.findall("link"):
        name = element.get("name")
        if not name:
            raise ValueError(f"{source}: every link requires a name")
        inertial = element.find("inertial")
        mass = 0.0
        com = (0.0, 0.0, 0.0)
        inertia = (0.0,) * 6
        if inertial is not None:
            origin = inertial.find("origin")
            com = _numbers(None if origin is None else origin.get("xyz"), 3, com)
            mass_node = inertial.find("mass")
            mass = float(0.0 if mass_node is None else mass_node.get("value", "0"))
            tensor = inertial.find("inertia")
            if tensor is not None:
                inertia = tuple(float(tensor.get(key, "0")) for key in (
                    "ixx", "iyy", "izz", "ixy", "ixz", "iyz"
                ))
        links.append(LinkConfig(name, mass, com, inertia))
    link_names = [link.name for link in links]
    if len(set(link_names)) != len(link_names):
        raise ValueError(f"{source}: duplicate URDF link name")

    joints: list[JointConfig] = []
    child_names: set[str] = set()
    for element in root.findall("joint"):
        name = element.get("name")
        raw_kind = element.get("type")
        if not name or raw_kind is None:
            raise ValueError(f"{source}: every joint requires name and type")
        kind = "revolute" if raw_kind == "continuous" else raw_kind
        if kind not in {"fixed", "revolute", "prismatic"}:
            raise UnsupportedConfigError(
                f"{source}: joint {name!r} has unsupported URDF type {raw_kind!r}"
            )
        parent_node, child_node = element.find("parent"), element.find("child")
        if parent_node is None or child_node is None:
            raise ValueError(f"{source}: joint {name!r} requires parent and child")
        parent, child = parent_node.get("link"), child_node.get("link")
        if parent not in link_names or child not in link_names:
            raise ValueError(f"{source}: joint {name!r} references an unknown link")
        if child in child_names:
            raise ValueError(f"{source}: link {child!r} has multiple parent joints")
        child_names.add(child)
        origin, axis_node, limit_node, mimic_node = (
            element.find("origin"), element.find("axis"), element.find("limit"),
            element.find("mimic"),
        )
        lower = -float("inf") if raw_kind == "continuous" else float(
            "-inf" if limit_node is None else limit_node.get("lower", "-inf")
        )
        upper = float("inf") if raw_kind == "continuous" else float(
            "inf" if limit_node is None else limit_node.get("upper", "inf")
        )
        joints.append(JointConfig(
            name=name, kind=kind, parent=str(parent), child=str(child),
            axis=_numbers(None if axis_node is None else axis_node.get("xyz"), 3, (1, 0, 0)),
            xyz=_numbers(None if origin is None else origin.get("xyz"), 3, (0, 0, 0)),
            rpy=_numbers(None if origin is None else origin.get("rpy"), 3, (0, 0, 0)),
            limits=JointLimits(
                lower, upper,
                float("inf" if limit_node is None else limit_node.get("velocity", "inf")),
                float("inf" if limit_node is None else limit_node.get("effort", "inf")),
            ),
            mimic_joint=None if mimic_node is None else mimic_node.get("joint"),
            mimic_multiplier=float(1.0 if mimic_node is None else mimic_node.get("multiplier", "1")),
            mimic_offset=float(0.0 if mimic_node is None else mimic_node.get("offset", "0")),
        ))
    roots = [name for name in link_names if name not in child_names]
    if base_link is None:
        if len(roots) != 1:
            raise ValueError(f"{source}: URDF must have exactly one root link")
        base_link = roots[0]
    elif base_link not in link_names:
        raise ValueError(f"{source}: base_link {base_link!r} is absent from URDF links")
    else:
        # cuRobo robot YAML may deliberately choose a kinematic base below
        # the physical URDF root (Franka uses ``panda_link0`` below
        # ``base_link``). Keep exactly that descendant subtree.
        child_map: dict[str, list[str]] = {}
        for joint in joints:
            child_map.setdefault(joint.parent, []).append(joint.child)
        reachable: set[str] = set()
        queue = [base_link]
        while queue:
            name = queue.pop(0)
            if name in reachable:
                raise ValueError(f"{source}: URDF joint graph contains a cycle")
            reachable.add(name)
            queue.extend(child_map.get(name, []))
        links = [link for link in links if link.name in reachable]
        joints = [
            joint
            for joint in joints
            if joint.parent in reachable and joint.child in reachable
        ]
        link_names = [link.name for link in links]
    children = {joint.parent for joint in joints}
    inferred_tools = [name for name in link_names if name not in children]
    result = RobotCfg(
        root.get("name", source.stem), base_link,
        list(inferred_tools if tool_frames is None else tool_frames),
        links, joints, urdf_path=str(source), source_path=str(source),
    )
    # Validate graph structure without imposing rigid-body inertia constraints.
    # FK-only upstream assets can contain rounded inertia tensors that are not
    # positive semidefinite; dynamics compilation performs that stricter check.
    _topological_links(result.base_link, result.links, result.joints)
    movable = [
        joint for joint in joints
        if joint.kind != "fixed" and joint.mimic_joint is None
    ]
    result.cspace = CSpaceConfig(
        joint_names=[joint.name for joint in movable],
        default_joint_position=[0.0] * len(movable),
        max_velocity=[joint.limits.velocity for joint in movable],
    )
    return result


def load_xrdf(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = load_yaml(source)
    if data.get("format") != "xrdf":
        raise ValueError(f"{source}: XRDF format must be 'xrdf'")
    geometry_name = data.get("collision", {}).get("geometry")
    geometries = data.get("geometry", {})
    sphere_mapping: Mapping[str, Any] = {}
    if geometry_name is not None:
        if geometry_name not in geometries:
            raise ValueError(f"{source}: XRDF collision geometry {geometry_name!r} is absent")
        sphere_mapping = geometries[geometry_name].get("spheres", {})
    spheres = _sphere_records(sphere_mapping)
    return {
        "tool_frames": list(data.get("tool_frames", [])),
        "cspace": dict(data.get("cspace", {})),
        "default_joint_positions": dict(data.get("default_joint_positions", {})),
        "collision_spheres": spheres,
        "self_collision_ignore": deepcopy(data.get("self_collision", {}).get("ignore", {})),
        "self_collision_buffer": deepcopy(
            data.get("self_collision", {}).get("buffer_distance", {})
        ),
    }


def _sphere_records(value: Mapping[str, Any]) -> list[CollisionSphere]:
    result: list[CollisionSphere] = []
    for link_name, rows in value.items():
        if not isinstance(rows, list):
            raise ValueError(f"collision_spheres.{link_name} must be a list")
        for row in rows:
            center = tuple(float(item) for item in row["center"])
            radius = float(row["radius"])
            if len(center) != 3:
                raise ValueError("collision sphere center must contain three values")
            # Negative XRDF radii are cuRobo's disabled-sphere convention.
            result.append(CollisionSphere(str(link_name), center, radius))
    return result


def _resolve_asset(path: str, source: Path | None, asset_root: str | None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    bases = []
    if source is not None:
        bases.append(source.parent)
        # Pinned cuRobo content paths are rooted at ``content/assets`` while
        # robot YAML files live under ``content/configs/robot``. For example,
        # ``franka.yml`` refers to
        # ``robot/franka_description/franka_panda.urdf``.
        for parent in source.parents:
            if parent.name == "content":
                bases.append(parent / "assets")
                break
        if asset_root:
            bases.append(source.parent / asset_root)
    # Parsed mappings lose the YAML file path, but cuRobo's packaged mappings
    # still use paths rooted at ``curobo/content/assets``.  Resolve that wheel
    # resource before ever falling back to the caller's current directory.
    packaged_assets = Path(__file__).resolve().parents[2] / "curobo" / "content" / "assets"
    if packaged_assets.is_dir():
        bases.append(packaged_assets)
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (bases[-1] / candidate).resolve() if bases else candidate.resolve()


def _pose_rpy(value: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert cuRobo's ``[x, y, z, qw, qx, qy, qz]`` pose to URDF fields."""
    pose = [float(item) for item in value]
    if len(pose) != 7:
        raise ValueError("extra_links.fixed_transform must contain seven values")
    x, y, z, qw, qx, qy, qz = pose
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("extra_links.fixed_transform quaternion must be nonzero")
    qw, qx, qy, qz = (item / norm for item in (qw, qx, qy, qz))
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (x, y, z), (roll, pitch, yaw)


def _insert_extra_links(robot: RobotCfg, value: Any) -> None:
    """Materialize YAML ``extra_links`` in the portable topology.

    ``child_link_name`` inserts the new link between its declared parent and
    an existing child.  This is the topology contract used for virtual-base
    joints and is preserved by FK and URDF export.
    """
    if value in (None, {}):
        return
    if not isinstance(value, Mapping):
        raise TypeError("extra_links must be a mapping or None")
    type_map = {
        "FIXED": ("fixed", (1.0, 0.0, 0.0)),
        "X_PRISM": ("prismatic", (1.0, 0.0, 0.0)),
        "Y_PRISM": ("prismatic", (0.0, 1.0, 0.0)),
        "Z_PRISM": ("prismatic", (0.0, 0.0, 1.0)),
        "X_PRISM_NEG": ("prismatic", (-1.0, 0.0, 0.0)),
        "Y_PRISM_NEG": ("prismatic", (0.0, -1.0, 0.0)),
        "Z_PRISM_NEG": ("prismatic", (0.0, 0.0, -1.0)),
        "X_ROT": ("revolute", (1.0, 0.0, 0.0)),
        "Y_ROT": ("revolute", (0.0, 1.0, 0.0)),
        "Z_ROT": ("revolute", (0.0, 0.0, 1.0)),
        "X_ROT_NEG": ("revolute", (-1.0, 0.0, 0.0)),
        "Y_ROT_NEG": ("revolute", (0.0, -1.0, 0.0)),
        "Z_ROT_NEG": ("revolute", (0.0, 0.0, -1.0)),
    }
    known_links = {link.name for link in robot.links}
    known_joints = {joint.name for joint in robot.joints}
    for key, raw in value.items():
        if not isinstance(raw, Mapping):
            raise TypeError("extra_links values must be mappings")
        row = dict(raw)
        link_name = str(row.get("link_name", key))
        if str(key) != link_name:
            raise ValueError("extra_links key must match link_name")
        joint_name = str(row.get("joint_name", ""))
        parent = row.get("parent_link_name")
        if not joint_name or not isinstance(parent, str) or not parent:
            raise ValueError("each extra link requires joint_name and parent_link_name")
        if link_name in known_links or joint_name in known_joints:
            raise ValueError(f"duplicate extra link or joint: {link_name!r}, {joint_name!r}")
        if parent not in known_links:
            raise ValueError(f"extra link {link_name!r} references unknown parent {parent!r}")
        type_name = getattr(row.get("joint_type"), "name", row.get("joint_type"))
        if type_name not in type_map:
            raise UnsupportedConfigError(f"unsupported extra-link joint type {type_name!r}")
        kind, default_axis = type_map[str(type_name)]
        axis_raw = row.get("joint_axis")
        axis = default_axis if axis_raw is None else tuple(float(item) for item in axis_raw)
        if len(axis) != 3:
            raise ValueError("extra_links.joint_axis must contain three values")
        xyz, rpy = _pose_rpy(row.get("fixed_transform", [0, 0, 0, 1, 0, 0, 0]))
        limits = row.get("joint_limits") or [-math.inf, math.inf]
        if len(limits) != 2:
            raise ValueError("extra_links.joint_limits must contain lower and upper")
        velocity = row.get("joint_velocity_limits") or [-math.inf, math.inf]
        velocity_limit = max(abs(float(item)) for item in velocity)
        effort = row.get("joint_effort_limit") or [math.inf]
        effort_limit = max(abs(float(item)) for item in effort)
        child = row.get("child_link_name")
        if child is not None:
            matches = [joint for joint in robot.joints if joint.child == child]
            if len(matches) != 1:
                raise ValueError(
                    f"extra link {link_name!r} child {child!r} must have one parent joint"
                )
            matches[0].parent = link_name
        robot.links.append(LinkConfig(
            link_name,
            float(row.get("link_mass", 0.01)),
            tuple(float(item) for item in row.get("link_com", (0.0, 0.0, 0.0))),
            tuple(float(item) for item in row.get(
                "link_inertia", (1e-4, 1e-4, 1e-4, 0.0, 0.0, 0.0)
            )),
        ))
        robot.joints.append(JointConfig(
            joint_name, kind, parent, link_name, axis=axis, xyz=xyz, rpy=rpy,
            limits=JointLimits(float(limits[0]), float(limits[1]), velocity_limit, effort_limit),
            mimic_joint=row.get("mimic_joint_name"),
        ))
        known_links.add(link_name)
        known_joints.add(joint_name)
    _topological_links(robot.base_link, robot.links, robot.joints)


def load_robot_config(
    value: str | Path | Mapping[str, Any],
    *,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> RobotCfg:
    source: Path | None = None
    if isinstance(value, (str, Path)):
        source = Path(value).resolve()
        suffix = source.suffix.lower()
        if suffix == ".urdf":
            result = load_urdf(source)
            result.device_cfg = device_cfg
            return result
        if suffix in {".usd", ".usda", ".usdc"}:
            raise UnsupportedConfigError(
                "USD/Isaac robot loading is unavailable; provide a URDF or YAML robot config"
            )
        if suffix not in {".yml", ".yaml", ".json", ".xrdf"}:
            raise UnsupportedConfigError(
                f"unsupported robot config extension {suffix!r}; expected YAML, JSON, URDF, or XRDF"
            )
        data = json.loads(source.read_text()) if suffix == ".json" else load_yaml(source)
    elif isinstance(value, Mapping):
        data = deepcopy(dict(value))
    else:
        raise TypeError("robot config must be a path or mapping")

    raw = data.get("robot_cfg", data)
    kinematics = raw.get("kinematics", raw)
    if not isinstance(kinematics, Mapping):
        raise ValueError("robot_cfg.kinematics must be a mapping")
    if any(key in kinematics for key in ("usd_path", "usd_robot_root", "isaac_asset")):
        raise UnsupportedConfigError(
            "USD/Isaac fields are unsupported; configure urdf_path instead"
        )
    urdf_path = kinematics.get("urdf_path")
    if not urdf_path:
        raise ValueError("robot YAML requires robot_cfg.kinematics.urdf_path")
    urdf = _resolve_asset(str(urdf_path), source, kinematics.get("asset_root_path"))
    result = load_urdf(
        urdf,
        base_link=kinematics.get("base_link"),
        tool_frames=kinematics.get("tool_frames"),
    )
    _insert_extra_links(result, kinematics.get("extra_links"))
    result.source_path = None if source is None else str(source)
    result.device_cfg = device_cfg
    result.metadata = {
        key: deepcopy(item) for key, item in kinematics.items()
        if key not in {
            "cspace", "collision_spheres", "self_collision_ignore",
            "self_collision_buffer"
        }
    }
    result.collision_link_names = list(kinematics.get("collision_link_names") or [])
    result.collision_spheres = _sphere_records(kinematics.get("collision_spheres") or {})
    extra_spheres = kinematics.get("extra_collision_spheres") or {}
    if not isinstance(extra_spheres, Mapping):
        raise TypeError("extra_collision_spheres must be a mapping or None")
    known_links = {link.name for link in result.links}
    for link_name, count in extra_spheres.items():
        if link_name not in known_links:
            raise ValueError(f"extra collision spheres reference unknown link {link_name!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("extra_collision_spheres counts must be non-negative integers")
        result.collision_spheres.extend(
            CollisionSphere(str(link_name), (0.0, 0.0, 0.0), -100.0)
            for _ in range(count)
        )
    result.self_collision_ignore = deepcopy(kinematics.get("self_collision_ignore") or {})
    result.self_collision_buffer = {
        str(key): float(item)
        for key, item in (kinematics.get("self_collision_buffer") or {}).items()
    }

    xrdf_path = kinematics.get("xrdf_path")
    xrdf: dict[str, Any] | None = None
    if xrdf_path:
        resolved_xrdf = _resolve_asset(
            str(xrdf_path), source, kinematics.get("asset_root_path")
        )
        xrdf = load_xrdf(resolved_xrdf)
        result.xrdf_path = str(resolved_xrdf)
        if xrdf["tool_frames"]:
            result.tool_frames = xrdf["tool_frames"]
        if xrdf["collision_spheres"]:
            result.collision_spheres = xrdf["collision_spheres"]
        result.self_collision_ignore = xrdf["self_collision_ignore"]
        result.self_collision_buffer = xrdf["self_collision_buffer"]
    cspace_raw = deepcopy(kinematics.get("cspace", {}))
    if xrdf:
        cspace_raw = {**xrdf["cspace"], **cspace_raw}
        if not cspace_raw.get("default_joint_position"):
            defaults = xrdf["default_joint_positions"]
            names = cspace_raw.get("joint_names", result.joint_names)
            cspace_raw["default_joint_position"] = [defaults.get(name, 0.0) for name in names]
    aliases = {
        "acceleration_limits": "max_acceleration",
        "jerk_limits": "max_jerk",
    }
    for old, new in aliases.items():
        if old in cspace_raw and new not in cspace_raw:
            cspace_raw[new] = cspace_raw.pop(old)
    allowed = set(CSpaceConfig.__dataclass_fields__)
    result.cspace = CSpaceConfig(**{
        key: item for key, item in cspace_raw.items() if key in allowed
    })
    if not result.cspace.joint_names:
        result.cspace.joint_names = [
            joint.name for joint in result.joints
            if joint.kind != "fixed" and joint.mimic_joint is None
        ]
    if not result.cspace.default_joint_position:
        result.cspace.default_joint_position = [0.0] * len(result.cspace.joint_names)
    unknown = set(result.cspace.joint_names) - {
        joint.name for joint in result.joints
        if joint.kind != "fixed" and joint.mimic_joint is None
    }
    if unknown:
        raise ValueError(f"cspace references unknown/non-independent joints: {sorted(unknown)}")
    return result


def load_world_config(value: str | Path | Mapping[str, Any]) -> WorldConfig:
    if isinstance(value, (str, Path)):
        path = Path(value)
        data = json.loads(path.read_text()) if path.suffix.lower() == ".json" else load_yaml(path)
    elif isinstance(value, Mapping):
        data = value
    else:
        raise TypeError("world config must be a path or mapping")
    return WorldConfig.from_dict(data)
