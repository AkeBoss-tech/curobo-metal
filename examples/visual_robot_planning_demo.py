"""Plan and render a real Franka mesh model using public cuRobo APIs.

The planner and FK run on CPU or Apple Metal. A small software renderer uses
the low-poly collision meshes shipped with the licensed Franka description, so
the checked-in GIF is deterministic and needs neither OpenGL nor a display.
"""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import DeviceCfg


LINKS = ["panda_link0", *[f"panda_link{i}" for i in range(1, 8)], "panda_hand"]
MESH_NAMES = ["link0", *[f"link{i}" for i in range(1, 8)], "hand"]
FRANKA_WHITE = np.array([222, 228, 231], dtype=np.float64)
FRANKA_DARK = np.array([43, 52, 59], dtype=np.float64)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if device.type not in {"cpu", "mps"}:
        raise ValueError("this portable demo supports cpu, mps, or auto")
    return device


def plan(device: torch.device, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    config = MotionPlannerCfg.create(
        "franka.yml",
        device_cfg=DeviceCfg(device),
        graph_planner_config=None,
        num_ik_seeds=2,
        num_trajopt_seeds=1,
        use_cuda_graph=False,
    )
    planner = MotionPlanner(config)
    start = planner.default_joint_state.unsqueeze(0)
    goal = start.clone()
    goal.position[..., 0] += 0.65
    goal.position[..., 1] -= 0.25
    goal.position[..., 3] += 0.20

    result = planner.plan_cspace(goal, start, max_attempts=1)
    if result is None or not bool(result.success.all().item()):
        raise RuntimeError("motion planner did not find the deterministic demo path")
    trajectory_state = result.get_interpolated_plan().reorder(planner.joint_names)
    trajectory = trajectory_state.position.reshape(-1, planner.action_dim)
    indices = torch.linspace(0, trajectory.shape[0] - 1, frame_count, device=device)
    sampled = trajectory.index_select(0, indices.round().long())
    poses = planner.kinematics.get_link_poses(sampled, LINKS)
    return poses.position.detach().cpu().numpy(), poses.quaternion.detach().cpu().numpy()


def quaternion_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized wxyz quaternions to rotation matrices."""
    q = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    result = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0, 0] = 1 - 2 * (y * y + z * z)
    result[..., 0, 1] = 2 * (x * y - z * w)
    result[..., 0, 2] = 2 * (x * z + y * w)
    result[..., 1, 0] = 2 * (x * y + z * w)
    result[..., 1, 1] = 1 - 2 * (x * x + z * z)
    result[..., 1, 2] = 2 * (y * z - x * w)
    result[..., 2, 0] = 2 * (x * z - y * w)
    result[..., 2, 1] = 2 * (y * z + x * w)
    result[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def load_franka_meshes():
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError('rendering requires: pip install "curobo-metal[demo]"') from exc
    mesh_root = (
        Path(str(files("curobo.content")))
        / "assets/robot/franka_description/meshes/collision"
    )
    meshes = []
    for link, name in zip(LINKS, MESH_NAMES):
        mesh = trimesh.load(mesh_root / f"{name}.obj", force="mesh", process=False)
        meshes.append((link, np.asarray(mesh.vertices), np.asarray(mesh.faces)))
    finger = trimesh.load(mesh_root / "finger.obj", force="mesh", process=False)
    finger_vertices, finger_faces = np.asarray(finger.vertices), np.asarray(finger.faces)
    meshes.extend(
        [
            (
                "panda_leftfinger",
                finger_vertices + [0.0, 0.025, 0.0584],
                finger_faces,
            ),
            (
                "panda_rightfinger",
                finger_vertices * [-1.0, -1.0, 1.0] + [0.0, -0.025, 0.0584],
                finger_faces,
            ),
        ]
    )
    return meshes


class Camera:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        eye = np.array([1.55, -1.75, 1.30])
        target = np.array([0.0, 0.0, 0.55])
        forward = target - eye
        self.forward = forward / np.linalg.norm(forward)
        self.right = np.cross(self.forward, np.array([0.0, 0.0, 1.0]))
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.forward)
        self.eye, self.focal = eye, 760.0

    def camera_points(self, world: np.ndarray) -> np.ndarray:
        offset = world - self.eye
        return np.stack(
            (offset @ self.right, offset @ self.up, offset @ self.forward), axis=-1
        )

    def project(self, world: np.ndarray) -> np.ndarray:
        camera = self.camera_points(world)
        depth = np.maximum(camera[..., 2:3], 0.05)
        xy = camera[..., :2] * self.focal / depth
        return np.stack(
            (self.width * 0.53 + xy[..., 0], self.height * 0.52 - xy[..., 1]),
            axis=-1,
        )


def render_gif(
    positions: np.ndarray,
    quaternions: np.ndarray,
    output: Path,
    device: torch.device,
    duration_ms: int,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError('rendering requires: pip install "curobo-metal[demo]"') from exc

    meshes = load_franka_meshes()
    rotations = quaternion_matrices(quaternions)
    width, height = 960, 540
    camera = Camera(width, height)
    font = ImageFont.load_default(size=18)
    small_font = ImageFont.load_default(size=14)
    light = np.array([-0.35, -0.45, 0.82])
    light /= np.linalg.norm(light)
    link_indices = {name: index for index, name in enumerate(LINKS)}
    trace_points = camera.project(positions[:, -1])
    frames = []

    for frame_index in range(len(positions)):
        image = Image.new("RGB", (width, height), "#07111f")
        draw = ImageDraw.Draw(image)
        for value in np.linspace(-0.8, 0.8, 9):
            line = camera.project(np.array([[value, -0.8, 0.0], [value, 0.8, 0.0]]))
            draw.line(tuple(map(tuple, line)), fill="#17304a", width=1)
            line = camera.project(np.array([[-0.8, value, 0.0], [0.8, value, 0.0]]))
            draw.line(tuple(map(tuple, line)), fill="#17304a", width=1)

        triangles = []
        for mesh_index, (link_name, vertices, faces) in enumerate(meshes):
            parent_name = "panda_hand" if "finger" in link_name else link_name
            pose_index = link_indices[parent_name]
            world_vertices = vertices @ rotations[frame_index, pose_index].T
            world_vertices = world_vertices + positions[frame_index, pose_index]
            screen = camera.project(world_vertices)
            camera_vertices = camera.camera_points(world_vertices)
            for face in faces:
                world_triangle = world_vertices[face]
                normal = np.cross(
                    world_triangle[1] - world_triangle[0],
                    world_triangle[2] - world_triangle[0],
                )
                length = np.linalg.norm(normal)
                if length < 1e-12:
                    continue
                normal /= length
                shade = 0.48 + 0.52 * max(0.0, float(normal @ light))
                dark_parts = {1, 3, 5, 7, 9, 10}
                base = FRANKA_DARK if mesh_index in dark_parts else FRANKA_WHITE
                color = tuple(np.clip(base * shade, 0, 255).astype(np.uint8))
                triangles.append(
                    (
                        float(camera_vertices[face, 2].mean()),
                        tuple(map(tuple, screen[face])),
                        color,
                    )
                )
        for _, polygon, color in sorted(triangles, key=lambda item: item[0], reverse=True):
            draw.polygon(polygon, fill=color)

        if frame_index:
            draw.line(
                tuple(map(tuple, trace_points[: frame_index + 1])),
                fill="#f5a65b",
                width=4,
            )
        endpoint = trace_points[frame_index]
        draw.ellipse(
            (endpoint[0] - 6, endpoint[1] - 6, endpoint[0] + 6, endpoint[1] + 6),
            fill="#f5a65b",
        )
        draw.text((34, 28), "cuRobo Metal", fill="#d9fbff", font=font)
        draw.text((34, 58), "Real Franka mesh motion plan", fill="#b995ff", font=small_font)
        draw.text(
            (34, height - 38),
            f"public cuRobo API  |  {device.type.upper()}  |  frame {frame_index + 1}/{len(positions)}",
            fill="#86a7c3",
            font=small_font,
        )
        frames.append(image)

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:] + list(reversed(frames[1:-1])),
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--duration-ms", type=int, default=55)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/franka-motion-plan.gif")
    )
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    device = select_device(args.device)
    positions, quaternions = plan(device, args.frames)
    render_gif(positions, quaternions, args.output, device, args.duration_ms)
    print(f"wrote {args.output} ({len(positions)} planned frames on {device.type})")


if __name__ == "__main__":
    main()
