from dataclasses import dataclass
import torch


@dataclass
class RobotMesh:
    vertices: torch.Tensor
    faces: torch.Tensor | None = None

    @classmethod
    def from_kinematics(cls, kinematics, joint_state):
        state = kinematics.compute_kinematics(joint_state)
        return cls(state.robot_spheres[..., :3].reshape(-1,3))
