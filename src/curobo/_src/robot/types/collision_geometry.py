"""Robot collision geometry tensor record."""

from dataclasses import dataclass
import torch


@dataclass
class RobotCollisionGeometry:
    link_sphere_idx_map: torch.Tensor
    num_links: int

    def clone(self) -> "RobotCollisionGeometry":
        return RobotCollisionGeometry(self.link_sphere_idx_map.clone(), self.num_links)

    def copy_(self, other: "RobotCollisionGeometry") -> None:
        self.link_sphere_idx_map.copy_(other.link_sphere_idx_map)
        self.num_links = other.num_links

    def detach(self) -> "RobotCollisionGeometry":
        return RobotCollisionGeometry(self.link_sphere_idx_map.detach(), self.num_links)


__all__ = ["RobotCollisionGeometry"]
