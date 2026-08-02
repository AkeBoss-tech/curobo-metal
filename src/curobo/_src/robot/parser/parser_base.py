"""Backend-neutral robot parser protocol."""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional

from curobo._src.geom.types import Mesh, Obstacle
from curobo._src.robot.types import JointType, LinkParams


class RobotParser:
    def __init__(self, extra_links: Optional[Dict[str, LinkParams]] = None) -> None:
        self.extra_links = {} if extra_links is None else dict(extra_links)
        self.link_parent: Dict[str, str] = {}

    @abstractmethod
    def build_link_parent(self) -> None: ...

    @abstractmethod
    def get_link_parameters(self, link_name: str, base: bool = False) -> LinkParams: ...

    def add_absolute_path_to_link_meshes(self, mesh_dir: str = "") -> None:
        del mesh_dir

    @abstractmethod
    def get_link_mesh(self, link_name: str) -> Mesh: ...

    @abstractmethod
    def get_link_geometry(self, link_name: str) -> List[Obstacle]: ...

    def get_chain(self, base_link: str, ee_link: str) -> List[str]:
        if base_link == ee_link:
            return [base_link]
        result = [ee_link]
        while result[-1] != base_link:
            parent = self.link_parent.get(result[-1])
            if parent is None:
                raise ValueError(f"{ee_link!r} is not a descendant of {base_link!r}")
            if parent in result:
                raise ValueError("robot link graph contains a cycle")
            result.append(parent)
        return list(reversed(result))

    def get_actuated_joint_names(self) -> List[str]:
        names = []
        for link in self.get_link_names():
            params = self.get_link_parameters(link, base=link not in self.link_parent)
            if params.joint_type.name != "FIXED" and params.mimic_joint_name is None:
                names.append(params.joint_name)
        return names

    def get_mimic_joint_map(self) -> Dict[str, str]:
        result = {}
        for link in self.get_link_names():
            params = self.get_link_parameters(link, base=link not in self.link_parent)
            if params.mimic_joint_name is not None:
                result[params.joint_name] = params.mimic_joint_name
        return result

    def _get_from_extra_links(self, link_name: str) -> LinkParams:
        try:
            return self.extra_links[link_name]
        except KeyError as error:
            raise ValueError(f"unknown robot link: {link_name}") from error

    @abstractmethod
    def get_link_names(self) -> List[str]: ...


__all__ = ["RobotParser"]
