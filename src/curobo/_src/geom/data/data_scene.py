from dataclasses import dataclass, field
from typing import List, Optional, Union

from curobo._src.geom.types import SceneCfg
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class SceneData:
    cuboids: object | None = None
    meshes: object | None = None
    voxels: object | None = None
    num_envs: int = 1
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)
    scene_model: Optional[Union[SceneCfg, List[SceneCfg]]] = None

    @classmethod
    def from_scene_model(cls, scene_model, device_cfg=DeviceCfg()):
        models = scene_model if isinstance(scene_model, list) else [scene_model]
        return cls(num_envs=len(models), device_cfg=device_cfg, scene_model=scene_model)


class SceneDataWarp:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Warp scene data is unavailable on the portable backend")


__all__ = ["SceneData", "SceneDataWarp"]
