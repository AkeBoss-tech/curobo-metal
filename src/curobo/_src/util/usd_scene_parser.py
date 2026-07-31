"""Lazy OpenUSD scene parser compatibility surface."""
from __future__ import annotations
from .usd_util import _usd
def _unsupported(*args,**kwargs):_usd();raise NotImplementedError("portable USD scene conversion is unavailable; use an explicit SceneCfg")
get_cylinder_attrs=get_capsule_attrs=get_cube_attrs=get_sphere_attrs=get_mesh_attrs=_unsupported
class UsdSceneParser:
    def __init__(self):self.stage=None
    def load_stage_from_file(self,file_path):_,Usd,_=_usd();self.stage=Usd.Stage.Open(file_path);return self.stage
    def load_stage(self,stage):_usd();self.stage=stage;return self
    def get_pose(self,*args,**kwargs):return _unsupported(*args,**kwargs)
    def get_obstacles_from_stage(self,*args,**kwargs):return _unsupported(*args,**kwargs)
__all__=["UsdSceneParser","get_cylinder_attrs","get_capsule_attrs","get_cube_attrs","get_sphere_attrs","get_mesh_attrs"]
