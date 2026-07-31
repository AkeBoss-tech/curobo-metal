"""OpenUSD writer facade; imports safely without usd-core."""
from __future__ import annotations
from .usd_util import *
from .usd_util import _usd
def _require_usd():_usd()
def _unsupported(*args,**kwargs):_require_usd();raise NotImplementedError("this USD operation is not implemented by the portable backend")
set_geom_mesh_attrs=set_geom_cube_attrs=set_geom_cylinder_attrs=set_geom_sphere_attrs=set_cylinder_attrs=_unsupported
get_cylinder_attrs=get_capsule_attrs=get_cube_attrs=get_sphere_attrs=get_mesh_attrs=_unsupported
class UsdWriter:
    def __init__(self,use_float=True):_require_usd();self.use_float=use_float;self.stage=None
    def create_stage(self,name="curobo_stage.usd",base_frame="/world",timesteps=None,dt=.02,interpolation_steps=1):self.stage=create_stage(name,base_frame);return self.stage
    def load_stage_from_file(self,file_path):_,Usd,_=_usd();self.stage=Usd.Stage.Open(file_path);return self.stage
    def load_stage(self,stage):self.stage=stage;return self
    def write_stage_to_file(self,file_path,flatten=False):
        if self.stage is None:raise RuntimeError("no USD stage loaded")
        self.stage.Flatten().Export(file_path) if flatten else self.stage.Export(file_path)
    def save(self):
        if self.stage is None:raise RuntimeError("no USD stage loaded")
        return self.stage.Save()
    def __getattr__(self,name):
        if name.startswith("_"):raise AttributeError(name)
        return _unsupported
__all__=["UsdWriter","join_usd_path","set_prim_translate","set_prim_transform","get_prim_world_pose","get_transform","get_position_quat","create_stage","set_geom_mesh_attrs","set_geom_cube_attrs","set_geom_cylinder_attrs","set_geom_sphere_attrs","set_cylinder_attrs","get_cylinder_attrs","get_capsule_attrs","get_cube_attrs","get_sphere_attrs","get_mesh_attrs"]
