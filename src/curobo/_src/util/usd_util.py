"""OpenUSD helpers with lazy optional dependency loading."""
from __future__ import annotations
def _usd():
    try:from pxr import Gf,Usd,UsdGeom
    except ImportError as e:raise ImportError("usd-core is required for OpenUSD operations; install usd-core") from e
    return Gf,Usd,UsdGeom
def join_usd_path(path1,path2):return path1.rstrip("/")+"/"+path2.lstrip("/")
def set_prim_translate(prim,translation):
    Gf,_,UsdGeom=_usd();return UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*translation))
def set_prim_transform(prim,pose,scale=[1,1,1],use_float=False):
    Gf,_,UsdGeom=_usd();x=UsdGeom.Xformable(prim);x.ClearXformOpOrder();x.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]));x.AddOrientOp().Set(Gf.Quatd(pose[3],*pose[4:]));x.AddScaleOp().Set(Gf.Vec3d(*scale));return prim
def get_prim_world_pose(cache,prim,inverse=False):
    m=cache.GetLocalToWorldTransform(prim);return m.GetInverse() if inverse else m
def get_transform(pose):
    Gf,_,_=_usd();m=Gf.Matrix4d(1);m.SetTranslateOnly(Gf.Vec3d(*pose[:3]));m.SetRotateOnly(Gf.Quatd(pose[3],*pose[4:]));return m
def get_position_quat(pose,use_float=True):
    t=pose.ExtractTranslation();q=pose.ExtractRotationQuat();return [*t,q.GetReal(),*q.GetImaginary()]
def create_stage(name="curobo_stage.usd",base_frame="/world"):
    _,Usd,UsdGeom=_usd();stage=Usd.Stage.CreateNew(name);UsdGeom.Xform.Define(stage,base_frame);return stage
__all__=["join_usd_path","set_prim_translate","set_prim_transform","get_prim_world_pose","get_transform","get_position_quat","create_stage"]
