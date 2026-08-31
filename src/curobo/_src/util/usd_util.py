"""OpenUSD helpers with lazy optional dependency loading."""
from __future__ import annotations
from typing import List

from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose

def _usd():
    try:from pxr import Gf,Usd,UsdGeom
    except ImportError as e:raise ImportError("usd-core is required for OpenUSD operations; install usd-core") from e
    return Gf,Usd,UsdGeom
def join_usd_path(path1: str, path2: str) -> str:
    path1 = path1.rstrip("/") if path1 else path1
    path2 = path2.lstrip("/") if path2 else path2
    if path1 and path2:
        return f"{path1}/{path2}"
    if path2:
        return f"/{path2}"
    return path1
def set_prim_translate(prim,translation):
    Gf,_,UsdGeom=_usd();return UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*translation))
def set_prim_transform(
    prim, pose: List[float], scale: List[float] = [1, 1, 1], use_float: bool = False
):
    Gf,_,UsdGeom=_usd();x=UsdGeom.Xformable(prim);x.ClearXformOpOrder();x.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]));x.AddOrientOp().Set(Gf.Quatd(pose[3],*pose[4:]));x.AddScaleOp().Set(Gf.Vec3d(*scale));return prim
def get_prim_world_pose(cache: UsdGeom.XformCache, prim: Usd.Prim, inverse: bool = False):
    world_transform = cache.GetLocalToWorldTransform(prim)
    scale = list(v.GetLength() for v in world_transform.ExtractRotationMatrix())
    transform = world_transform.RemoveScaleShear()
    if inverse:
        transform = transform.GetInverse()
    translation = transform.ExtractTranslation()
    quaternion = transform.ExtractRotation().GetQuaternion()
    orientation = [quaternion.GetReal()] + list(quaternion.GetImaginary())
    matrix = (
        Pose.from_list(list(translation) + orientation, DeviceCfg())
        .get_matrix()
        .view(4, 4)
        .cpu()
        .numpy()
    )
    return matrix, scale
def get_transform(pose):
    Gf,_,_=_usd();m=Gf.Matrix4d(1);m.SetTranslateOnly(Gf.Vec3d(*pose[:3]));m.SetRotateOnly(Gf.Quatd(pose[3],*pose[4:]));return m
def get_position_quat(pose, use_float: bool = True):
    Gf, _, _ = _usd()
    quat = pose[3:]
    if use_float:
        return Gf.Vec3f(pose[:3]), Gf.Quatf(quat[0], quat[1:])
    return Gf.Vec3d(pose[:3]), Gf.Quatd(quat[0], quat[1:])


def create_stage(
    name: str = "curobo_stage.usd",
    base_frame: str = "/world",
):
    _, Usd, UsdGeom = _usd()
    from pxr import UsdPhysics

    stage = Usd.Stage.CreateNew(name)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    xform = stage.DefinePrim(base_frame, "Xform")
    stage.SetDefaultPrim(xform)
    return stage
__all__=["join_usd_path","set_prim_translate","set_prim_transform","get_prim_world_pose","get_transform","get_position_quat","create_stage"]
