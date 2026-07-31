import importlib
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from curobo._src.geom.convex_polygon_helper import ConvexPolygon2DHelper
from curobo._src.geom.data import CuboidData, MeshData, VoxelData
from curobo._src.geom.sphere_fit.fit_voxel import sample_even_fit_mesh, voxel_fit_mesh
from curobo._src.geom.sphere_fit.metrics import compute_sphere_fit_metrics
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.benchmark_metrics import CuroboGroupMetrics, CuroboMetrics, Statistic
from curobo._src.util.error_metrics import rotation_error_matrix
from curobo._src.util.xrdf_util import convert_curobo_to_xrdf

def test_convex_hull_signed_distance():
    helper=ConvexPolygon2DHelper();helper.build_convex_hull(torch.tensor([[[0.,0.],[1,0],[1,1],[0,1],[.5,.5]]]))
    result=helper.compute_point_hull_distance(torch.tensor([[[[.5,.5],[2.,.5]]]]),torch.tensor([0]))
    torch.testing.assert_close(result,torch.tensor([[[-.5,1.]]]))

def test_portable_data_caches():
    cfg=DeviceCfg()
    cub=Cuboid("box",[0,0,0,1,0,0,0],dims=[1,2,3])
    data=CuboidData.from_scene_cfg(SceneCfg(cuboid=[cub]),cfg)
    assert data.get_names()==["box"] and data.dims.shape==(1,1,4)
    mesh=Mesh("tri",vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]])
    md=MeshData.from_scene_cfg(SceneCfg(mesh=[mesh]),cfg)
    assert md.get_cached_mesh_names()==["tri"]
    grid=VoxelGrid("grid",pose=[0,0,0,1,0,0,0],dims=[2,2,2],voxel_size=1,feature_tensor=torch.arange(8.))
    vd=VoxelData.from_scene_cfg(SceneCfg(voxel=[grid]),cfg)
    assert vd.get_grid_shape(name="grid")==torch.Size([2,2,2])
    with pytest.raises(NotImplementedError):data.to_warp()

def test_deterministic_sphere_fit_and_metrics():
    vertices=np.array([[-.5,-.5,-.5],[.5,-.5,-.5],[.5,.5,-.5],[-.5,.5,-.5],
                       [-.5,-.5,.5],[.5,-.5,.5],[.5,.5,.5],[-.5,.5,.5]])
    faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                    [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    mesh=SimpleNamespace(vertices=vertices,faces=faces,bounds=np.array([[-.5]*3,[.5]*3]),
                         is_watertight=True,volume=1.)
    c1,r1=sample_even_fit_mesh(mesh,4,.1);c2,r2=sample_even_fit_mesh(mesh,4,.1)
    np.testing.assert_array_equal(c1,c2);np.testing.assert_array_equal(r1,r2)
    centers,radii=voxel_fit_mesh(mesh,8,torch.device("cpu"))
    assert centers is not None and np.all(radii>0)
    metrics=compute_sphere_fit_metrics(mesh,centers,radii,n_interior=64,n_surface=32,n_sphere_surface=8)
    assert metrics.num_spheres==len(centers) and 0<=metrics.coverage<=1

def test_metrics_and_xrdf_helpers():
    stat=Statistic.from_list([1,2,3]);assert stat.mean==2
    group=CuroboGroupMetrics.from_list([CuroboMetrics(skip=False,success=True,position_error=.5,orientation_error=5)])
    assert group.success==100
    xrdf=convert_curobo_to_xrdf({"robot_cfg":{"kinematics":{"tool_frames":{"tool0":{}},
        "collision_spheres":{},"cspace":{"joint_names":["j"],"default_joint_position":[.2],"max_acceleration":[1.],"max_jerk":[2.]}}}})
    assert xrdf["format"]=="xrdf" and xrdf["default_joint_positions"]["j"]==.2
    torch.testing.assert_close(rotation_error_matrix(torch.eye(3),torch.eye(3)),torch.tensor(0.))

def test_optional_platform_modules_import_safely():
    for name in ("curobo._src.util.usd_util","curobo._src.util.usd_scene_parser","curobo._src.util.usd_writer","curobo._src.util.viser_visualizer","curobo.profiling","curobo.viewer"):
        importlib.import_module(name)
    from curobo.viewer import UsdWriter
    with pytest.raises(ImportError,match="usd-core"):UsdWriter()
