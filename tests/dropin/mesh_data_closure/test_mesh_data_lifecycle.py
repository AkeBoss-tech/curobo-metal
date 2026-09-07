"""Transactional cache and transform coverage for portable ``MeshData``."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.data.data_mesh import MeshData
from curobo._src.geom.types import Mesh, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg


def _tetra(name: str, pose=None) -> Mesh:
    return Mesh(
        name=name,
        pose=pose or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        device_cfg=DeviceCfg("cpu"),
    )


def test_load_batch_is_transactional_when_a_later_mesh_is_invalid() -> None:
    data = MeshData.from_scene_cfg(SceneCfg(mesh=[_tetra("stable")]), DeviceCfg("cpu"), max_n=2)
    malformed = _tetra("malformed")
    malformed.faces = [[0, 0, 1]]

    with pytest.raises(ValueError, match="nondegenerate"):
        data.load_batch([_tetra("new"), malformed], 0)

    assert data.get_names(0) == ["stable"]
    assert data.get_cached_mesh_names() == ["stable"]
    torch.testing.assert_close(data.get_world_pose("stable"), torch.tensor([0.0, 0, 0, 1, 0, 0, 0]))


def test_shared_cache_refuses_orphaning_another_environment() -> None:
    data = MeshData.from_batch_scene_cfg(
        [SceneCfg(mesh=[_tetra("a")]), SceneCfg(mesh=[_tetra("b")])], DeviceCfg("cpu")
    )
    with pytest.raises(ValueError, match="another environment"):
        data.clear(env_idx=0, clear_warp_cache=True)
    assert data.get_names(0) == ["a"]
    assert data.get_names(1) == ["b"]

    data.clear(clear_warp_cache=True)
    assert data.get_cached_mesh_names() == []
    assert data.mesh_ids.eq(0).all()
    assert data.inv_pose[..., 3].eq(1).all()


def test_pose_validation_happens_before_cache_or_environment_mutation() -> None:
    data = MeshData.create_cache(1, 1, DeviceCfg("cpu"))
    with pytest.raises(ValueError, match="quaternion"):
        _tetra("bad_pose", pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Geometry values now reject an invalid pose at construction, before a
    # cache can observe it.  The cache therefore remains untouched.
    assert data.get_active_count() == 0
    assert data.get_cached_mesh_names() == []


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable")
def test_mps_cache_rejects_host_tensor_geometry_without_hidden_copy() -> None:
    host = _tetra("host")
    host.vertices = torch.as_tensor(host.vertices)
    host.faces = torch.as_tensor(host.faces)
    data = MeshData.create_cache(1, 1, DeviceCfg("mps"))
    with pytest.raises(ValueError, match="already reside"):
        data.add(host)
