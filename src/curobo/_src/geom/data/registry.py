"""Obstacle data module registry."""

OBSTACLE_SDF_MODULES = [
    "curobo._src.geom.data.data_cuboid",
    "curobo._src.geom.data.data_mesh",
    "curobo._src.geom.data.data_voxel",
]
__all__=["OBSTACLE_SDF_MODULES"]
