from .data_cuboid import CuboidData
from .data_mesh import MeshData
from .data_voxel import VoxelData
OBSTACLE_SDF_MODULES={"cuboid":CuboidData,"mesh":MeshData,"voxel":VoxelData}
__all__=["OBSTACLE_SDF_MODULES"]
