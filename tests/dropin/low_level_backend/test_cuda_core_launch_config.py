from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import (
    DynamicsLaunchCfg,
)
from curobo._src.curobolib.backends.cuda_core_backend.geometry_config import GeometryKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.kernel_config import CudaCoreKernelCfg
from curobo._src.curobolib.backends.cuda_core_backend.kinematics_config import (
    KinematicsLaunchCfg,
)
from curobo._src.curobolib.backends.cuda_core_backend.pba_config import PBALaunchCfg


def test_cuda_core_release_flags_match_upstream_contract() -> None:
    flags = CudaCoreKernelCfg("geometry").get_compile_flags()
    assert flags == [
        "-O3",
        "--ftz=true",
        "--fmad=true",
        "--prec-div=false",
        "--prec-sqrt=false",
        "--generate-line-info",
    ]


def test_cuda_core_debug_flags_match_upstream_contract() -> None:
    flags = CudaCoreKernelCfg("geometry").get_compile_flags(debug=True)
    assert flags == ["-G", "-g", "--generate-line-info", "--device-debug"]


def test_backend_kernel_paths_and_pba_grid_are_compatible() -> None:
    geometry = GeometryKernelCfg()
    assert geometry.get_kernel_files("self_collision") == [
        "self_collision/self_collision_kernel.cuh"
    ]
    assert geometry.get_kernel_files("unknown") == []
    config = PBALaunchCfg.flood_z(65, 9)
    assert config.grid == (3, 3)
    assert config.block == (32, 4)


def test_dynamics_backward_memory_separates_block_and_batch_storage() -> None:
    config = DynamicsLaunchCfg.calculate_backward_config(
        batch_size=2,
        num_links=10,
        max_batches_per_block=2,
        max_shared_mem=48 * 1024,
    )
    # One 12-float/link block allocation plus 30 floats/link for each batch.
    assert config.shmem_size == 10 * 12 * 4 + 2 * 10 * 30 * 4
    assert config.grid == 1
    assert config.block == 2


def test_kinematics_backward_configuration_reports_reduction_contract() -> None:
    config, threads_per_batch, use_warp_reduce, max_joints = (
        KinematicsLaunchCfg.calculate_backward_config(
            batch_size=2,
            num_links=10,
            num_spheres=64,
            n_tool_frames=1,
            n_joints=7,
        )
    )
    assert use_warp_reduce is True
    assert threads_per_batch == 32
    assert max_joints == 16
    assert config.grid == 1
    assert config.block == 64

