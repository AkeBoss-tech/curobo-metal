"""Pinned-V2 observable behavior for tensor-free solver problem metadata."""

import pytest
import torch

from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.solve_mode import SolveMode
from curobo._src.solver.solve_state import MotionPlanSolveState, SolveState
from curobo._src.types.device_cfg import DeviceCfg


def test_seed_precedence_and_batch_accounting_match_v2():
    state = SolveState(
        SolveMode.BATCH, batch_size=3, num_envs=1,
        num_ik_seeds=2, num_trajopt_seeds=4, num_graph_seeds=5,
    )
    assert state.num_seeds == 2
    assert state.batch_mode
    assert not state.multi_env
    assert state.get_batch_size() == 6
    assert state.get_ik_batch_size() == 6
    assert state.get_trajopt_batch_size() == 12


def test_explicit_single_batch_mode_is_preserved_and_env_mode_is_derived():
    state = SolveState(
        SolveMode.SINGLE, batch_size=1, num_envs=2, num_seeds=1, batch_mode=True,
    )
    assert state.batch_mode
    assert state.multi_env

    forced = SolveState(SolveMode.SINGLE, batch_size=2, num_envs=1, num_seeds=1)
    assert forced.batch_mode
    assert not forced.multi_env


def test_clone_preserves_all_metadata_and_upstream_tool_frame_aliasing():
    frames = ["tool0", "tool1"]
    original = SolveState(
        SolveMode.MULTI_ENV, batch_size=2, num_envs=2, num_goalset=3,
        num_seeds=7, num_ik_seeds=2, num_graph_seeds=4, num_trajopt_seeds=5,
        tool_frames=frames,
    )
    copied = original.clone()
    assert copied == original
    assert copied is not original
    assert copied.tool_frames is frames

    motion = MotionPlanSolveState(SolveMode.MULTI_ENV, copied, original)
    assert motion.ik_solve_state.get_batch_size() == 14
    assert motion.trajopt_solve_state.get_trajopt_batch_size() == 10


def test_generic_batch_size_requires_resolved_seed_count_but_specific_paths_are_zero():
    state = SolveState(SolveMode.SINGLE, batch_size=1, num_envs=1)
    assert state.get_ik_batch_size() == 0
    assert state.get_trajopt_batch_size() == 0
    with pytest.raises(TypeError):
        state.get_batch_size()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_solve_state_drives_mps_goal_index_shape_without_cpu_fallback():
    """The tensor-free descriptor yields the same seeded layout on MPS."""
    state = SolveState(SolveMode.MULTI_ENV, 2, 2, num_goalset=3, num_ik_seeds=4)
    goal = GoalManager(DeviceCfg(device=torch.device("mps"))).create_goal_buffer(state)
    assert goal.get_index_size() == state.get_batch_size() == 8
    assert goal.idxs_link_pose.device.type == "mps"
    assert goal.idxs_env.device.type == "mps"
    assert goal.idxs_env[:, 0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
