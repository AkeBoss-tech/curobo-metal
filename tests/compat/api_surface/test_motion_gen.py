from pathlib import Path

import pytest
import torch

from curobo_metal.api_compat import (
    IKSolverConfig, InterpolationType, MotionGen, MotionGenPlanConfig,
    OptimizerType, TrajOptSolverConfig, UnsupportedCompatOption,
    compile_motion_gen_config,
)
from curobo_metal.motion_gen import JointState, MotionGenConfig


FIXTURE = Path(__file__).parents[2] / "fixtures" / "two_link_curobo_v2.json"


def _base(device="cpu"):
    return MotionGenConfig.load_from_robot_config(FIXTURE, device=device)


def test_compile_wires_supported_fields():
    cfg = compile_motion_gen_config(
        _base(),
        ik=IKSolverConfig(num_seeds=3, max_iterations=7, random_seed=11),
        trajopt=TrajOptSolverConfig(
            steps=8, dt=.1, max_iterations=9, interpolation_dt=.05,
        ),
    )
    assert (cfg.num_ik_seeds, cfg.max_ik_iterations, cfg.graph_seed) == (3, 7, 11)
    assert (cfg.steps, cfg.dt, cfg.max_trajectory_iterations, cfg.interpolation_dt) == (8, .1, 9, .05)


@pytest.mark.parametrize("kwargs, match", [
    ({"ik": IKSolverConfig(optimizer=OptimizerType.PARTICLE)}, "LBFGS"),
    ({"ik": IKSolverConfig(retract_config=(0.0, 0.0))}, "retract"),
    ({"trajopt": TrajOptSolverConfig(interpolation_type=InterpolationType.BSPLINE)}, "linear"),
])
def test_compile_rejects_precisely(kwargs, match):
    with pytest.raises(UnsupportedCompatOption, match=match):
        compile_motion_gen_config(_base(), **kwargs)


def test_cpu_warmup_retry_and_graph_lifecycle():
    planner = MotionGen(_base())
    assert planner.warmup(enable_graph=False)
    q = (planner.config.lower + planner.config.upper) * .5
    result = planner.plan_single_js(
        JointState(q), JointState(q.clone()),
        MotionGenPlanConfig(max_attempts=2, enable_graph_attempt=2),
    )
    assert bool(result.success.item())
    planner.reset_graph()
    assert planner._graph_generation == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_behavior():
    planner = MotionGen(_base("mps"))
    q = (planner.config.lower + planner.config.upper) * .5
    assert bool(planner.plan_single_js(q, q, enable_graph=False).success.item())
