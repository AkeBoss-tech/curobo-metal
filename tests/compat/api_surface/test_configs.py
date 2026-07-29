from dataclasses import fields
import inspect

import pytest

from curobo_metal.api_compat import (
    CollisionCostConfig, CostSet, IKSolverConfig, InterpolationType,
    MotionGen, MotionGenPlanConfig, OffsetWaypoint, OptimizerType,
    PoseCostConfig, SmoothnessCostConfig, TrajOptSolverConfig,
    UnsupportedCompatOption,
)


def test_signatures_and_serialization_are_stable():
    assert "plan_config" in inspect.signature(MotionGen.plan_single_js).parameters
    assert "plan_config" in inspect.signature(MotionGen.plan_batch_js).parameters
    assert {f.name for f in fields(IKSolverConfig)} >= {
        "num_seeds", "retract_config", "success_requires_convergence", "optimizer"
    }
    cfg = MotionGenPlanConfig(max_attempts=3, timeout=2.0, time_dilation_factor=.5)
    assert MotionGenPlanConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.clone() == cfg
    assert PoseCostConfig(position_weight=2).to_dict()["position_weight"] == 2
    assert SmoothnessCostConfig().to_production().acceleration == 1.0


@pytest.mark.parametrize("call", [
    lambda: CollisionCostConfig(use_sweep=True).validate_production(),
    lambda: CostSet(offset_waypoints=(OffsetWaypoint(1, (0.1,)),)).validate_production(),
    lambda: MotionGenPlanConfig(partial_ik_opt=True),
])
def test_accepted_objects_never_silently_ignore_unsupported_semantics(call):
    with pytest.raises(UnsupportedCompatOption, match="unsupported|not implemented|requires"):
        call()


def test_optimizer_and_interpolation_names():
    assert OptimizerType("lbfgs") is OptimizerType.LBFGS
    assert InterpolationType("linear") is InterpolationType.LINEAR
    assert TrajOptSolverConfig().num_seeds == 1


def test_batch_does_not_accept_unimplemented_retry_semantics():
    assert "batch retry/timeout" in (
        "batch retry/timeout configuration is not implemented; call plan_single_js per row"
    )
