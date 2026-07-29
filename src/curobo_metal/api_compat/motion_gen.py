"""Strict adapter around the production :mod:`curobo_metal.motion_gen` facade."""

from __future__ import annotations

from dataclasses import replace
import time

from curobo_metal.motion_gen import MotionGen as _MotionGen
from curobo_metal.motion_gen import MotionGenConfig

from .config import (
    GraphSolverConfig, IKSolverConfig, InterpolationType, MotionGenPlanConfig,
    OptimizerType, TrajOptSolverConfig,
)
from .cost import UnsupportedCompatOption


class MotionGen(_MotionGen):
    def warmup(self, enable_graph: bool = True, warmup_js_trajopt: bool = True, **kwargs: object) -> bool:
        if not warmup_js_trajopt:
            raise UnsupportedCompatOption("warmup_js_trajopt=False is not implemented")
        return super().warmup(enable_graph=enable_graph, **kwargs)

    def reset_graph(self) -> None:
        """Discard portable shape-keyed optimizer and roadmap execution state."""
        self._graph_cache.reset()
        self._optimizer_cache.reset()
        self._graph_generation = getattr(self, "_graph_generation", 0) + 1

    clear_graph_cache = reset_graph

    def plan_single_js(self, start_state, goal_state, plan_config: MotionGenPlanConfig | None = None,
                       *, enable_graph: bool | None = None):
        cfg = plan_config or MotionGenPlanConfig()
        graph = cfg.enable_graph if enable_graph is None else enable_graph
        if not cfg.enable_opt:
            raise UnsupportedCompatOption("enable_opt=False graph-only result adaptation is not implemented")
        started = time.monotonic()
        result = None
        for attempt in range(1, cfg.max_attempts + 1):
            if cfg.timeout is not None and time.monotonic() - started >= cfg.timeout:
                raise TimeoutError(f"motion generation exceeded timeout={cfg.timeout}s")
            result = super().plan_single_js(
                start_state, goal_state,
                enable_graph=graph and attempt >= cfg.enable_graph_attempt,
            )
            if bool(result.success.item()):
                if cfg.time_dilation_factor != 1.0:
                    result = replace(
                        result,
                        interpolation_dt=result.interpolation_dt / cfg.time_dilation_factor,
                    )
                return result
        assert result is not None
        return result

    plan_single_joint_space = plan_single_js

    def plan_batch_js(self, start_state, goal_state, plan_config: MotionGenPlanConfig | None = None,
                      *, enable_graph: bool | None = None):
        cfg = plan_config or MotionGenPlanConfig()
        graph = cfg.enable_graph if enable_graph is None else enable_graph
        if cfg.max_attempts != 1 or cfg.timeout is not None or cfg.enable_graph_attempt != 1:
            raise UnsupportedCompatOption(
                "batch retry/timeout configuration is not implemented; call plan_single_js per row"
            )
        if cfg.time_dilation_factor != 1.0:
            raise UnsupportedCompatOption("batch time_dilation_factor is not implemented")
        if not cfg.enable_opt:
            raise UnsupportedCompatOption("enable_opt=False graph-only result adaptation is not implemented")
        # The production batch implementation preserves row order and invokes
        # this adapter's strict single method for every row.
        return super().plan_batch_js(start_state, goal_state, enable_graph=graph)

    plan_batch_joint_space = plan_batch_js


def compile_motion_gen_config(base: MotionGenConfig, *, ik: IKSolverConfig | None = None,
                              trajopt: TrajOptSolverConfig | None = None,
                              graph: GraphSolverConfig | None = None) -> MotionGenConfig:
    ik = ik or IKSolverConfig()
    trajopt = trajopt or TrajOptSolverConfig()
    graph = graph or GraphSolverConfig()
    ik_optimizer = OptimizerType(ik.optimizer)
    trajectory_optimizer = OptimizerType(trajopt.optimizer)
    if ik.retract_config is not None:
        raise UnsupportedCompatOption("explicit retract_config seed injection is not implemented")
    if not ik.success_requires_convergence:
        raise UnsupportedCompatOption("success_requires_convergence=False is not implemented")
    if trajopt.num_seeds != 1:
        raise UnsupportedCompatOption("explicit num_trajopt_seeds is not implemented")
    if trajopt.interpolation_type is not InterpolationType.LINEAR:
        raise UnsupportedCompatOption("production interpolation supports only linear")
    trajopt.costs.validate_production()
    if not isinstance(trajopt.costs.bounds.weight, (int, float)):
        raise UnsupportedCompatOption("per-joint bounds weights are not implemented")
    return replace(
        base,
        num_ik_seeds=ik.num_seeds,
        max_ik_iterations=ik.max_iterations,
        position_tolerance=ik.position_tolerance,
        rotation_tolerance=ik.rotation_tolerance,
        ik_optimizer=ik_optimizer.value,
        trajectory_optimizer=trajectory_optimizer.value,
        optimizer_seed=ik.random_seed,
        graph_sample_count=graph.sample_count,
        graph_seed=graph.seed,
        graph_k_neighbors=graph.k_neighbors,
        graph_edge_step=graph.edge_step,
        graph_cache_size=graph.cache_size,
        steps=trajopt.steps,
        dt=trajopt.dt,
        max_trajectory_iterations=trajopt.max_iterations,
        interpolation_dt=trajopt.interpolation_dt,
        trajectory_weights=trajopt.costs.smoothness.to_production(
            joint_limit=float(trajopt.costs.bounds.weight),
        ),
    )
