"""Portable PRM facade routed to curobo-metal's deterministic graph planner.

The upstream implementation owns CUDA rollout and graph buffers.  This module
keeps the useful *observable* PRM lifecycle on CPU/MPS: explicitly extended
roadmaps are deterministic, collision-checked at query time, resettable, and
are really used by subsequent calls to :meth:`find_path`.  It deliberately
does not emulate CUDA graph capture, Warp steering kernels, or analytic CCD.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional

import torch

from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.graph_planner.result import GraphPlannerResult
from curobo._src.state.state_joint import JointState
from curobo._src.util.trajectory import TrajInterpolationType, linear_smooth
from curobo_metal.ops.graph_planning import GraphPlanningProblem, PersistentRoadmap


class PRMGraphPlanner:
    """Pinned-V2 shaped PRM planner over the portable graph-planning backend.

    ``extend_roadmap_*`` is intentionally stateful, as it is upstream: added
    vertices are retained until ``reset_buffer`` and become the candidate
    samples for later queries.  Candidate validity is always recomputed by the
    production graph operation, so mutable collision callbacks cannot make a
    previously accepted vertex unsafe by accident.
    """

    def __init__(
        self, config: PRMGraphPlannerCfg, scene_collision_checker: Optional[Any] = None
    ):
        if not isinstance(config, PRMGraphPlannerCfg):
            raise TypeError("config must be PRMGraphPlannerCfg")
        if config.max_nodes < 2:
            raise ValueError("max_nodes must be at least two")
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = scene_collision_checker
        self._roadmap = PersistentRoadmap(capacity=config.max_nodes)
        self._roadmap_samples: torch.Tensor | None = None
        self._generation = 0
        self._reset_sampler()

    def _reset_sampler(self) -> None:
        self._sampler = torch.Generator(device="cpu")
        self._sampler.manual_seed(int(self.config.sampler_seed))

    def _validate_actions(self, action_samples: torch.Tensor, *, name: str = "action_samples") -> None:
        if not isinstance(action_samples, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if action_samples.ndim != 2:
            raise ValueError(f"{name} must be a 2D tensor (batch_size, action_dim)")
        if action_samples.shape[-1] != self.action_dim:
            raise ValueError(f"{name} must end in action_dim={self.action_dim}")
        if action_samples.device != self.action_bound_lows.device:
            raise ValueError(f"{name} must be on {self.action_bound_lows.device}")
        if action_samples.dtype != self.action_bound_lows.dtype:
            raise ValueError(f"{name} must use {self.action_bound_lows.dtype}")
        if not bool(torch.isfinite(action_samples).all().item()):
            raise ValueError(f"{name} must contain finite values")

    def check_samples_feasibility(self, action_samples: torch.Tensor) -> torch.Tensor:
        """Return a device-resident boolean feasibility mask of shape ``[N]``."""
        self._validate_actions(action_samples)
        if self.config.check_feasibility_fn is not None:
            result = self.config.check_feasibility_fn(action_samples)
            if not isinstance(result, torch.Tensor):
                raise TypeError("check_feasibility_fn must return a torch tensor")
            if result.shape != action_samples.shape[:-1] or result.dtype != torch.bool:
                raise ValueError("check_feasibility_fn must return bool shape [N]")
            if result.device != action_samples.device:
                raise ValueError("check_feasibility_fn must return a mask on the input device")
            return result
        # A world-less graph planner is a valid all-free-space configuration.
        return torch.ones(action_samples.shape[:-1], dtype=torch.bool, device=action_samples.device)

    def _append_samples(self, samples: torch.Tensor) -> None:
        if samples.numel() == 0:
            return
        self._validate_actions(samples, name="roadmap samples")
        existing = 0 if self._roadmap_samples is None else self._roadmap_samples.shape[0]
        if existing + samples.shape[0] > self.config.max_nodes:
            raise ValueError(
                "graph node buffer capacity exceeded; call reset_buffer or request fewer samples"
            )
        self._roadmap_samples = (
            samples.clone()
            if self._roadmap_samples is None
            else torch.cat((self._roadmap_samples, samples), dim=0)
        )
        # ``_sample`` caches candidates by query shape.  Invalidate those
        # shape entries without discarding the PersistentRoadmap object itself.
        self._roadmap.cache.reset()
        self._generation += 1

    def _random_samples(self, count: int) -> torch.Tensor:
        if not isinstance(count, int) or count < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        if count == 0:
            return self.action_bound_lows.new_empty((0, self.action_dim))
        unit = torch.rand(
            (count, self.action_dim), generator=self._sampler, dtype=self.action_bound_lows.dtype,
            device="cpu",
        ).to(self.action_bound_lows.device)
        return self.action_bound_lows + unit * (self.action_bound_highs - self.action_bound_lows)

    def _feasible_random_samples(self, count: int) -> torch.Tensor:
        """Match upstream rejection sampling without leaking infeasible rows."""
        if count == 0:
            return self.action_bound_lows.new_empty((0, self.action_dim))
        requested = max(count, count * max(1, int(self.config.sample_rejection_ratio)))
        candidates = self._random_samples(requested)
        feasible = self.check_samples_feasibility(candidates)
        return candidates[feasible][:count]

    def _ellipsoidal_samples(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        max_sampling_radius: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        self._validate_actions(x_start.reshape(1, -1), name="x_start")
        self._validate_actions(x_goal.reshape(1, -1), name="x_goal")
        radius = torch.as_tensor(
            max_sampling_radius, dtype=x_start.dtype, device=x_start.device
        )
        if radius.numel() != 1 or not bool(torch.isfinite(radius).all().item()) or float(radius) < 0:
            raise ValueError("max_sampling_radius must be one finite nonnegative scalar")
        if count == 0:
            return x_start.new_empty((0, self.action_dim))
        # Uniformly sample a unit ball (rather than a cube) and map its radius
        # around the start/goal midpoint.  The exact CUDA SVD/Householder
        # kernels are intentionally not claimed; this is the documented
        # differentiable portable approximation used by NodeSamplingStrategy.
        candidate_count = max(count, count * max(1, int(self.config.sample_rejection_ratio)))
        direction = self._random_samples(candidate_count)
        direction = 2 * (direction - self.action_bound_lows) / (
            self.action_bound_highs - self.action_bound_lows
        ).clamp_min(torch.finfo(direction.dtype).eps) - 1
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1)
        unit = direction / norm
        midpoint = (x_start + x_goal) * 0.5
        points = midpoint + unit * radius
        points = torch.maximum(torch.minimum(points, self.action_bound_highs), self.action_bound_lows)
        return points[self.check_samples_feasibility(points)][:count]

    @staticmethod
    def _sample_cache_key(problem: GraphPlanningProblem, batch_index: int, dof: int) -> tuple[object, ...]:
        """Mirror the production cache key so persistent PRM nodes are consumed."""
        return (
            "graph_samples", problem.sample_count, problem.seed, batch_index, dof, problem.sampling,
            tuple(problem.lower.shape), str(problem.lower.dtype), problem.lower.device.type,
            tuple(float(x) for x in problem.lower.detach().cpu()),
            tuple(float(x) for x in problem.upper.detach().cpu()),
        )

    def _inject_roadmap_samples(self, problem: GraphPlanningProblem, batch: int) -> None:
        if self._roadmap_samples is None:
            return
        for index in range(batch):
            entry = self._roadmap.cache.acquire(
                self._sample_cache_key(problem, index, self.action_dim)
            )
            entry["samples"] = self._roadmap_samples

    def _candidate_count(self) -> int:
        if self._roadmap_samples is not None:
            return self._roadmap_samples.shape[0]
        # Reserve the terminal pair from the configured buffer limit.
        return min(max(0, self.config.max_nodes - 2), self.config.new_nodes_per_iteration)

    def _interpolate_paths(
        self,
        paths: List[torch.Tensor | None],
        success: torch.Tensor,
        interpolation_steps: int,
        interpolation_type: TrajInterpolationType,
    ) -> torch.Tensor:
        if not isinstance(interpolation_steps, int) or interpolation_steps < 1:
            raise ValueError("interpolation_steps must be a positive integer")
        supported = (
            TrajInterpolationType.LINEAR,
            TrajInterpolationType.CUBIC,
            TrajInterpolationType.QUINTIC,
        )
        if interpolation_type not in supported:
            raise ValueError(
                f"Unsupported interpolation type: {interpolation_type}. "
                f"Supported types: {[item.value for item in supported]}"
            )
        output = self.action_bound_lows.new_zeros((len(paths), interpolation_steps, self.action_dim))
        for batch_index, ok in enumerate(success.detach().cpu().tolist()):
            if not ok:
                continue
            path = paths[batch_index]
            if path is None or path.ndim != 2 or path.shape != (path.shape[0], self.action_dim):
                raise ValueError("successful graph paths must have shape [waypoint, action_dim]")
            if path.shape[0] < 2:
                raise ValueError("successful graph paths must include at least start and goal")
            # Preserve the pinned public interpolation variants while keeping
            # the tensor on its source device. ``linear_smooth`` centralizes
            # the matching portable cubic/quintic stencils.
            values = [
                linear_smooth(path[:, joint].detach().cpu().numpy(), n=interpolation_steps,
                              kind=interpolation_type)
                for joint in range(self.action_dim)
            ]
            output[batch_index] = torch.stack([
                torch.as_tensor(value, dtype=output.dtype, device=output.device)
                for value in values
            ], dim=-1)
        return output

    def find_path(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        interpolate_waypoints: bool = True,
        interpolation_steps: int = 100,
        interpolation_type: TrajInterpolationType = TrajInterpolationType.LINEAR,
        validate_interpolated_trajectory: bool = True,
    ) -> GraphPlannerResult:
        self._validate_actions(x_start, name="x_start")
        self._validate_actions(x_goal, name="x_goal")
        if x_start.shape != x_goal.shape:
            raise ValueError("x_start and x_goal must have the same shape")
        if not isinstance(interpolation_type, TrajInterpolationType):
            raise TypeError("interpolation_type must be a TrajInterpolationType")
        if self.n_nodes > self.config.max_nodes * 0.75:
            # Match upstream's bounded lifecycle: terminal-query growth must
            # never make an old graph silently exceed its declared capacity.
            self.reset_buffer()
        begin = time.perf_counter()
        endpoints = self.check_samples_feasibility(torch.cat((x_start, x_goal), dim=0))
        if not bool(endpoints.all().item()):
            batch = x_start.shape[0]
            return GraphPlannerResult(
                torch.zeros(batch, dtype=torch.bool, device=x_start.device),
                [None] * batch, None, self.joint_names,
                x_start.new_full((batch,), float("inf")), time.perf_counter() - begin,
                False, "Start or End state in collision",
            )
        problem = GraphPlanningProblem(
            starts=x_start,
            goals=x_goal,
            lower=self.action_bound_lows,
            upper=self.action_bound_highs,
            validity=self.check_samples_feasibility,
            sample_count=self._candidate_count(),
            seed=int(self.config.sampler_seed),
            k_neighbors=int(self.config.neighbors_per_node),
            connection_radius=self.config.connection_radius,
            edge_step=float(self.config.edge_step),
            interpolation_step=float(self.config.edge_step),
            execution_cache=self._roadmap.cache,
        )
        self._inject_roadmap_samples(problem, x_start.shape[0])
        backend = self._roadmap.plan(problem)
        plans: List[torch.Tensor | None] = [
            path if bool(ok) else None for path, ok in zip(backend.roadmap_paths, backend.success)
        ]
        lengths = x_start.new_tensor([metric.path_cost for metric in backend.metrics])
        interpolated = None
        success = backend.success.clone()
        if interpolate_waypoints and bool(success.any()):
            interpolated = self._interpolate_paths(
                plans, success, interpolation_steps, interpolation_type
            )
            if validate_interpolated_trajectory:
                mask = self.check_samples_feasibility(interpolated.reshape(-1, self.action_dim))
                success &= mask.reshape(x_start.shape[0], interpolation_steps).all(dim=1)
        return GraphPlannerResult(
            success, plans, interpolated, self.joint_names, lengths,
            time.perf_counter() - begin, bool(success.all()),
            {"status": backend.status, "metrics": backend.metrics, "generation": self._generation},
        )

    def get_interpolated_trajectory(
        self, paths: List[torch.Tensor | None], success: torch.Tensor,
        interpolation_steps: int, interpolation_type: TrajInterpolationType,
    ) -> torch.Tensor:
        if not isinstance(success, torch.Tensor) or success.dtype != torch.bool or success.ndim != 1:
            raise ValueError("success must be a one-dimensional bool tensor")
        if len(paths) != success.shape[0]:
            raise ValueError("paths and success must have matching batch size")
        return self._interpolate_paths(paths, success, interpolation_steps, interpolation_type)

    def reset_buffer(self) -> None:
        self._roadmap.reset()
        self._roadmap_samples = None
        self._generation += 1

    def reset_seed(self) -> None:
        """Reset sampling streams while preserving the explicitly built roadmap."""
        self._reset_sampler()
        # Fresh random query samples must be regenerated after seed reset.
        self._roadmap.cache.reset()

    def extend_roadmap_with_random_samples(
        self, num_samples: int, neighbors_per_node: int = 10
    ) -> None:
        if not isinstance(neighbors_per_node, int) or neighbors_per_node <= 0:
            raise ValueError("neighbors_per_node must be positive")
        self._append_samples(self._feasible_random_samples(num_samples))

    def extend_roadmap_with_ellipsoidal_samples(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        max_sampling_radius: torch.Tensor,
        num_samples: int,
        neighbors_per_node: int = 5,
    ) -> None:
        if not isinstance(neighbors_per_node, int) or neighbors_per_node <= 0:
            raise ValueError("neighbors_per_node must be positive")
        self._append_samples(
            self._ellipsoidal_samples(x_start, x_goal, max_sampling_radius, num_samples)
        )

    def reset_cuda_graph(self) -> None:
        raise NotImplementedError(
            "CUDA graph capture has no Metal equivalent; reset_buffer controls portable caches"
        )

    def get_all_rollout_instances(self) -> List[Any]:
        # There is no CUDA RobotRollout.  Returning an empty list is the
        # pinned-compatible capability signal for portable direct PRM users.
        return []

    def warmup(self, num_warmup_iterations: int = 10, max_batch_size: int = 4) -> None:
        if not isinstance(num_warmup_iterations, int) or num_warmup_iterations < 0:
            raise ValueError("num_warmup_iterations must be a nonnegative integer")
        if not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        # Warmup has no CUDA capture effect.  It does execute the production
        # planner and returns with a pristine portable roadmap/seed state.
        actions = self._feasible_random_samples(num_warmup_iterations * 2 * max_batch_size)
        for index in range(num_warmup_iterations):
            start = actions[index:index + max_batch_size]
            goal = actions[num_warmup_iterations + index:num_warmup_iterations + index + max_batch_size]
            if start.shape[0] == max_batch_size and goal.shape[0] == max_batch_size:
                self.find_path(start, goal)
            self.reset_buffer()
        self.reset_seed()

    @property
    def action_dim(self) -> int:
        lower = self.action_bound_lows
        if lower is None:
            raise ValueError("planner config requires action_lower_bounds and action_upper_bounds")
        return int(lower.numel())

    @property
    def action_bound_lows(self) -> torch.Tensor:
        if self.config.action_lower_bounds is None or self.config.action_upper_bounds is None:
            raise ValueError("planner config requires action_lower_bounds and action_upper_bounds")
        return self.config.action_lower_bounds

    @property
    def action_bound_highs(self) -> torch.Tensor:
        if self.config.action_lower_bounds is None or self.config.action_upper_bounds is None:
            raise ValueError("planner config requires action_lower_bounds and action_upper_bounds")
        return self.config.action_upper_bounds

    @property
    def n_nodes(self) -> int:
        return 0 if self._roadmap_samples is None else int(self._roadmap_samples.shape[0])

    @property
    def cspace_distance_weight(self) -> torch.Tensor:
        return torch.ones(self.action_dim, **self.device_cfg.as_torch_dict())

    @property
    def joint_names(self) -> List[str]:
        if self.config.robot_config is None:
            return [f"joint_{index}" for index in range(self.action_dim)]
        return list(self.config.robot_config.kinematics.joint_names)

    @property
    def default_joint_state(self) -> JointState:
        if self.config.robot_config is None:
            position = (self.action_bound_lows + self.action_bound_highs) * 0.5
        else:
            position = self.config.robot_config.kinematics.retract_config
        return JointState.from_position(position, self.joint_names)

    @property
    def kinematics(self) -> Any:
        if self.config.robot_config is None:
            raise NotImplementedError("PRM kinematics requires a portable RobotCfg")
        from curobo.kinematics import Kinematics

        return Kinematics(self.config.robot_config.kinematics, self.device_cfg)

    @property
    def transition_model(self) -> Any:
        raise NotImplementedError("CUDA rollout transition models are not used by portable PRM")

    def compute_kinematics(self, state: JointState) -> Any:
        return self.kinematics.compute_kinematics(state)


__all__ = ["PRMGraphPlanner"]
