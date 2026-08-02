"""Portable PRM facade routed to curobo-metal's deterministic planner."""

from __future__ import annotations

import time
from typing import List, Optional

import torch

from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.graph_planner.result import GraphPlannerResult
from curobo._src.state.state_joint import JointState
from curobo._src.util.trajectory import TrajInterpolationType
from curobo_metal.ops.graph_planning import GraphPlanningProblem, PersistentRoadmap


class PRMGraphPlanner:
    def __init__(
        self, config: PRMGraphPlannerCfg, scene_collision_checker: Optional[SceneCollision] = None
    ):
        if not isinstance(config, PRMGraphPlannerCfg):
            raise TypeError("config must be PRMGraphPlannerCfg")
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = scene_collision_checker
        self._roadmap = PersistentRoadmap(capacity=config.max_nodes)

    def check_samples_feasibility(self, action_samples):
        if self.config.check_feasibility_fn is not None:
            result = self.config.check_feasibility_fn(action_samples)
            if result.shape != action_samples.shape[:-1] or result.dtype != torch.bool:
                raise ValueError("check_feasibility_fn must return bool shape [N]")
            return result
        # A world-less graph planner is a valid all-free-space configuration.
        return torch.ones(action_samples.shape[:-1], dtype=torch.bool, device=action_samples.device)

    def find_path(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        interpolate_waypoints: bool = True,
        interpolation_steps: int = 100,
        interpolation_type: TrajInterpolationType = TrajInterpolationType.LINEAR,
        validate_interpolated_trajectory: bool = True,
    ) -> GraphPlannerResult:
        del validate_interpolated_trajectory
        if interpolation_type == TrajInterpolationType.BSPLINE_KNOTS_CUDA:
            raise NotImplementedError("CUDA B-spline interpolation is unavailable on CPU/MPS")
        start = x_start.unsqueeze(0) if x_start.ndim == 1 else x_start
        goal = x_goal.unsqueeze(0) if x_goal.ndim == 1 else x_goal
        lower = self.action_bound_lows
        upper = self.action_bound_highs
        if lower is None or upper is None:
            raise ValueError("planner config requires action_lower_bounds and action_upper_bounds")
        begin = time.perf_counter()
        result = self._roadmap.plan(GraphPlanningProblem(
            starts=start,
            goals=goal,
            lower=lower,
            upper=upper,
            validity=self.check_samples_feasibility,
            sample_count=min(self.config.max_nodes - 2, self.config.new_nodes_per_iteration),
            seed=self.config.sampler_seed,
            k_neighbors=self.config.neighbors_per_node,
            connection_radius=self.config.connection_radius,
            edge_step=self.config.edge_step,
            interpolation_step=self.config.edge_step,
        ))
        plans = [path if bool(ok) else None for path, ok in zip(result.roadmap_paths, result.success)]
        lengths = start.new_tensor([metric.path_cost for metric in result.metrics])
        interpolated = None
        if interpolate_waypoints and bool(result.success.any()):
            padded = []
            for path, ok in zip(result.paths, result.success):
                if not bool(ok):
                    padded.append(start.new_zeros((interpolation_steps, start.shape[-1])))
                    continue
                coordinate = torch.linspace(
                    0, path.shape[0] - 1, interpolation_steps,
                    device=path.device, dtype=path.dtype,
                )
                low = coordinate.floor().to(torch.int64).clamp_max(path.shape[0] - 2)
                alpha = (coordinate - low).unsqueeze(-1)
                padded.append(path[low] * (1 - alpha) + path[low + 1] * alpha)
            interpolated = torch.stack(padded)
        return GraphPlannerResult(
            result.success, plans, interpolated, self.joint_names, lengths,
            time.perf_counter() - begin, bool(result.success.all()),
            {"status": result.status, "metrics": result.metrics},
        )

    def get_interpolated_trajectory(
        self, paths: List[torch.Tensor], success: torch.Tensor,
        interpolation_steps: int, interpolation_type: TrajInterpolationType,
    ):
        starts = torch.stack([path[0] for path in paths])
        goals = torch.stack([path[-1] for path in paths])
        result = self.find_path(
            starts, goals, True, interpolation_steps, interpolation_type
        )
        result.success &= success
        return result.interpolated_waypoints

    def reset_buffer(self): self._roadmap.reset()
    def reset_seed(self): self._roadmap.reset()

    def extend_roadmap_with_random_samples(
        self, num_samples: int, neighbors_per_node: int = 10
    ):
        """Reset portable roadmap state and return deterministic free samples.

        Unlike upstream this does not mutate a CUDA graph buffer; callers can
        feed the returned samples to their normal planner invocation.
        """
        del neighbors_per_node
        count = int(num_samples)
        generator = torch.Generator(device="cpu").manual_seed(self.config.sampler_seed)
        values = torch.rand((count, self.action_dim), generator=generator, dtype=self.action_bound_lows.dtype)
        values = values.to(self.action_bound_lows.device)
        return values * (self.action_bound_highs - self.action_bound_lows) + self.action_bound_lows

    def extend_roadmap_with_ellipsoidal_samples(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        max_sampling_radius: torch.Tensor,
        num_samples: int,
        neighbors_per_node: int = 5,
    ):
        del neighbors_per_node
        samples = self.extend_roadmap_with_random_samples(num_samples)
        radius = max_sampling_radius
        midpoint = (x_start + x_goal) * 0.5
        return torch.maximum(torch.minimum(
            midpoint + (samples - midpoint) * radius,
            self.action_bound_highs,
        ), self.action_bound_lows)

    def reset_cuda_graph(self):
        raise NotImplementedError(
            "CUDA graph capture has no Metal equivalent; reset_buffer controls portable caches"
        )

    def get_all_rollout_instances(self) -> List[RobotRollout]:
        return []

    def warmup(self, num_warmup_iterations: int = 10, max_batch_size: int = 4):
        del num_warmup_iterations, max_batch_size
        return None

    @property
    def action_dim(self) -> int: return int(self.action_bound_lows.numel())
    @property
    def action_bound_lows(self) -> torch.Tensor: return self.config.action_lower_bounds
    @property
    def action_bound_highs(self) -> torch.Tensor: return self.config.action_upper_bounds
    @property
    def n_nodes(self) -> int: return 0
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
    def kinematics(self) -> Kinematics:
        raise NotImplementedError("use curobo.kinematics.Kinematics explicitly")
    @property
    def transition_model(self) -> RobotStateTransition:
        raise NotImplementedError("CUDA rollout transition models are not used by portable PRM")

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        return self.kinematics.compute_kinematics(state)


__all__ = ["PRMGraphPlanner"]
