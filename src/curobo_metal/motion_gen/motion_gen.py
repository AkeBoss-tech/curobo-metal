"""cuRoboV2-style end-to-end MotionGen facade."""

from __future__ import annotations

from dataclasses import replace

import torch

from curobo_metal.ops.costs import robot_collision_cost
from curobo_metal.ops.graph_planning import (
    GraphPlanningProblem,
    paths_to_trajectory_seeds,
    plan_graph,
)
from curobo_metal.ops.ik import IKProblem, solve_ik
from curobo_metal.ops.kinematics import forward_kinematics
from curobo_metal.ops.trajectory import (
    TrajectoryProblem,
    interpolate_trajectory,
    optimize_trajectory,
    trajectory_metrics,
)

from .config import MotionGenConfig
from .types import (
    JointState,
    MotionGenMetrics,
    MotionGenResult,
    MotionGenStatus,
    Pose,
    UnsupportedMotionGenFeature,
)


class MotionGen:
    """Compose portable IK, graph planning, and trajectory optimization."""

    def __init__(self, config: MotionGenConfig) -> None:
        if not isinstance(config, MotionGenConfig):
            raise TypeError("config must be MotionGenConfig")
        self.config = config
        self._warmed_up = False

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.config.joint_names

    def warmup(self, enable_graph: bool = True, **kwargs: object) -> bool:
        if kwargs:
            raise UnsupportedMotionGenFeature(
                f"unsupported warmup options: {sorted(kwargs)}"
            )
        midpoint = (self.config.lower + self.config.upper) * 0.5
        delta = torch.zeros_like(midpoint)
        delta[0] = min(0.05, float((self.config.upper[0] - midpoint[0]).item()))
        self.plan_single_js(
            JointState(midpoint), JointState(midpoint + delta),
            enable_graph=enable_graph,
        )
        self._warmed_up = True
        return True

    def _position(self, state: JointState | torch.Tensor, *, batch: bool) -> torch.Tensor:
        value = state.position if isinstance(state, JointState) else state
        if not isinstance(value, torch.Tensor):
            raise TypeError("joint state position must be a torch.Tensor")
        expected = 2 if batch else 1
        if value.ndim != expected or value.shape[-1] != self.config.chain.dof:
            shape = f"[B,{self.config.chain.dof}]" if batch else f"[{self.config.chain.dof}]"
            raise ValueError(f"joint state position must have shape {shape}")
        if value.device.type != self.config.device.type or value.dtype != self.config.dtype:
            raise ValueError("joint state must match MotionGen config device and dtype")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("joint state must contain finite values")
        return value

    def _state_valid(self, q: torch.Tensor) -> bool:
        if bool(((q < self.config.lower) | (q > self.config.upper)).any().item()):
            return False
        if self.config.collision_model is None:
            return True
        transforms = forward_kinematics(self.config.chain, q).transforms
        _, clearance = robot_collision_cost(transforms, self.config.collision_model)
        return bool((clearance >= 0).all().item())

    def _failure(self, status: MotionGenStatus, *, ik: object = None, graph: object = None) -> MotionGenResult:
        return MotionGenResult(
            torch.tensor(False, device=self.config.device), status, None, None,
            self.config.interpolation_dt, None, ik_result=ik, graph_result=graph,
        )

    def plan_single_js(
        self,
        start_state: JointState | torch.Tensor,
        goal_state: JointState | torch.Tensor,
        *,
        enable_graph: bool = True,
    ) -> MotionGenResult:
        start = self._position(start_state, batch=False)
        goal = self._position(goal_state, batch=False)
        if not self._state_valid(start):
            return self._failure(MotionGenStatus.INVALID_START)
        if bool(((goal < self.config.lower) | (goal > self.config.upper)).any().item()):
            return self._failure(MotionGenStatus.INVALID_GOAL)
        problem = TrajectoryProblem(
            self.config.chain, start, goal, self.config.lower, self.config.upper,
            self.config.steps, self.config.dt,
            weights=self.config.trajectory_weights,
            collision_model=self.config.collision_model,
            collision_subdivisions=2,
            max_iterations=self.config.max_trajectory_iterations,
            learning_rate=0.02,
            endpoint_tolerance=1e-5 if self.config.dtype == torch.float32 else 1e-8,
        )
        result = optimize_trajectory(problem)
        graph_result = None
        graph_used = False
        if result.selected_seed is None and enable_graph:
            graph_result = plan_graph(GraphPlanningProblem(
                start, goal, self.config.lower, self.config.upper,
                chain=self.config.chain, collision_model=self.config.collision_model,
                sample_count=self.config.graph_sample_count, seed=self.config.graph_seed,
                k_neighbors=self.config.graph_k_neighbors,
                edge_step=self.config.graph_edge_step,
                interpolation_step=self.config.graph_edge_step,
            ))
            if bool(graph_result.success[0].item()):
                seeds = paths_to_trajectory_seeds(graph_result, self.config.steps)[0]
                result = optimize_trajectory(replace(problem, seeds=seeds))
                graph_used = True
            else:
                return self._failure(MotionGenStatus.GRAPH_FAILED, graph=graph_result)
        if result.selected_seed is None:
            return self._failure(MotionGenStatus.TRAJECTORY_FAILED, graph=graph_result)
        q = result.trajectories[result.selected_seed]
        dense = interpolate_trajectory(q, self.config.dt, self.config.interpolation_dt)
        raw_metrics = trajectory_metrics(problem, q)
        metrics = MotionGenMetrics(
            raw_metrics.duration, raw_metrics.path_length, raw_metrics.maximum_velocity,
            raw_metrics.maximum_acceleration, raw_metrics.maximum_jerk,
            raw_metrics.minimum_clearance, raw_metrics.maximum_limit_violation, graph_used,
        )
        return MotionGenResult(
            torch.tensor(True, device=q.device), MotionGenStatus.SUCCESS,
            JointState(q, self.joint_names), JointState(dense, self.joint_names),
            self.config.interpolation_dt, metrics, result, graph_result=graph_result,
        )

    plan_single_joint_space = plan_single_js

    def plan_batch_js(
        self,
        start_state: JointState | torch.Tensor,
        goal_state: JointState | torch.Tensor,
        *,
        enable_graph: bool = True,
    ) -> MotionGenResult:
        starts = self._position(start_state, batch=True)
        goals = self._position(goal_state, batch=True)
        if starts.shape != goals.shape:
            raise ValueError("batched start and goal shapes must match")
        rows = [
            self.plan_single_js(starts[index], goals[index], enable_graph=enable_graph)
            for index in range(starts.shape[0])
        ]
        success = torch.stack([row.success for row in rows])
        statuses = tuple(row.status for row in rows)
        if not bool(success.all().item()):
            return MotionGenResult(
                success, statuses, None, None, self.config.interpolation_dt, None
            )
        plans = torch.stack([row.optimized_plan.position for row in rows if row.optimized_plan])
        dense = torch.stack([row.interpolated_plan.position for row in rows if row.interpolated_plan])
        return MotionGenResult(
            success, statuses, JointState(plans, self.joint_names),
            JointState(dense, self.joint_names), self.config.interpolation_dt, None,
        )

    plan_batch_joint_space = plan_batch_js

    def plan_single(
        self,
        start_state: JointState | torch.Tensor,
        goal_pose: Pose,
        *,
        enable_graph: bool = True,
    ) -> MotionGenResult:
        start = self._position(start_state, batch=False)
        if not isinstance(goal_pose, Pose):
            raise TypeError("goal_pose must be Pose")
        generator = torch.Generator(device="cpu").manual_seed(self.config.graph_seed)
        unit = torch.rand(
            (self.config.num_ik_seeds, self.config.chain.dof), generator=generator,
            dtype=self.config.dtype, device="cpu",
        ).to(self.config.device)
        seeds = self.config.lower + unit * (self.config.upper - self.config.lower)
        seeds[0] = start
        pose_weights = torch.ones(6, device=self.config.device, dtype=self.config.dtype)
        ik_problem = IKProblem(
            self.config.chain, goal_pose.position, goal_pose.quaternion, seeds,
            self.config.lower, self.config.upper, pose_weights,
            position_tolerance=self.config.position_tolerance,
            rotation_tolerance=self.config.rotation_tolerance,
            max_iterations=self.config.max_ik_iterations,
            collision_model=self.config.collision_model,
        )
        ik = solve_ik(ik_problem)
        if not isinstance(ik.selected_seed, int):
            return self._failure(MotionGenStatus.IK_FAILED, ik=ik)
        result = self.plan_single_js(start, ik.solutions[ik.selected_seed], enable_graph=enable_graph)
        return replace(result, ik_result=ik)

    plan_single_pose = plan_single

    def plan_batch(
        self,
        start_state: JointState | torch.Tensor,
        goal_pose: Pose,
        *,
        enable_graph: bool = True,
    ) -> MotionGenResult:
        starts = self._position(start_state, batch=True)
        if goal_pose.position.ndim != 2 or goal_pose.position.shape[0] != starts.shape[0]:
            raise ValueError("batched pose positions must have shape [B,3]")
        if goal_pose.quaternion.shape != (starts.shape[0], 4):
            raise ValueError("batched pose quaternions must have shape [B,4]")
        rows = [
            self.plan_single(
                starts[index], Pose(goal_pose.position[index], goal_pose.quaternion[index]),
                enable_graph=enable_graph,
            )
            for index in range(starts.shape[0])
        ]
        success = torch.stack([row.success for row in rows])
        statuses = tuple(row.status for row in rows)
        if not bool(success.all().item()):
            return MotionGenResult(success, statuses, None, None, self.config.interpolation_dt, None)
        return MotionGenResult(
            success, statuses,
            JointState(torch.stack([row.optimized_plan.position for row in rows if row.optimized_plan]), self.joint_names),
            JointState(torch.stack([row.interpolated_plan.position for row in rows if row.interpolated_plan]), self.joint_names),
            self.config.interpolation_dt, None,
        )

    plan_batch_pose = plan_batch

    def update_world(self, world: object) -> None:
        raise UnsupportedMotionGenFeature(
            "runtime world mutation is unsupported; create a new MotionGenConfig"
        )

    def attach_objects_to_robot(self, *args: object, **kwargs: object) -> None:
        raise UnsupportedMotionGenFeature("attached-object mutation is unsupported")
