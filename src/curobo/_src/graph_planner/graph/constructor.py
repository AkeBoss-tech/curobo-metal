"""Composable graph construction facade."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from curobo._src.state.state_joint import JointState


class GraphConstructor:
    def __init__(
        self,
        config: PRMGraphPlannerCfg,
        linear_connector: LinearConnector,
        distance_calculator: DistanceNeighborCalculator,
        node_manager: GraphNodeManager,
        action_dim: int,
        check_feasibility_fn, device_cfg,
    ):
        self.config = config
        self.linear_connector = linear_connector
        self.distance_calculator = distance_calculator
        self.node_manager = node_manager
        self.action_dim = action_dim
        self.check_feasibility_fn = check_feasibility_fn
        self.device_cfg = device_cfg

    def connect_nodes(
        self, new_nodes: torch.Tensor, add_exact_node=False, neighbors_per_node=10
    ):
        del neighbors_per_node
        return self.node_manager.add_nodes_to_roadmap(new_nodes, add_exact_node)

    def steer_and_register_edges(
        self, start_nodes: torch.Tensor, goal_nodes: torch.Tensor, add_exact_node=False
    ):
        steered = self.linear_connector.steer_until_infeasible(start_nodes, goal_nodes)
        return self.connect_nodes(steered, add_exact_node)

    def initialize_default_node(
        self, default_joint_state: JointState
    ) -> Tuple[Optional[torch.Tensor], bool]:
        result = self.connect_nodes(default_joint_state.position.reshape(1, -1), True)
        return result, True

    def initialize_terminal_graph_connections(
        self, x_init_batch: torch.Tensor, x_goal_batch: torch.Tensor, default_joint_state: JointState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del default_joint_state
        return self.connect_nodes(x_init_batch, True), self.connect_nodes(x_goal_batch, True)

    def reset(self):
        self.node_manager.reset_buffer()


__all__ = ["GraphConstructor"]
