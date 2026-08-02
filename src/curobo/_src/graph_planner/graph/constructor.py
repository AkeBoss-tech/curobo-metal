"""Deterministic CPU/MPS construction of the pinned PRM graph lifecycle.

The upstream class joins CUDA rollout buffers, Warp steering and a NetworkX
search graph.  The useful public contract is independent of that machinery:
terminal rows are assigned stable roadmap indices, a feasible default posture
is cached for a graph generation, and each new row is connected to a stable
set of nearest existing rows.  This implementation preserves that behaviour
with the portable graph components.  CUDA graph capture, Warp kernels, and
analytic continuous collision detection are deliberately not emulated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple

import torch

from curobo._src.graph_planner.graph.connector_linear import LinearConnector
from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator
from curobo._src.graph_planner.graph.node_manager import GraphNodeManager
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.state.state_joint import JointState


class GraphConstructor:
    """Build and maintain portable PRM connections for one roadmap generation.

    Rows passed to the steering/registration methods are ``[action..., index]``
    tensors.  Index values are maintained by :class:`GraphNodeManager`; callers
    may supply action-only rows to :meth:`connect_nodes` and terminal setup.
    """

    def __init__(
        self,
        config: PRMGraphPlannerCfg,
        linear_connector: LinearConnector,
        distance_calculator: DistanceNeighborCalculator,
        node_manager: GraphNodeManager,
        action_dim: int,
        check_feasibility_fn,
        device_cfg,
    ):
        if not isinstance(action_dim, int) or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if not callable(check_feasibility_fn):
            raise TypeError("check_feasibility_fn must be callable")
        self.config = config
        self.device_cfg = device_cfg
        self.linear_connector = linear_connector
        self.distance_calculator = distance_calculator
        self.node_manager = node_manager
        self.action_dim = action_dim
        self.check_feasibility_fn = check_feasibility_fn
        self._default_joint_position_feasible: Optional[bool] = None
        self._default_node_in_roadmap: Optional[torch.Tensor] = None
        # PyTorch materializes ``mps`` tensors as ``mps:0`` whereas a portable
        # DeviceCfg intentionally accepts the index-less spelling.  Older
        # GraphNodeManager registration performs one strict device comparison;
        # normalize *its* private descriptor to its already allocated buffer
        # so constructor-driven MPS registration remains device-resident.
        buffer_device = node_manager.preallocated_node_buffer.device
        manager_cfg = node_manager.device_cfg
        if manager_cfg.is_same_torch_device(buffer_device) and manager_cfg.device != buffer_device:
            node_manager.device_cfg = replace(manager_cfg, device=buffer_device)

    def _validate_actions(self, nodes: torch.Tensor, *, name: str) -> None:
        if not isinstance(nodes, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if nodes.ndim != 2 or nodes.shape[-1] != self.action_dim:
            raise ValueError(f"{name} must have shape [B, {self.action_dim}]")
        if not self.device_cfg.is_same_torch_device(nodes.device):
            raise ValueError(f"{name} must be on {self.device_cfg.device}")
        if nodes.dtype != self.device_cfg.dtype:
            raise ValueError(f"{name} must use {self.device_cfg.dtype}")
        if not bool(torch.isfinite(nodes).all().item()):
            raise ValueError(f"{name} must contain finite values")

    def _validate_indexed(self, nodes: torch.Tensor, *, name: str) -> None:
        if not isinstance(nodes, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor")
        if nodes.ndim != 2 or nodes.shape[-1] != self.action_dim + 1:
            raise ValueError(f"{name} must have shape [B, {self.action_dim + 1}]")
        if not self.device_cfg.is_same_torch_device(nodes.device):
            raise ValueError(f"{name} must be on {self.device_cfg.device}")
        if nodes.dtype != self.device_cfg.dtype:
            raise ValueError(f"{name} must use {self.device_cfg.dtype}")
        if not bool(torch.isfinite(nodes).all().item()):
            raise ValueError(f"{name} must contain finite values")

    def _indexed(self, nodes: torch.Tensor, *, name: str) -> torch.Tensor:
        """Accept action rows or already-indexed graph rows without copying rows."""
        if not isinstance(nodes, torch.Tensor) or nodes.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional tensor")
        if nodes.shape[-1] == self.action_dim + 1:
            self._validate_indexed(nodes, name=name)
            return nodes
        self._validate_actions(nodes, name=name)
        padding = self.node_manager.node_idx_padding_buffer.expand(nodes.shape[0], 1)
        return torch.cat((nodes, padding), dim=-1)

    def _steer(self, start_nodes: torch.Tensor, goal_nodes: torch.Tensor) -> torch.Tensor:
        """Run the configured portable connector and normalize its row shape."""
        steered = self.linear_connector.steer_until_infeasible(start_nodes, goal_nodes)
        if not isinstance(steered, torch.Tensor) or steered.ndim != 2:
            raise TypeError("linear_connector must return a two-dimensional tensor")
        if steered.shape[0] != start_nodes.shape[0]:
            raise ValueError("linear_connector changed the steering batch size")
        if steered.shape[-1] == self.action_dim:
            # Some portable connectors return action rows.  Retain the source
            # index so registered edges remain anchored to their start node.
            steered = torch.cat((steered, goal_nodes[:, -1:]), dim=-1)
        self._validate_indexed(steered, name="steered nodes")
        return steered

    def steer_and_register_edges(
        self,
        start_nodes: torch.Tensor,
        goal_nodes: torch.Tensor,
        add_exact_node: bool = False,
    ) -> None:
        """Steer paired rows and atomically register their graph connections."""
        self._validate_indexed(start_nodes, name="start_nodes")
        self._validate_indexed(goal_nodes, name="goal_nodes")
        if start_nodes.shape[0] != goal_nodes.shape[0]:
            raise ValueError("start_nodes and goal_nodes must have the same batch size")
        if start_nodes.shape[0] == 0:
            return
        steered = self._steer(start_nodes, goal_nodes)
        self.node_manager.register_nodes_and_connections(
            steered, start_nodes, add_exact_node=add_exact_node
        )

    def connect_nodes(
        self,
        new_nodes: torch.Tensor,
        add_exact_node: bool = False,
        neighbors_per_node: int = 10,
    ) -> None:
        """Connect each candidate to its nearest current roadmap vertices.

        The terminal lifecycle initializes an empty graph explicitly.  Calling
        this method before that initialization is therefore a useful error,
        rather than silently creating unconnected graph vertices.
        """
        if not isinstance(neighbors_per_node, int) or neighbors_per_node < 1:
            raise ValueError("neighbors_per_node must be a positive integer")
        candidates = self._indexed(new_nodes, name="new_nodes")
        if candidates.shape[0] == 0:
            return
        if self.node_manager.n_nodes == 0:
            raise ValueError("cannot connect nodes before initial roadmap nodes are registered")
        neighbor_indices = self.distance_calculator.find_nearest_neighbors(
            candidates[:, :self.action_dim],
            self.node_manager.valid_node_buffer[:, :self.action_dim],
            neighbors_per_node=min(neighbors_per_node, self.node_manager.n_nodes),
        )
        if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != candidates.shape[0]:
            raise ValueError("distance_calculator returned invalid nearest-neighbor indices")
        if neighbor_indices.shape[1] == 0:
            return
        starts = self.node_manager.valid_node_buffer[neighbor_indices]
        goals = candidates[:, None, :].expand(-1, starts.shape[1], -1)
        self.steer_and_register_edges(
            starts.reshape(-1, self.action_dim + 1),
            goals.reshape(-1, self.action_dim + 1),
            add_exact_node=add_exact_node,
        )

    def initialize_default_node(
        self, default_joint_state: JointState
    ) -> Tuple[Optional[torch.Tensor], Optional[bool]]:
        """Cache and optionally install the configured default joint posture."""
        if not bool(getattr(self.config, "use_default_position_heuristic", False)):
            return None, None
        if self._default_joint_position_feasible is not None:
            return self._default_node_in_roadmap, self._default_joint_position_feasible
        if not isinstance(default_joint_state, JointState):
            raise TypeError("default_joint_state must be a JointState")
        position = default_joint_state.position.reshape(-1, self.action_dim)
        if position.shape[0] != 1:
            raise ValueError("default_joint_state must contain exactly one action row")
        self._validate_actions(position, name="default_joint_state.position")
        feasible = self.check_feasibility_fn(position)
        if not isinstance(feasible, torch.Tensor) or feasible.shape != (1,) or feasible.dtype != torch.bool:
            raise ValueError("check_feasibility_fn must return bool shape [1] for default position")
        if feasible.device != position.device:
            raise ValueError("check_feasibility_fn must preserve the input device")
        self._default_joint_position_feasible = bool(feasible.item())
        if self._default_joint_position_feasible:
            if self.node_manager.n_nodes == 0:
                self._default_node_in_roadmap = self.node_manager.add_initial_exact_nodes_to_roadmap(
                    position.clone()
                )
            else:
                self._default_node_in_roadmap = self.node_manager.add_nodes_to_roadmap(
                    position.clone(), add_exact_node=True
                )
        return self._default_node_in_roadmap, self._default_joint_position_feasible

    def _preprocess_terminal_nodes(
        self,
        start_nodes_in_roadmap: torch.Tensor,
        goal_nodes_in_roadmap: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return stable bidirectional terminal/default steering pairs."""
        self._validate_indexed(start_nodes_in_roadmap, name="start_nodes_in_roadmap")
        self._validate_indexed(goal_nodes_in_roadmap, name="goal_nodes_in_roadmap")
        if start_nodes_in_roadmap.shape[0] != goal_nodes_in_roadmap.shape[0]:
            raise ValueError("terminal node batches must have the same size")
        start_steer = torch.cat((start_nodes_in_roadmap, goal_nodes_in_roadmap), dim=0)
        goal_steer = torch.cat((goal_nodes_in_roadmap, start_nodes_in_roadmap), dim=0)
        if self._default_joint_position_feasible:
            assert self._default_node_in_roadmap is not None
            default = self._default_node_in_roadmap.expand(start_nodes_in_roadmap.shape[0], -1)
            start_steer = torch.cat(
                (start_steer, start_nodes_in_roadmap, default, goal_nodes_in_roadmap, default), dim=0
            )
            goal_steer = torch.cat(
                (goal_steer, default, start_nodes_in_roadmap, default, goal_nodes_in_roadmap), dim=0
            )
        return start_steer, goal_steer

    def initialize_terminal_graph_connections(
        self,
        x_init_batch: torch.Tensor,
        x_goal_batch: torch.Tensor,
        default_joint_state: JointState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize batched terminal vertices and their direct/nearest edges."""
        self._validate_actions(x_init_batch, name="x_init_batch")
        self._validate_actions(x_goal_batch, name="x_goal_batch")
        if x_init_batch.shape[0] != x_goal_batch.shape[0]:
            raise ValueError("x_init_batch and x_goal_batch must have the same batch size")
        if x_init_batch.shape[0] == 0:
            empty = x_init_batch.new_empty((0, self.action_dim + 1))
            return empty, empty
        self.initialize_default_node(default_joint_state)
        terminals = torch.cat((x_init_batch, x_goal_batch), dim=0)
        if self.node_manager.n_nodes == 0:
            indexed = self.node_manager.add_initial_exact_nodes_to_roadmap(terminals)
        else:
            indexed = self.node_manager.add_nodes_to_roadmap(terminals, add_exact_node=True)
        batch = x_init_batch.shape[0]
        starts, goals = indexed[:batch], indexed[batch:]
        start_steer, goal_steer = self._preprocess_terminal_nodes(starts, goals)
        self.steer_and_register_edges(start_steer, goal_steer, add_exact_node=False)
        if bool(getattr(self.config, "connect_terminal_nodes_with_nearest", False)):
            self.connect_nodes(
                terminals,
                add_exact_node=False,
                neighbors_per_node=int(getattr(self.config, "neighbors_per_node", 10)),
            )
        return starts, goals

    def reset(self) -> None:
        """Clear per-generation default-node state without resetting the roadmap."""
        self._default_joint_position_feasible = None
        self._default_node_in_roadmap = None


__all__ = ["GraphConstructor"]
