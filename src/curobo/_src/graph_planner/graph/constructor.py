"""Composable graph construction facade."""


class GraphConstructor:
    def __init__(
        self, config, linear_connector, distance_calculator, node_manager, action_dim,
        check_feasibility_fn, device_cfg,
    ):
        self.config = config
        self.linear_connector = linear_connector
        self.distance_calculator = distance_calculator
        self.node_manager = node_manager
        self.action_dim = action_dim
        self.check_feasibility_fn = check_feasibility_fn
        self.device_cfg = device_cfg

    def connect_nodes(self, new_nodes, add_exact_node=False, neighbors_per_node=10):
        del neighbors_per_node
        return self.node_manager.add_nodes_to_roadmap(new_nodes, add_exact_node)

    def steer_and_register_edges(self, start_nodes, goal_nodes, add_exact_node=False):
        steered = self.linear_connector.steer_until_infeasible(start_nodes, goal_nodes)
        return self.connect_nodes(steered, add_exact_node)

    def initialize_default_node(self, default_joint_state):
        return self.connect_nodes(default_joint_state.position.reshape(1, -1), True)

    def initialize_terminal_graph_connections(self, x_init_batch, x_goal_batch, default_joint_state):
        del default_joint_state
        return self.connect_nodes(x_init_batch, True), self.connect_nodes(x_goal_batch, True)

    def reset(self):
        self.node_manager.reset_buffer()


__all__ = ["GraphConstructor"]
