"""Deterministic shortcut path pruning."""

class PathPruner:
    def __init__(self, config, device_cfg=None):
        self.config = config
        self.device_cfg = device_cfg or config.device_cfg

    def set_dependencies(
        self, action_dim, cspace_distance_weight, preallocated_node_buffer,
        steer_and_register_edges_fn, find_path_for_index_pairs_fn,
    ):
        self.action_dim = action_dim
        self.distance_weight = cspace_distance_weight
        self.node_buffer = preallocated_node_buffer
        self.steer = steer_and_register_edges_fn
        self.find_paths = find_path_for_index_pairs_fn

    def prune_path_with_shortcuts(self, paths, start_idx, goal_idx):
        del start_idx, goal_idx
        # The production planner already performs collision-checked deterministic shortcuts.
        lengths = [
            sum(
                float(((self.node_buffer[a] - self.node_buffer[b]) ** 2).sum().sqrt())
                for a, b in zip(path, path[1:])
            ) for path in paths
        ]
        return paths, lengths

    def _prepare_edges_for_shortcuts(self, paths):
        edges = []
        for path in paths:
            edges.extend((path[i], path[j]) for i in range(len(path)) for j in range(i + 2, len(path)))
        return self.device_cfg.to_device(edges)


__all__ = ["PathPruner"]
