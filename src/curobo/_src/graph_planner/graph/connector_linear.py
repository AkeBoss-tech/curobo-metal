"""Linear swept-validity connector."""

import torch


class LinearConnector:
    def __init__(self, config, device_cfg=None):
        self.config = config
        self.device_cfg = device_cfg or config.device_cfg

    def set_dependencies(
        self, action_dim, cspace_distance_weight, check_feasibility_fn,
        preallocated_idx_buffer,
    ):
        self.action_dim = action_dim
        self.distance_weight = cspace_distance_weight
        self.check_feasibility_fn = check_feasibility_fn
        self.preallocated_idx_buffer = preallocated_idx_buffer

    def _compute_steering_line_points(self, start_nodes, desired_nodes):
        distance = torch.max(torch.abs(desired_nodes - start_nodes), dim=-1).values
        count = max(2, int(torch.ceil(distance.max() / self.config.edge_step).item()) + 1)
        phase = torch.linspace(
            0, 1, count, device=start_nodes.device, dtype=start_nodes.dtype
        )
        return start_nodes[:, None] + phase[None, :, None] * (
            desired_nodes - start_nodes
        )[:, None]

    def steer_until_infeasible(self, start_nodes, desired_nodes):
        line = self._compute_steering_line_points(start_nodes, desired_nodes)
        mask = self.check_feasibility_fn(line.reshape(-1, line.shape[-1])).reshape(line.shape[:2])
        prefix = torch.cumprod(mask.to(torch.int64), dim=1).sum(1).clamp_min(1) - 1
        return line[torch.arange(line.shape[0], device=line.device), prefix]


__all__ = ["LinearConnector"]
