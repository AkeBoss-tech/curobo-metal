"""Pinned MPC result model."""

from dataclasses import dataclass, fields
from typing import Optional

import torch

from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState


@dataclass
class MPCSolverResult(BaseSolverResult):
    next_action: Optional[JointState] = None
    action_sequence: Optional[JointState] = None
    full_action_sequence: Optional[JointState] = None
    robot_state_sequence: Optional[RobotState] = None
    action_buffer: Optional[torch.Tensor] = None
    action_dt: Optional[float] = None

    def clone(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.clone() if hasattr(value, "clone") else (
                dict(value) if isinstance(value, dict) else value
            )
        return type(self)(**values)


__all__ = ["MPCSolverResult"]
