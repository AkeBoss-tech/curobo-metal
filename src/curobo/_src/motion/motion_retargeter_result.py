from dataclasses import dataclass
from typing import Optional

from curobo._src.state.state_joint import JointState


@dataclass
class RetargetResult:
    joint_state: JointState
    trajectory: Optional[JointState] = None


__all__ = ["RetargetResult"]
