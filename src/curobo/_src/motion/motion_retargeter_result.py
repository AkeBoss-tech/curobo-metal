from dataclasses import dataclass
from typing import Optional

from curobo._src.state.state_joint import JointState


@dataclass
class RetargetResult:
    """Output of one retargeted frame or an entire target sequence.

    ``joint_state`` has shape ``[environment, dof]`` for
    :meth:`MotionRetargeter.solve_frame` and ``[environment, frame, dof]``
    for :meth:`MotionRetargeter.solve_sequence`.  ``trajectory`` is the
    executed MPC endpoint stream when MPC is enabled; it is ``None`` for
    warm-started IK.
    """
    joint_state: JointState
    trajectory: Optional[JointState] = None


__all__ = ["RetargetResult"]
