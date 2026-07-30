"""Joint-state filtering and command integration."""

from dataclasses import dataclass
from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_joint_ops import blend_joint_states
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg


@dataclass(frozen=True)
class FilterCfg:
    filter_coeff: FilterCoeff
    dt: float
    control_space: ControlSpace
    device_cfg: DeviceCfg = DeviceCfg()
    enable: bool = True
    teleport_mode: bool = False

    @staticmethod
    def create(coeff_dict, enable=True, dt=0.0, control_space=ControlSpace.ACCELERATION,
               device_cfg=DeviceCfg(), teleport_mode=False):
        return FilterCfg(FilterCoeff(**coeff_dict), dt, control_space, device_cfg, enable, teleport_mode)


class JointStateFilter(FilterCfg):
    def __init__(self, filter_config):
        super().__init__(**vars(filter_config))
        self.cmd_joint_state = None
        self.integrate_action = {
            ControlSpace.ACCELERATION: self.integrate_acc,
            ControlSpace.VELOCITY: self.integrate_vel,
        }.get(self.control_space, self.integrate_pos)

    def filter_joint_state(self, raw_joint_state):
        if not self.enable:
            return raw_joint_state
        raw_joint_state = raw_joint_state.to(self.device_cfg)
        if self.cmd_joint_state is None:
            self.cmd_joint_state = raw_joint_state.clone()
        else:
            blend_joint_states(self.cmd_joint_state, raw_joint_state, self.filter_coeff)
        return self.cmd_joint_state

    def _set_state(self, state):
        if state is not None:
            self.cmd_joint_state = state.clone()
        if self.cmd_joint_state is None:
            raise ValueError("cmd_joint_state is required")

    def integrate_jerk(self, qddd_des, cmd_joint_state=None, dt=None):
        self._set_state(cmd_joint_state); dt = self.dt if dt is None else dt
        self.cmd_joint_state.jerk = qddd_des
        self.cmd_joint_state.acceleration = self.cmd_joint_state.acceleration + qddd_des * dt
        self.cmd_joint_state.velocity = self.cmd_joint_state.velocity + self.cmd_joint_state.acceleration * dt
        self.cmd_joint_state.position = self.cmd_joint_state.position + self.cmd_joint_state.velocity * dt
        return self.cmd_joint_state

    def integrate_acc(self, qdd_des, cmd_joint_state=None, dt=None):
        self._set_state(cmd_joint_state); dt = self.dt if dt is None else dt
        self.cmd_joint_state.acceleration = qdd_des
        self.cmd_joint_state.velocity = self.cmd_joint_state.velocity + qdd_des * dt
        self.cmd_joint_state.position = self.cmd_joint_state.position + self.cmd_joint_state.velocity * dt
        self.cmd_joint_state.jerk = qdd_des * 0
        return self.cmd_joint_state.clone()

    def integrate_vel(self, qd_des, cmd_joint_state=None, dt=None):
        self._set_state(cmd_joint_state); dt = self.dt if dt is None else dt
        self.cmd_joint_state.velocity = qd_des
        self.cmd_joint_state.position = self.cmd_joint_state.position + qd_des * dt
        return self.cmd_joint_state

    def integrate_pos(self, q_des, cmd_joint_state=None, dt=None):
        self._set_state(cmd_joint_state); dt = self.dt if dt is None else dt
        if not self.teleport_mode:
            self.cmd_joint_state.velocity = (q_des - self.cmd_joint_state.position) / dt
        self.cmd_joint_state.position = q_des
        return self.cmd_joint_state

    def reset(self):
        self.cmd_joint_state = None
