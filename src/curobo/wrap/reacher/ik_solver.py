"""Established inverse-kinematics wrapper names."""

from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_ik_result import IKSolverResult


class IKSolverConfig(IKSolverCfg):
    @staticmethod
    def load_from_robot_config(robot_config, world_config=None, tensor_args=None, **kwargs):
        if tensor_args is not None:
            kwargs["device_cfg"] = tensor_args
        return IKSolverCfg.create(robot_config, scene_model=world_config, **kwargs)


__all__ = ["IKSolver", "IKSolverConfig", "IKSolverResult"]
