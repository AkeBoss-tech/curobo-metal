from .kinematics_config import KinematicsKernelCfg, KinematicsLaunchCfg


def _unsupported(name):
    raise NotImplementedError(
        f"{name} uses raw CUDA pointer buffers; use curobo.kinematics.Kinematics "
        "for portable CPU/MPS forward kinematics and Jacobians"
    )


def launch_kinematics_forward(*args, **kwargs): return _unsupported("launch_kinematics_forward")
def launch_kinematics_forward_spheres(*args, **kwargs): return _unsupported("launch_kinematics_forward_spheres")
def launch_kinematics_forward_spheres_jacobian(*args, **kwargs): return _unsupported("launch_kinematics_forward_spheres_jacobian")
def launch_kinematics_backward(*args, **kwargs): return _unsupported("launch_kinematics_backward")
