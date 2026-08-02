import torch
class ToolPoseDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,current_position,current_quat,goal_position,goal_quat,*args):
        p=(current_position.unsqueeze(-2)-goal_position).square().sum(-1)
        q=1-(current_quat.unsqueeze(-2)*goal_quat).sum(-1).abs().clamp_max(1).square()
        value,index=(p+q).min(-1); return value,p.min(-1).values,q.min(-1).values,index
    @staticmethod
    def backward(ctx,*grad_outputs):
        raise RuntimeError("call ToolPoseCost for differentiable portable pose costs")
def create_goalset_pose_distance_kernel_with_constants(num_goalset,rotation_method=0):
    return ToolPoseDistance
def compute_position_error(current_position,goal_position,dof_weight,position_weight,convergence_tolerance):
    return (current_position-goal_position)*dof_weight*position_weight


def _weighted_quaternion_error(current_quat, goal_quat, rotation_dof_weight, rotation_weight):
    """Sign-invariant differentiable orientation discrepancy for portable callers."""
    current = torch.nn.functional.normalize(current_quat, dim=-1)
    goal = torch.nn.functional.normalize(goal_quat, dim=-1)
    residual = 1.0 - (current * goal).sum(-1).square().clamp_max(1.0)
    axis_weight = torch.as_tensor(rotation_dof_weight, device=residual.device, dtype=residual.dtype)
    scalar_weight = torch.as_tensor(rotation_weight, device=residual.device, dtype=residual.dtype)
    return residual * axis_weight.mean() * scalar_weight


def compute_rotation_error(current_quat,goal_quat,rotation_dof_weight,rotation_weight,
                           convergence_tolerance,rotation_error_method):
    del convergence_tolerance, rotation_error_method
    return _weighted_quaternion_error(current_quat, goal_quat, rotation_dof_weight, rotation_weight)


def compute_rotation_error_axis_angle(current_quat,goal_quat,rotation_dof_weight,rotation_weight,
                                      convergence_tolerance):
    del convergence_tolerance
    return _weighted_quaternion_error(current_quat, goal_quat, rotation_dof_weight, rotation_weight)


def compute_rotation_error_lie_group(current_quat,goal_quat,rotation_dof_weight,rotation_weight,
                                     convergence_tolerance):
    del convergence_tolerance
    return _weighted_quaternion_error(current_quat, goal_quat, rotation_dof_weight, rotation_weight)


def compute_rotation_error_lie_group_advanced(current_quat,goal_quat,rotation_dof_weight,
                                              rotation_weight,convergence_tolerance):
    return compute_rotation_error_lie_group(
        current_quat, goal_quat, rotation_dof_weight, rotation_weight, convergence_tolerance
    )


def convert_angular_velocity_to_quaternion_rate(angular_velocity,current_quat):
    """Map xyz angular velocity to a scalar-first quaternion derivative."""
    omega = torch.cat((torch.zeros_like(angular_velocity[..., :1]), angular_velocity), dim=-1)
    w, x, y, z = current_quat.unbind(-1)
    ow, ox, oy, oz = omega.unbind(-1)
    return 0.5 * torch.stack((
        w * ow - x * ox - y * oy - z * oz,
        w * ox + x * ow + y * oz - z * oy,
        w * oy - x * oz + y * ow + z * ox,
        w * oz + x * oy - y * ox + z * ow,
    ), dim=-1)


def scale_quaternion_difference_by_axis(q,weights):
    value = torch.as_tensor(weights, device=q.device, dtype=q.dtype)
    if value.shape[-1] == 3:
        value = torch.cat((torch.ones_like(value[..., :1]), value), dim=-1)
    return q * value
__all__=["ToolPoseDistance","create_goalset_pose_distance_kernel_with_constants",
         "compute_position_error","compute_rotation_error","compute_rotation_error_axis_angle",
         "compute_rotation_error_lie_group","compute_rotation_error_lie_group_advanced",
         "convert_angular_velocity_to_quaternion_rate","scale_quaternion_difference_by_axis"]
