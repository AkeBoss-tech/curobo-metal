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
def compute_rotation_error(*args,**kwargs): raise RuntimeError("use ToolPoseCost")
compute_rotation_error_axis_angle=compute_rotation_error
compute_rotation_error_lie_group=compute_rotation_error
compute_rotation_error_lie_group_advanced=compute_rotation_error
def convert_angular_velocity_to_quaternion_rate(*args,**kwargs): raise RuntimeError("use PyTorch quaternion helpers")
def scale_quaternion_difference_by_axis(*args,**kwargs): raise RuntimeError("use ToolPoseCost")
__all__=["ToolPoseDistance","create_goalset_pose_distance_kernel_with_constants",
         "compute_position_error","compute_rotation_error","compute_rotation_error_axis_angle",
         "compute_rotation_error_lie_group","compute_rotation_error_lie_group_advanced",
         "convert_angular_velocity_to_quaternion_rate","scale_quaternion_difference_by_axis"]
