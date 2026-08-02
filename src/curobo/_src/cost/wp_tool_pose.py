import torch
class ToolPoseDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,current_position,current_quat,goal_position,goal_quat,*args):
        if current_position.shape[-1] != 3 or current_quat.shape[-1] != 4:
            raise ValueError("current positions/quaternions must end in 3/4")
        if goal_position.shape[-1] != 3 or goal_quat.shape[-1] != 4:
            raise ValueError("goal positions/quaternions must end in 3/4")
        idxs_goal = args[0] if args and isinstance(args[0], torch.Tensor) and args[0].dtype in (torch.int32, torch.int64) else None
        if idxs_goal is not None and goal_position.shape[0] != current_position.shape[0]:
            goal_position = goal_position.index_select(0, idxs_goal.to(goal_position.device, torch.long))
            goal_quat = goal_quat.index_select(0, idxs_goal.to(goal_quat.device, torch.long))
        delta = current_position.unsqueeze(-2) - goal_position
        p = delta.square().sum(-1)
        cquat = torch.nn.functional.normalize(current_quat, dim=-1)
        gquat = torch.nn.functional.normalize(goal_quat, dim=-1)
        dot = (cquat.unsqueeze(-2) * gquat).sum(-1)
        q = 1 - dot.abs().clamp_max(1).square()
        value, index = (p + q).min(-1)
        p_selected = p.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        q_selected = q.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        # Store exactly the selected goal and enough information for a first
        # order portable VJP.  The CUDA implementation only promises first
        # order gradients for this bridge as well.
        gathered_pos = goal_position.gather(-2, index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, 3)).squeeze(-2)
        gathered_quat = gquat.gather(-2, index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, 4)).squeeze(-2)
        selected_dot = dot.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(current_position, cquat, gathered_pos, gathered_quat, selected_dot, index)
        ctx.goal_shape = goal_position.shape
        ctx.goal_batch_indexed = idxs_goal is not None
        ctx.n_inputs = 4 + len(args)
        return value, p_selected, q_selected, index.to(torch.int32)
    @staticmethod
    def backward(ctx,*grad_outputs):
        grad_value = grad_outputs[0]
        if grad_value is None:
            return (None,) * ctx.n_inputs
        current_pos, current_quat, goal_pos, goal_quat, dot, index = ctx.saved_tensors
        scale = grad_value.unsqueeze(-1)
        grad_pos = 2 * (current_pos - goal_pos) * scale
        # This is the Euclidean derivative of sign-invariant dot-square.  The
        # normalized-input projection avoids a CUDA/Warp-only quaternion ABI.
        grad_quat = -2 * dot.abs().sign().unsqueeze(-1) * dot.unsqueeze(-1) * goal_quat * scale
        grad_goal_pos = torch.zeros(ctx.goal_shape, device=current_pos.device, dtype=current_pos.dtype)
        grad_goal_quat = torch.zeros((*ctx.goal_shape[:-1], 4), device=current_quat.device, dtype=current_quat.dtype)
        grad_goal_pos.scatter_add_(-2, index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, 3), -grad_pos.unsqueeze(-2))
        grad_goal_quat.scatter_add_(-2, index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, 4), -grad_quat.unsqueeze(-2))
        # We cannot safely invert batch indirection for arbitrary repeated
        # indices in a custom Function; ToolPoseCost handles that path with
        # ordinary PyTorch graph composition instead.
        if ctx.goal_batch_indexed:
            grad_goal_pos = None
            grad_goal_quat = None
        return (grad_pos, grad_quat, grad_goal_pos, grad_goal_quat) + (None,) * (ctx.n_inputs - 4)
def create_goalset_pose_distance_kernel_with_constants(num_goalset,rotation_method=0):
    return ToolPoseDistance
def compute_position_error(current_position,goal_position,dof_weight,position_weight,convergence_tolerance):
    delta = current_position - goal_position
    weight = torch.as_tensor(dof_weight, device=delta.device, dtype=delta.dtype)
    scalar = torch.as_tensor(position_weight, device=delta.device, dtype=delta.dtype)
    cost = 0.5 * scalar * (delta * weight).square().sum(-1)
    gradient = scalar * weight.square() * delta
    tolerance = torch.as_tensor(convergence_tolerance, device=delta.device, dtype=delta.dtype)
    return torch.where(cost < tolerance, torch.zeros_like(cost), cost), torch.where(
        (cost < tolerance).unsqueeze(-1), torch.zeros_like(gradient), gradient
    )


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
