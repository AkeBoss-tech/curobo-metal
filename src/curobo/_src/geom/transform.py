"""Portable differentiable pose and point transforms (``wxyz`` quaternions).

The pinned implementation uses Warp custom autograd kernels.  The routines
below deliberately expose the same value-level interface using regular
PyTorch tensor expressions, which makes them usable on both CPU and MPS and
keeps first-order autograd available without a CUDA/Warp runtime.
"""

from typing import Optional, Tuple
import torch

from .quaternion import normalize_quaternion, quat_multiply


def torch_quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    if quaternions.shape[-1:] != (4,):
        raise ValueError("quaternions must end in dimension 4 (wxyz)")
    q = normalize_quaternion(quaternions)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


def quaternion_to_matrix(quaternions, out_mat=None, adj_quaternion=None):
    result = torch_quaternion_to_matrix(quaternions)
    if out_mat is not None:
        out_mat.copy_(result)
        return out_mat
    return result


def matrix_to_quaternion(matrix, out_quat=None, adj_matrix=None):
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must end in 3x3")
    if not matrix.is_floating_point():
        raise TypeError("matrix_to_quaternion requires a floating-point tensor")
    m = matrix
    q_abs = torch.sqrt(torch.clamp(torch.stack((
        1+m[...,0,0]+m[...,1,1]+m[...,2,2],
        1+m[...,0,0]-m[...,1,1]-m[...,2,2],
        1-m[...,0,0]+m[...,1,1]-m[...,2,2],
        1-m[...,0,0]-m[...,1,1]+m[...,2,2],
    ), -1), min=0.0))
    candidates = torch.stack((
        torch.stack((q_abs[...,0]**2, m[...,2,1]-m[...,1,2], m[...,0,2]-m[...,2,0], m[...,1,0]-m[...,0,1]), -1),
        torch.stack((m[...,2,1]-m[...,1,2], q_abs[...,1]**2, m[...,1,0]+m[...,0,1], m[...,0,2]+m[...,2,0]), -1),
        torch.stack((m[...,0,2]-m[...,2,0], m[...,1,0]+m[...,0,1], q_abs[...,2]**2, m[...,2,1]+m[...,1,2]), -1),
        torch.stack((m[...,1,0]-m[...,0,1], m[...,2,0]+m[...,0,2], m[...,2,1]+m[...,1,2], q_abs[...,3]**2), -1),
    ), -2)
    denom = (2*q_abs).clamp_min(torch.finfo(m.dtype).eps)[..., None]
    index = q_abs.argmax(-1)
    result = normalize_quaternion(candidates.gather(-2, index[...,None,None].expand(index.shape+(1,4))).squeeze(-2) / denom.gather(-2, index[...,None,None].expand(index.shape+(1,1))).squeeze(-2))
    result = torch.where(result[..., :1] < 0, -result, result)
    if out_quat is not None:
        out_quat.copy_(result); return out_quat
    return result


def pose_to_matrix(position, quaternion, out_matrix=None):
    if position.shape[-1:] != (3,) or quaternion.shape[:-1] != position.shape[:-1] or quaternion.shape[-1:] != (4,):
        raise ValueError("position [...,3] and quaternion [...,4] batch dimensions must match")
    rotation = quaternion_to_matrix(quaternion)
    matrix = torch.zeros(rotation.shape[:-2] + (4,4), dtype=rotation.dtype, device=rotation.device)
    matrix[..., :3, :3] = rotation
    matrix[..., :3, 3] = position
    matrix[..., 3, 3] = 1
    if out_matrix is not None:
        out_matrix.copy_(matrix); return out_matrix
    return matrix


def pose_to_affine_matrix(position, quaternion, out_matrix=None):
    result = pose_to_matrix(position, quaternion)[..., :3, :]
    if out_matrix is not None:
        out_matrix.copy_(result); return out_matrix
    return result


def transform_points(position, quaternion, points, out_points=None, **kwargs):
    if position.shape[-1:] != (3,) or quaternion.shape[-1:] != (4,) or points.shape[-1:] != (3,):
        raise ValueError("position, quaternion, and points must end in [3], [4], and [3]")
    if position.shape[:-1] != quaternion.shape[:-1]:
        raise ValueError("position and quaternion batch dimensions must match")
    result = torch.matmul(points, quaternion_to_matrix(quaternion).transpose(-1,-2)) + position.unsqueeze(-2)
    if out_points is not None:
        out_points.copy_(result); return out_points
    return result


def batch_transform_points(position, quaternion, points, out_points=None, **kwargs):
    return transform_points(position, quaternion, points, out_points)


def batch_transform_points_inverse(position, quaternion, points, out_points=None, **kwargs):
    result = torch.matmul(points-position.unsqueeze(-2), quaternion_to_matrix(quaternion))
    if out_points is not None:
        out_points.copy_(result); return out_points
    return result


def get_inv_transform(w_rot_c: torch.Tensor, w_trans_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert a rotation-matrix/translation rigid transform.

    This has a different signature from :func:`pose_inverse`: it is the
    matrix-form helper used by geometry and mapper call sites in pinned V2.
    """
    if w_rot_c.shape[-2:] != (3, 3) or w_trans_c.shape[-1:] != (3,):
        raise ValueError("rotation must end in [3,3] and translation in [3]")
    if w_rot_c.shape[:-2] != w_trans_c.shape[:-1]:
        raise ValueError("rotation and translation batch dimensions must match")
    c_rot_w = w_rot_c.transpose(-1, -2)
    c_trans_w = -torch.matmul(c_rot_w, w_trans_c.unsqueeze(-1)).squeeze(-1)
    return c_rot_w, c_trans_w


def transform_point_inverse(point: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Transform points by the inverse of a matrix-form rigid transform."""
    c_rot_w, c_trans_w = get_inv_transform(rot, trans)
    if point.shape[-1:] != (3,):
        raise ValueError("point must end in dimension 3")
    return torch.matmul(point, c_rot_w.transpose(-1, -2)) + c_trans_w


# These names refer to Warp kernels upstream.  In a portable installation the
# mathematical helpers above are the supported interface; preserving aliases
# helps ordinary Python callers while not claiming a raw Warp kernel ABI.
compute_transform_point = transform_points
compute_batch_transform_point = batch_transform_points
compute_batch_transform_point_inverse = batch_transform_points_inverse


def pose_multiply(position, quaternion, position2, quaternion2, out_position=None, out_quaternion=None, **kwargs):
    p = transform_points(position, quaternion, position2.unsqueeze(-2)).squeeze(-2)
    q = normalize_quaternion(quat_multiply(quaternion, quaternion2))
    if out_position is not None: out_position.copy_(p); p = out_position
    if out_quaternion is not None: out_quaternion.copy_(q); q = out_quaternion
    return p, q


compute_pose_multipy = pose_multiply
compute_batch_pose_multipy = pose_multiply


def pose_inverse(position, quaternion, out_position=None, out_quaternion=None, **kwargs):
    q = normalize_quaternion(quaternion)
    qi = torch.cat((q[..., :1], -q[..., 1:]), -1)
    p = -torch.matmul(position.unsqueeze(-2), quaternion_to_matrix(qi).transpose(-1,-2)).squeeze(-2)
    if out_position is not None: out_position.copy_(p); p = out_position
    if out_quaternion is not None: out_quaternion.copy_(qi); qi = out_quaternion
    return p, qi


compute_pose_inverse = pose_inverse
compute_quat_to_matrix = quaternion_to_matrix
compute_matrix_to_quat = matrix_to_quaternion


def quaternion_rate_to_axis_angle_rate(quaternion_rate, current_quaternion):
    """Convert a quaternion rate to the pinned V2 axis-angle-rate residual.

    Keep this expression in the same convention as V2's implementation.  It
    is intentionally not the similarly named utility in the tool-pose cost,
    whose inputs are angular velocities rather than quaternion residuals.
    """
    if quaternion_rate.shape[-1:] != (4,) or current_quaternion.shape[-1:] != (4,):
        raise ValueError("quaternion rate and current quaternion must end in dimension 4")
    q = normalize_quaternion(current_quaternion)
    qw, qx, qy, qz = q.unbind(-1)
    dqw, dqx, dqy, dqz = quaternion_rate.unbind(-1)
    return 0.5 * torch.stack(
        (
            -qx * dqw + qw * dqx + qz * dqy - qy * dqz,
            -qy * dqw - qz * dqx + qw * dqy + qx * dqz,
            -qz * dqw + qy * dqx - qx * dqy + qw * dqz,
        ),
        dim=-1,
    )


class _FunctionFacade(torch.autograd.Function):
    """Compatibility shell for callers that directly use pinned ``.apply`` APIs.

    The public helpers do not need this wrapper, but upstream code sometimes
    calls the Function classes directly.  Save only the differentiable inputs
    and reconstruct their ordinary PyTorch VJP in ``backward``; extra Warp
    output/adjoint buffers are accepted but intentionally receive no gradient.
    """

    @classmethod
    def _run(cls, fn, ctx, inputs, extra, *args):
        ctx.fn = fn
        ctx.input_count = len(inputs)
        ctx.extra_count = len(extra)
        ctx.save_for_backward(*inputs)
        return fn(*args)

    @staticmethod
    def _vjp(ctx, grad_output):
        saved = ctx.saved_tensors
        required_indices = [index for index, value in enumerate(saved) if value.requires_grad]
        if not required_indices:
            return (None,) * (ctx.input_count + ctx.extra_count)
        with torch.enable_grad():
            values = [value.detach().requires_grad_(value.requires_grad) for value in saved]
            output = ctx.fn(*values)
            grads_required = torch.autograd.grad(
                output,
                [values[index] for index in required_indices],
                grad_output,
                allow_unused=True,
            )
        grads = [None] * ctx.input_count
        for index, grad in zip(required_indices, grads_required):
            grads[index] = grad
        return (*grads, *((None,) * ctx.extra_count))

    @staticmethod
    def _copy_output(result, buffers):
        """Honor the first optional pinned output buffer when supplied."""
        if buffers and isinstance(buffers[0], torch.Tensor) and buffers[0].shape == result.shape:
            buffers[0].copy_(result)
            return buffers[0]
        return result


class TransformPoint(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, points, *buffers):
        result = TransformPoint._run(transform_points, ctx, (position, quaternion, points), buffers, position, quaternion, points)
        return TransformPoint._copy_output(result, buffers)

    @staticmethod
    def backward(ctx, grad_output):
        return TransformPoint._vjp(ctx, grad_output)


class BatchTransformPoint(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, points, *buffers):
        result = BatchTransformPoint._run(batch_transform_points, ctx, (position, quaternion, points), buffers, position, quaternion, points)
        return BatchTransformPoint._copy_output(result, buffers)

    @staticmethod
    def backward(ctx, grad_output):
        return BatchTransformPoint._vjp(ctx, grad_output)


class BatchTransformPointInverse(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, points, *buffers):
        result = BatchTransformPointInverse._run(batch_transform_points_inverse, ctx, (position, quaternion, points), buffers, position, quaternion, points)
        return BatchTransformPointInverse._copy_output(result, buffers)

    @staticmethod
    def backward(ctx, grad_output):
        return BatchTransformPointInverse._vjp(ctx, grad_output)


class TransformPose(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, position2, quaternion2, *buffers):
        result_position, result_quaternion = TransformPose._run(pose_multiply, ctx, (position, quaternion, position2, quaternion2), buffers, position, quaternion, position2, quaternion2)
        if len(buffers) >= 2 and all(isinstance(value, torch.Tensor) for value in buffers[:2]):
            out_position, out_quaternion = buffers[:2]
            if out_position.shape == result_position.shape and out_quaternion.shape == result_quaternion.shape:
                out_position.copy_(result_position)
                out_quaternion.copy_(result_quaternion)
                return out_position, out_quaternion
        return result_position, result_quaternion

    @staticmethod
    def backward(ctx, grad_position, grad_quaternion):
        # ``autograd.Function`` supports tuple outputs by receiving one
        # gradient per output.  Recompute a scalar VJP for both components.
        saved = ctx.saved_tensors
        required = [index for index, value in enumerate(saved) if value.requires_grad]
        if not required:
            return (None,) * (ctx.input_count + ctx.extra_count)
        with torch.enable_grad():
            values = [value.detach().requires_grad_(value.requires_grad) for value in saved]
            result_position, result_quaternion = pose_multiply(*values)
            grads_required = torch.autograd.grad(
                (result_position, result_quaternion),
                [values[index] for index in required],
                (grad_position, grad_quaternion),
                allow_unused=True,
            )
        grads = [None] * ctx.input_count
        for index, grad in zip(required, grads_required):
            grads[index] = grad
        return (*grads, *((None,) * ctx.extra_count))


BatchTransformPose = TransformPose


class PoseInverse(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, *buffers):
        result_position, result_quaternion = PoseInverse._run(pose_inverse, ctx, (position, quaternion), buffers, position, quaternion)
        if len(buffers) >= 2 and all(isinstance(value, torch.Tensor) for value in buffers[:2]):
            out_position, out_quaternion = buffers[:2]
            if out_position.shape == result_position.shape and out_quaternion.shape == result_quaternion.shape:
                out_position.copy_(result_position)
                out_quaternion.copy_(result_quaternion)
                return out_position, out_quaternion
        return result_position, result_quaternion

    @staticmethod
    def backward(ctx, grad_position, grad_quaternion):
        saved = ctx.saved_tensors
        required = [index for index, value in enumerate(saved) if value.requires_grad]
        if not required:
            return (None,) * (ctx.input_count + ctx.extra_count)
        with torch.enable_grad():
            values = [value.detach().requires_grad_(value.requires_grad) for value in saved]
            result_position, result_quaternion = pose_inverse(*values)
            grads_required = torch.autograd.grad(
                (result_position, result_quaternion),
                [values[index] for index in required],
                (grad_position, grad_quaternion),
                allow_unused=True,
            )
        grads = [None] * ctx.input_count
        for index, grad in zip(required, grads_required):
            grads[index] = grad
        return (*grads, *((None,) * ctx.extra_count))


class QuatToMatrix(_FunctionFacade):
    @staticmethod
    def forward(ctx, quaternion, *buffers):
        result = QuatToMatrix._run(quaternion_to_matrix, ctx, (quaternion,), buffers, quaternion)
        return QuatToMatrix._copy_output(result, buffers)

    @staticmethod
    def backward(ctx, grad_output):
        return QuatToMatrix._vjp(ctx, grad_output)


class MatrixToQuaternion(_FunctionFacade):
    @staticmethod
    def forward(ctx, matrix, *buffers):
        result = MatrixToQuaternion._run(matrix_to_quaternion, ctx, (matrix,), buffers, matrix)
        return MatrixToQuaternion._copy_output(result, buffers)

    @staticmethod
    def backward(ctx, grad_output):
        return MatrixToQuaternion._vjp(ctx, grad_output)


__all__ = [name for name in globals() if not name.startswith("_")]
