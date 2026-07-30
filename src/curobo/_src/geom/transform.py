"""Portable differentiable pose and point transforms (wxyz quaternions)."""

from typing import Optional, Tuple
import torch

from .quaternion import normalize_quaternion, quat_multiply


def torch_quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
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


transform_point_inverse = batch_transform_points_inverse
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
get_inv_transform = pose_inverse
compute_quat_to_matrix = quaternion_to_matrix
compute_matrix_to_quat = matrix_to_quaternion


def quaternion_rate_to_axis_angle_rate(quaternion_rate, current_quaternion):
    inverse = torch.cat((current_quaternion[..., :1], -current_quaternion[..., 1:]), -1)
    return 2.0 * quat_multiply(quaternion_rate, inverse)[..., 1:]


class _FunctionFacade(torch.autograd.Function):
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("direct Function backward is unsupported; call the differentiable helper")


class TransformPoint(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, points, *buffers):
        return transform_points(position, quaternion, points)


BatchTransformPoint = TransformPoint


class BatchTransformPointInverse(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, points, *buffers):
        return batch_transform_points_inverse(position, quaternion, points)


class TransformPose(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, position2, quaternion2, *buffers):
        return pose_multiply(position, quaternion, position2, quaternion2)


BatchTransformPose = TransformPose


class PoseInverse(_FunctionFacade):
    @staticmethod
    def forward(ctx, position, quaternion, *buffers):
        return pose_inverse(position, quaternion)


class QuatToMatrix(_FunctionFacade):
    @staticmethod
    def forward(ctx, quaternion, *buffers):
        return quaternion_to_matrix(quaternion)


class MatrixToQuaternion(_FunctionFacade):
    @staticmethod
    def forward(ctx, matrix, *buffers):
        return matrix_to_quaternion(matrix)


__all__ = [name for name in globals() if not name.startswith("_")]
