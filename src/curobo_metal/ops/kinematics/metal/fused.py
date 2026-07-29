"""Single-dispatch Metal forward kinematics with a custom VJP."""

from __future__ import annotations

import torch

_MAX_DOF = 64

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint MAX_DOF = 64;

inline void matmul4(thread const float *a, thread const float *b, thread float *out) {
  for (uint r = 0; r < 4; ++r) {
    for (uint c = 0; c < 4; ++c) {
      float value = 0.0f;
      for (uint k = 0; k < 4; ++k) value += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = value;
    }
  }
}

kernel void fk_forward(
    const device float *q [[buffer(0)]],
    const device float *origins [[buffer(1)]],
    const device float *axes [[buffer(2)]],
    const device int *kinds [[buffer(3)]],
    const device int *q_indices [[buffer(4)]],
    device float *transforms [[buffer(5)]],
    device float *jacobian [[buffer(6)]],
    device float *geometric [[buffer(7)]],
    constant uint& batch [[buffer(8)]],
    constant uint& links [[buffer(9)]],
    constant uint& dof [[buffer(10)]],
    uint b [[thread_position_in_grid]]) {
  if (b >= batch) return;

  float current[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  float derivatives[MAX_DOF * 16];
  for (uint i = 0; i < dof * 16; ++i) derivatives[i] = 0.0f;

  for (uint link = 0; link < links; ++link) {
    const int kind = kinds[link]; // 0 fixed, 1 revolute, 2 prismatic
    const int qi = q_indices[link];
    const float position = qi < 0 ? 0.0f : q[b * dof + uint(qi)];
    const device float *axis = axes + link * 3;
    float motion[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float dmotion[16] = {0};

    if (kind == 2) {
      motion[3] = position * axis[0];
      motion[7] = position * axis[1];
      motion[11] = position * axis[2];
      dmotion[3] = axis[0]; dmotion[7] = axis[1]; dmotion[11] = axis[2];
    } else if (kind == 1) {
      const float x = axis[0], y = axis[1], z = axis[2];
      const float s = sin(position), c = cos(position), v = 1.0f - c;
      motion[0]=c+x*x*v; motion[1]=x*y*v-z*s; motion[2]=x*z*v+y*s;
      motion[4]=y*x*v+z*s; motion[5]=c+y*y*v; motion[6]=y*z*v-x*s;
      motion[8]=z*x*v-y*s; motion[9]=z*y*v+x*s; motion[10]=c+z*z*v;
      dmotion[0]=-s+x*x*s; dmotion[1]=x*y*s-z*c; dmotion[2]=x*z*s+y*c;
      dmotion[4]=y*x*s+z*c; dmotion[5]=-s+y*y*s; dmotion[6]=y*z*s-x*c;
      dmotion[8]=z*x*s-y*c; dmotion[9]=z*y*s+x*c; dmotion[10]=-s+z*z*s;
    }

    float origin[16], local[16], dlocal[16], next[16], direct[16];
    for (uint i = 0; i < 16; ++i) origin[i] = origins[link * 16 + i];
    matmul4(origin, motion, local);
    matmul4(origin, dmotion, dlocal);
    matmul4(current, local, next);
    if (qi >= 0) matmul4(current, dlocal, direct);

    for (uint j = 0; j < dof; ++j) {
      float prior[16], propagated[16];
      for (uint i = 0; i < 16; ++i) prior[i] = derivatives[j * 16 + i];
      matmul4(prior, local, propagated);
      for (uint i = 0; i < 16; ++i)
        derivatives[j * 16 + i] = propagated[i] + ((int(j) == qi) ? direct[i] : 0.0f);
    }
    for (uint i = 0; i < 16; ++i) {
      current[i] = next[i];
      transforms[(b * links + link) * 16 + i] = next[i];
    }

    for (uint j = 0; j < dof; ++j) {
      for (uint r = 0; r < 4; ++r) for (uint c2 = 0; c2 < 4; ++c2)
        jacobian[((((b * links + link) * 4 + r) * 4 + c2) * dof) + j] =
            derivatives[j * 16 + r * 4 + c2];
      for (uint r = 0; r < 3; ++r)
        geometric[(((b * links + link) * 6 + r) * dof) + j] =
            derivatives[j * 16 + r * 4 + 3];
      const float wx = derivatives[j * 16 + 2 * 4 + 0] * next[4] +
                       derivatives[j * 16 + 2 * 4 + 1] * next[5] +
                       derivatives[j * 16 + 2 * 4 + 2] * next[6];
      const float wy = derivatives[j * 16 + 0 * 4 + 0] * next[8] +
                       derivatives[j * 16 + 0 * 4 + 1] * next[9] +
                       derivatives[j * 16 + 0 * 4 + 2] * next[10];
      const float wz = derivatives[j * 16 + 1 * 4 + 0] * next[0] +
                       derivatives[j * 16 + 1 * 4 + 1] * next[1] +
                       derivatives[j * 16 + 1 * 4 + 2] * next[2];
      geometric[(((b * links + link) * 6 + 3) * dof) + j] = wx;
      geometric[(((b * links + link) * 6 + 4) * dof) + j] = wy;
      geometric[(((b * links + link) * 6 + 5) * dof) + j] = wz;
    }
  }
}

kernel void fk_backward(
    const device float *grad_transform [[buffer(0)]],
    const device float *jacobian [[buffer(1)]],
    device float *grad_q [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& links [[buffer(4)]],
    constant uint& dof [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= batch * dof) return;
  const uint b = index / dof, j = index % dof;
  float value = 0.0f;
  for (uint link = 0; link < links; ++link)
    for (uint i = 0; i < 16; ++i)
      value += grad_transform[(b * links + link) * 16 + i] *
               jacobian[((b * links + link) * 16 + i) * dof + j];
  grad_q[index] = value;
}
"""

_library = None


def _metal_library():
    global _library
    if _library is None:
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("fused Metal FK requires torch.mps.compile_shader")
        _library = torch.mps.compile_shader(_SOURCE)
    return _library


def supports_fused_chain(chain) -> bool:
    return (
        chain.device.type == "mps"
        and chain.dtype == torch.float32
        and chain.dof <= _MAX_DOF
    )


class _FusedFK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, origins, axes, kinds, q_indices):
        batch, dof = q.shape
        links = origins.shape[0]
        transforms = q.new_empty((batch, links, 4, 4))
        jacobian = q.new_empty((batch, links, 4, 4, dof))
        geometric = q.new_empty((batch, links, 6, dof))
        if batch:
            _metal_library().fk_forward(
                q,
                origins,
                axes,
                kinds,
                q_indices,
                transforms,
                jacobian,
                geometric,
                batch,
                links,
                dof,
                threads=batch,
            )
        ctx.save_for_backward(jacobian)
        ctx.shape = (batch, links, dof)
        ctx.mark_non_differentiable(jacobian, geometric)
        return transforms, jacobian, geometric

    @staticmethod
    def backward(ctx, grad_transform, _grad_jacobian, _grad_geometric):
        (jacobian,) = ctx.saved_tensors
        batch, links, dof = ctx.shape
        grad_q = jacobian.new_empty((batch, dof))
        if batch:
            _metal_library().fk_backward(
                grad_transform.contiguous(),
                jacobian,
                grad_q,
                batch,
                links,
                dof,
                threads=batch * dof,
            )
        return grad_q, None, None, None, None


def fused_forward_kinematics(chain, q: torch.Tensor):
    """Return fused Metal tensors for validated, batched ``q``."""
    if not supports_fused_chain(chain):
        raise ValueError("chain is not supported by fused Metal FK")
    q = q.contiguous()
    return _FusedFK.apply(
        q, chain.origins, chain.axes, chain._metal_kinds, chain._metal_q_indices
    )
