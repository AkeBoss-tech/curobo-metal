"""Fused float32 MPS collision evaluation and first-tie reductions."""

from __future__ import annotations

import torch

_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void sphere_pair_items(
    device float* distances [[buffer(0)]],
    device float* gradients [[buffer(1)]],
    const device float* spheres [[buffer(2)]],
    const device long* pairs [[buffer(3)]],
    const device bool* enabled [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& sphere_count [[buffer(6)]],
    constant uint& pair_count [[buffer(7)]],
    constant float& padding [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= batch * pair_count) return;
  uint p = index % pair_count;
  uint b = index / pair_count;
  long first = pairs[2 * p], second = pairs[2 * p + 1];
  uint gradient_base = index * sphere_count * 4;
  for (uint k = 0; k < sphere_count * 4; ++k)
    gradients[gradient_base + k] = 0.0f;
  if (!enabled[p]) {
    distances[index] = INFINITY;
    return;
  }
  const device float* a = spheres + (b * sphere_count + first) * 4;
  const device float* c = spheres + (b * sphere_count + second) * 4;
  float3 delta = float3(a[0] - c[0], a[1] - c[1], a[2] - c[2]);
  float length = metal::length(delta);
  float3 direction = length > 0.0f ? delta / length : float3(1.0f, 0.0f, 0.0f);
  distances[index] = length - a[3] - c[3] - padding;
  device float* ga = gradients + gradient_base + first * 4;
  device float* gc = gradients + gradient_base + second * 4;
  ga[0] = direction.x; ga[1] = direction.y; ga[2] = direction.z; ga[3] = -1.0f;
  gc[0] = -direction.x; gc[1] = -direction.y; gc[2] = -direction.z; gc[3] = -1.0f;
}

kernel void sphere_pair_reduce(
    device float* reduced [[buffer(0)]],
    device long* winners [[buffer(1)]],
    device float* reduced_gradients [[buffer(2)]],
    const device float* distances [[buffer(3)]],
    const device float* gradients [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& sphere_count [[buffer(6)]],
    constant uint& pair_count [[buffer(7)]],
    uint b [[thread_position_in_grid]]) {
  if (b >= batch) return;
  float best = INFINITY;
  long winner = -1;
  for (uint p = 0; p < pair_count; ++p) {
    float value = distances[b * pair_count + p];
    if (value < best) { best = value; winner = long(p); }
  }
  reduced[b] = best;
  winners[b] = winner;
  uint output_base = b * sphere_count * 4;
  if (winner < 0) {
    for (uint k = 0; k < sphere_count * 4; ++k)
      reduced_gradients[output_base + k] = 0.0f;
  } else {
    uint input_base = (b * pair_count + uint(winner)) * sphere_count * 4;
    for (uint k = 0; k < sphere_count * 4; ++k)
      reduced_gradients[output_base + k] = gradients[input_base + k];
  }
}

kernel void sphere_pair_backward(
    device float* grad_spheres [[buffer(0)]],
    const device float* gradients [[buffer(1)]],
    const device float* grad_distances [[buffer(2)]],
    const device float* grad_reduced [[buffer(3)]],
    const device long* winners [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& sphere_count [[buffer(6)]],
    constant uint& pair_count [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= batch * sphere_count * 4) return;
  uint component = index % (sphere_count * 4);
  uint b = index / (sphere_count * 4);
  float value = 0.0f;
  for (uint p = 0; p < pair_count; ++p)
    value += grad_distances[b * pair_count + p]
      * gradients[(b * pair_count + p) * sphere_count * 4 + component];
  long winner = winners[b];
  if (winner >= 0)
    value += grad_reduced[b]
      * gradients[(b * pair_count + uint(winner)) * sphere_count * 4 + component];
  grad_spheres[index] = value;
}

kernel void sphere_cuboid_items(
    device float* distances [[buffer(0)]],
    device float* gradients [[buffer(1)]],
    const device float* spheres [[buffer(2)]],
    const device float* centers [[buffer(3)]],
    const device float* rotations [[buffer(4)]],
    const device float* extents [[buffer(5)]],
    const device bool* sphere_enabled [[buffer(6)]],
    const device bool* cuboid_enabled [[buffer(7)]],
    constant uint& batch [[buffer(8)]],
    constant uint& sphere_count [[buffer(9)]],
    constant uint& cuboid_count [[buffer(10)]],
    constant float& padding [[buffer(11)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= batch * sphere_count * cuboid_count) return;
  uint c = index % cuboid_count;
  uint s = (index / cuboid_count) % sphere_count;
  uint b = index / (sphere_count * cuboid_count);
  device float* gradient = gradients + index * 4;
  if (!sphere_enabled[s] || !cuboid_enabled[c]) {
    distances[index] = INFINITY;
    gradient[0] = gradient[1] = gradient[2] = gradient[3] = 0.0f;
    return;
  }
  const device float* sphere = spheres + (b * sphere_count + s) * 4;
  const device float* center = centers + c * 3;
  const device float* rotation = rotations + c * 9;
  const device float* extent = extents + c * 3;
  float3 offset = float3(sphere[0]-center[0], sphere[1]-center[1], sphere[2]-center[2]);
  float3 local = float3(
    rotation[0]*offset.x + rotation[3]*offset.y + rotation[6]*offset.z,
    rotation[1]*offset.x + rotation[4]*offset.y + rotation[7]*offset.z,
    rotation[2]*offset.x + rotation[5]*offset.y + rotation[8]*offset.z);
  float3 q = abs(local) - float3(extent[0], extent[1], extent[2]);
  float3 outside = max(q, float3(0.0f));
  float outside_length = metal::length(outside);
  float3 sign = select(float3(1.0f), float3(-1.0f), local < 0.0f);
  float3 local_gradient;
  float box_distance;
  if (outside_length > 0.0f) {
    box_distance = outside_length;
    local_gradient = outside / outside_length * sign;
  } else {
    uint axis = q.y > q.x ? 1 : 0;
    axis = q.z > q[axis] ? 2 : axis;
    box_distance = q[axis];
    local_gradient = float3(0.0f);
    local_gradient[axis] = sign[axis];
  }
  distances[index] = box_distance - sphere[3] - padding;
  float3 world = float3(
    rotation[0]*local_gradient.x + rotation[1]*local_gradient.y + rotation[2]*local_gradient.z,
    rotation[3]*local_gradient.x + rotation[4]*local_gradient.y + rotation[5]*local_gradient.z,
    rotation[6]*local_gradient.x + rotation[7]*local_gradient.y + rotation[8]*local_gradient.z);
  gradient[0]=world.x; gradient[1]=world.y; gradient[2]=world.z; gradient[3]=-1.0f;
}

kernel void sphere_cuboid_reduce(
    device float* reduced [[buffer(0)]],
    device long* winners [[buffer(1)]],
    device float* reduced_gradients [[buffer(2)]],
    const device float* distances [[buffer(3)]],
    const device float* gradients [[buffer(4)]],
    const device bool* sphere_enabled [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& sphere_count [[buffer(7)]],
    constant uint& cuboid_count [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= batch * sphere_count) return;
  uint s = index % sphere_count;
  float best = INFINITY;
  long winner = -1;
  if (sphere_enabled[s]) {
    for (uint c = 0; c < cuboid_count; ++c) {
      float value = distances[index * cuboid_count + c];
      if (value < best) { best = value; winner = long(c); }
    }
  }
  reduced[index] = best; winners[index] = winner;
  for (uint k=0; k<4; ++k)
    reduced_gradients[index*4+k] =
      winner < 0 ? 0.0f : gradients[(index*cuboid_count+uint(winner))*4+k];
}

kernel void sphere_cuboid_backward(
    device float* grad_spheres [[buffer(0)]],
    const device float* gradients [[buffer(1)]],
    const device float* grad_distances [[buffer(2)]],
    const device float* grad_reduced [[buffer(3)]],
    const device long* winners [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& sphere_count [[buffer(6)]],
    constant uint& cuboid_count [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= batch * sphere_count * 4) return;
  uint component = index % 4;
  uint item = index / 4;
  float value = 0.0f;
  for (uint c=0; c<cuboid_count; ++c)
    value += grad_distances[item*cuboid_count+c]
      * gradients[(item*cuboid_count+c)*4+component];
  long winner = winners[item];
  if (winner >= 0)
    value += grad_reduced[item]
      * gradients[(item*cuboid_count+uint(winner))*4+component];
  grad_spheres[index] = value;
}
"""

_library = None


def _metal_library():
    global _library
    if _library is None:
        if not hasattr(torch.mps, "compile_shader"):
            raise RuntimeError("fused collision kernels require PyTorch 2.13 or newer")
        _library = torch.mps.compile_shader(_SOURCE)
    return _library


class _SphereSphere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spheres, pairs, enabled, padding):
        batch, sphere_count = spheres.shape[:2]
        pair_count = pairs.shape[0]
        distances = spheres.new_empty((batch, pair_count))
        gradients = spheres.new_empty((batch, pair_count, sphere_count, 4))
        reduced = spheres.new_empty((batch,))
        winners = torch.empty((batch,), dtype=torch.int64, device=spheres.device)
        reduced_gradients = spheres.new_empty(spheres.shape)
        if batch and pair_count:
            lib = _metal_library()
            lib.sphere_pair_items(
                distances, gradients, spheres, pairs, enabled,
                batch, sphere_count, pair_count, float(padding),
            )
            lib.sphere_pair_reduce(
                reduced, winners, reduced_gradients, distances, gradients,
                batch, sphere_count, pair_count,
            )
        else:
            reduced.fill_(torch.inf)
            winners.fill_(-1)
            reduced_gradients.zero_()
        ctx.save_for_backward(gradients, winners)
        ctx.dimensions = batch, sphere_count, pair_count
        ctx.mark_non_differentiable(gradients, winners, reduced_gradients)
        return distances, gradients, reduced, reduced_gradients, winners

    @staticmethod
    def backward(ctx, grad_distances, _gg, grad_reduced, _grg, _gw):
        gradients, winners = ctx.saved_tensors
        batch, sphere_count, pair_count = ctx.dimensions
        grad_spheres = gradients.new_empty((batch, sphere_count, 4))
        if grad_distances is None:
            grad_distances = gradients.new_zeros((batch, pair_count))
        if grad_reduced is None:
            grad_reduced = gradients.new_zeros((batch,))
        if grad_spheres.numel():
            _metal_library().sphere_pair_backward(
                grad_spheres, gradients, grad_distances.contiguous(),
                grad_reduced.contiguous(), winners, batch, sphere_count, pair_count,
            )
        return grad_spheres, None, None, None


class _SphereCuboid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spheres, centers, rotations, half, sphere_enabled, cuboid_enabled, padding):
        batch, sphere_count = spheres.shape[:2]
        cuboid_count = centers.shape[0]
        distances = spheres.new_empty((batch, sphere_count, cuboid_count))
        gradients = spheres.new_empty((batch, sphere_count, cuboid_count, 4))
        reduced = spheres.new_empty((batch, sphere_count))
        winners = torch.empty((batch, sphere_count), dtype=torch.int64, device=spheres.device)
        reduced_gradients = spheres.new_empty(spheres.shape)
        if batch and sphere_count and cuboid_count:
            lib = _metal_library()
            lib.sphere_cuboid_items(
                distances, gradients, spheres, centers, rotations, half,
                sphere_enabled, cuboid_enabled, batch, sphere_count, cuboid_count,
                float(padding),
            )
            lib.sphere_cuboid_reduce(
                reduced, winners, reduced_gradients, distances, gradients,
                sphere_enabled, batch, sphere_count, cuboid_count,
            )
        else:
            distances.fill_(torch.inf)
            gradients.zero_()
            reduced.fill_(torch.inf)
            winners.fill_(-1)
            reduced_gradients.zero_()
        ctx.save_for_backward(gradients, winners)
        ctx.dimensions = batch, sphere_count, cuboid_count
        ctx.mark_non_differentiable(gradients, winners, reduced_gradients)
        return distances, gradients, reduced, reduced_gradients, winners

    @staticmethod
    def backward(ctx, grad_distances, _gg, grad_reduced, _grg, _gw):
        gradients, winners = ctx.saved_tensors
        batch, sphere_count, cuboid_count = ctx.dimensions
        grad_spheres = gradients.new_empty((batch, sphere_count, 4))
        if grad_distances is None:
            grad_distances = gradients.new_zeros((batch, sphere_count, cuboid_count))
        if grad_reduced is None:
            grad_reduced = gradients.new_zeros((batch, sphere_count))
        if grad_spheres.numel():
            _metal_library().sphere_cuboid_backward(
                grad_spheres, gradients, grad_distances.contiguous(),
                grad_reduced.contiguous(), winners, batch, sphere_count, cuboid_count,
            )
        return grad_spheres, None, None, None, None, None, None


def sphere_sphere_metal(spheres, pairs, enabled, padding):
    return _SphereSphere.apply(
        spheres.contiguous(), pairs.contiguous(), enabled.contiguous(), padding
    )


def sphere_cuboid_metal(
    spheres, centers, rotations, half, sphere_enabled, cuboid_enabled, padding
):
    return _SphereCuboid.apply(
        spheres.contiguous(), centers.contiguous(), rotations.contiguous(),
        half.contiguous(), sphere_enabled.contiguous(), cuboid_enabled.contiguous(),
        padding,
    )
