# Portable support-polygon cost

`curobo._src.cost.cost_support_polygon.CostSupportPolygon` is implemented
with ordinary PyTorch tensors on CPU and Apple Metal.  It keeps the pinned V2
contract: foot-sphere XY positions from the first horizon step form a detached,
per-batch convex hull; a centre-of-mass outside that hull receives its positive
signed distance, and an optional small interior-margin cost encourages it away
from a support boundary.

The centre-of-mass path is differentiable.  Hull formation is intentionally
detached because the discrete contact ordering is not differentiable in the V2
implementation either.  A cached hull is reused only for a compatible
batch/device/dtype; changing those conditions triggers a rebuild from configured
`foot_sphere_indices`, or raises if a caller supplied only a manual hull.

`foot_link_names` remains configuration metadata: resolving links to sphere
indices requires a robot kinematics configuration and is performed by the
rollout builder, not by this tensor cost.  CUDA/Warp kernels are not used or
claimed.
