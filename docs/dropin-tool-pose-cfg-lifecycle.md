# Portable ToolPose cost configuration lifecycle

`ToolPoseCostCfg` owns a validated ordered set of tool frames and independent
`ToolPoseCriteria` objects. Explicit criteria passed by frame are retained,
new frames receive cloned defaults, and `clone()` deep-copies weights and
criteria so temporary planner changes cannot mutate a reusable configuration.

`ToolPoseCost` uses the existing differentiable PyTorch residual and goalset
selection implementation. Its public façade validates pose types, floating
tensor/device consistency, and batched integer goal indices before evaluation.
It remains composable with normal CPU/MPS autograd and scalar or vector cost
weights supplied to the portable cost stack.

CUDA/Warp pose kernels, graph capture, and raw packed-buffer ABI behavior are
not emulated; this surface intentionally retains only portable tensor
semantics.
