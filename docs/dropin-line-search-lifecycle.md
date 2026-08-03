# Portable line-search lifecycle

The gradient line-search strategy surface executes fixed candidate searches in
ordinary PyTorch on CPU and fallback-disabled float32 MPS.

## Supported behavior

- Greedy, Armijo, weak/strong Wolfe, approximate Wolfe, and approximate strong
  Wolfe selection with independent choices for every batch element.
- Deterministic first-index ties for greedy choices and deterministic largest
  accepted scale for Armijo/Wolfe choices.
- Candidate action tensors are locally autograd-enabled, so context callbacks
  can compute true gradients even when the enclosing solver invokes a search
  inside `torch.no_grad()`.
- Strict `[B,H,D]` action/direction, context batch, device, dtype, scale, and
  base-gradient validation; terminal-control locking and relative step limits.
- Reusable strategy batch-resize lifecycle via `update_num_problems`.

## Explicit boundaries

`use_cuda_kernel_line_search` belongs to the CUDA-only context and is rejected
there on the portable backend. These strategies use device-resident PyTorch
tensors rather than CUDA line-search kernels, CUDA graph buffers, Warp, or raw
packed rollout ABIs. CUDA numerical equivalence is not claimed without paired
replay evidence.
