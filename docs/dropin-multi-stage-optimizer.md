# Portable multi-stage optimizer

`curobo._src.optim.multi_stage_optimizer.MultiStageOptimizer` composes the
existing portable optimizer stages on CPU and Apple MPS.  An enabled stage
receives the previous stage's action, so particle warm-starting followed by a
gradient refinement stage uses one device-resident action chain.  Seeds accept
`[problems, horizon, action_dim]` or flattened `[problems, horizon * action_dim]`
layouts.  With the standard default `num_problems=1`, an unambiguous batched
seed establishes the runtime problem count for all stages.

Stages must have the same action size per problem, though horizon and action
dimension may be factored differently.  `reinitialize`, `shift`, batch-size
updates, rollout/goal updates, reset hooks, parameter updates, traces, and
debug dumps are broadcast across the complete stage list.  Metric evaluation
is deliberately rejected at the composite level because there is no
unambiguous rollout to use; call `compute_metrics` on the chosen stage.

The wrapper synchronizes elapsed time on MPS before publishing `solve_time`.
It has no CUDA Graph or Warp rollout-buffer implementation.  `reset_cuda_graph`
therefore clears portable stage state where necessary; direct CUDA graph and
raw CUDA/Warp APIs remain explicit unsupported boundaries.  Numerical
equivalence with NVIDIA CUDA is not claimed without paired replay evidence.
