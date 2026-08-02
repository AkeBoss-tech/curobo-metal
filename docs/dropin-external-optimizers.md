# External optimizer compatibility

`curobo._src.optim.external.torch_opt.TorchOpt` adapts a normal
`torch.optim` class to an eager differentiable rollout.  Its action, best-cost,
debug, reset, goal-update, and multi-problem lifecycle is device-resident on
CPU or Apple Metal.  `TorchOptCfg.torch_optim_class` takes precedence over its
name, matching the pinned public configuration behavior.

`ScipyOpt` evaluates a differentiable rollout on the configured CPU/MPS
device, transfers only objective/gradient values to `scipy.optimize.minimize`,
then returns the solution on that device.  It supports one problem at a time,
rollout bounds, and cuRobo-style positive constraint-violation costs.  SciPy
is optional: without it, the wrapper reports a deterministic PyTorch LBFGS
fallback in `last_result` rather than making SciPy a package requirement.

The `CudaGraphScipyOpt` import alias is retained, but requesting
`use_cuda_graph=True` raises `UnsupportedExternalOptimizerFeature`.  CUDA graph
capture, raw CUDA/Warp kernels, and NVIDIA numerical equivalence are not
implemented or claimed by this Metal port.
