import torch


class StateCSpaceFunction(torch.autograd.Function):
    """Boundary for cuRobo's raw Warp packed-buffer state-cost bridge."""

    @staticmethod
    def forward(ctx, *args):
        raise RuntimeError("StateCSpaceFunction is a raw Warp/CUDA ABI; use StateCSpaceCost")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("StateCSpaceFunction is a raw Warp/CUDA ABI; use StateCSpaceCost")

def forward_cspace_state_warp(*args,**kwargs):
    raise RuntimeError("forward_cspace_state_warp is unavailable on CPU/MPS; use StateCSpaceCost")
__all__=["StateCSpaceFunction","forward_cspace_state_warp"]
