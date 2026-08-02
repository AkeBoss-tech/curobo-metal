import torch


class PositionCSpaceFunction(torch.autograd.Function):
    """Boundary for cuRobo's raw Warp packed-buffer autograd bridge.

    Use :class:`PositionCSpaceCost` for the differentiable CPU/MPS operation.
    The raw function's CUDA buffer ABI is deliberately not emulated.
    """

    @staticmethod
    def forward(ctx, *args):
        raise RuntimeError("PositionCSpaceFunction is a raw Warp/CUDA ABI; use PositionCSpaceCost")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("PositionCSpaceFunction is a raw Warp/CUDA ABI; use PositionCSpaceCost")

def forward_cspace_position_warp(*args,**kwargs):
    raise RuntimeError("forward_cspace_position_warp is unavailable on CPU/MPS; use PositionCSpaceCost")
__all__=["PositionCSpaceFunction","forward_cspace_position_warp"]
