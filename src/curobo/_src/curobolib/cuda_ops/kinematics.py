"""Compatibility facade for cuRobo's fused CUDA kinematics autograd function."""

import torch


class KinematicsFusedFunction(torch.autograd.Function):
    @staticmethod
    def create_buffers(*args, **kwargs):
        raise NotImplementedError(
            "Raw CUDA kinematics buffers are unavailable on Metal; use "
            "curobo.kinematics.Kinematics, which owns portable output buffers"
        )

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError(
            "KinematicsFusedFunction consumes CUDA-specific packed buffers; use "
            "curobo.kinematics.Kinematics for CPU/MPS FK, spheres, and Jacobians"
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        del ctx, grad_outputs
        raise NotImplementedError(
            "KinematicsFusedFunction's CUDA packed-buffer VJP is unavailable; "
            "use curobo.kinematics.Kinematics with ordinary PyTorch autograd"
        )
