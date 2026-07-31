"""Pose refinement facade over the differentiable dense-map renderer."""

from dataclasses import dataclass

import torch

from curobo_metal.ops.perception import CameraObservation


@dataclass
class BlockSparseRefinementState:
    pose: object
    loss: object = None
    iterations: int = 0

    def clone(self):
        return type(self)(
            self.pose.clone() if hasattr(self.pose, "clone") else self.pose,
            self.loss.clone() if hasattr(self.loss, "clone") else self.loss,
            self.iterations,
        )

    def copy_(self, other):
        self.pose = other.pose.clone() if hasattr(other.pose, "clone") else other.pose
        self.loss = other.loss.clone() if hasattr(other.loss, "clone") else other.loss
        self.iterations = other.iterations
        return self


@dataclass
class BlockSparseRaycastRefinerCfg:
    iterations: int = 10
    learning_rate: float = 1e-2

    @property
    def outer_iterations(self):
        return self.iterations


class BlockSparseRaycastPoseRefiner:
    def __init__(self, mapper, cfg=None):
        self.mapper = mapper
        self.cfg = cfg or BlockSparseRaycastRefinerCfg()

    def refine_pose(self, depth, intrinsics, estimated_pose):
        native = self.mapper
        if hasattr(native, "mapper"):
            native = native.mapper
        if hasattr(native, "_mapper"):
            native = native._mapper
        matrix = (
            estimated_pose.get_matrix()
            if hasattr(estimated_pose, "get_matrix")
            else torch.as_tensor(estimated_pose)
        )
        if matrix.ndim == 3:
            matrix = matrix[0]
        observation = CameraObservation(depth, intrinsics, matrix)
        return native.refine_pose(
            observation,
            iterations=self.cfg.iterations,
            learning_rate=self.cfg.learning_rate,
        )

    __call__ = refine_pose
