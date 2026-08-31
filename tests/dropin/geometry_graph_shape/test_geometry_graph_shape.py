"""Portable graph/geometry API-shape and tensor-behavior coverage."""

from __future__ import annotations

import inspect

import pytest
import torch

from curobo._src.geom.cv import (
    extract_depth_from_structured_pointcloud,
    get_projection_rays,
    project_depth_to_pointcloud,
    project_depth_using_rays,
)
from curobo._src.geom.quaternion import quat_multiply
from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator
from curobo._src.graph_planner.graph.node_sampling_strategy import NodeSamplingStrategy
from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner


def test_pinned_graph_helper_shapes() -> None:
    assert list(inspect.signature(DistanceNeighborCalculator.jit_get_unique_nodes).parameters) == [
        "nodes", "distance_weight", "node_similarity_threshold", "action_dim"
    ]
    assert list(inspect.signature(NodeSamplingStrategy.jit_transform_unit_ball_to_ellipsoid_svd).parameters) == [
        "x_start", "x_goal", "distance_weight", "max_sampling_radius", "action_dim",
        "rot_frame_col", "unit_ball_samples", "low_bounds", "high_bounds",
    ]
    assert list(inspect.signature(PRMGraphPlanner.extend_roadmap_with_ellipsoidal_samples).parameters) == [
        "self", "x_start", "x_goal", "max_sampling_radius", "num_samples", "neighbors_per_node"
    ]


def test_portable_graph_helpers_are_deterministic_and_differentiable() -> None:
    nodes = torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]], requires_grad=True)
    unique, index = DistanceNeighborCalculator.jit_get_unique_nodes(
        nodes, torch.ones(2), 1e-6, 2
    )
    assert index.tolist() == [0, 2]
    unique.sum().backward()
    assert nodes.grad is not None

    samples = torch.tensor([[1.0, -1.0]], requires_grad=True)
    mapped = NodeSamplingStrategy.jit_transform_unit_ball_to_ellipsoid_householder(
        torch.zeros(2), torch.ones(2), torch.ones(2), torch.tensor(0.5), 2,
        torch.eye(2), samples, torch.full((2,), -1.0), torch.full((2,), 1.0),
    )
    assert mapped.shape == (1, 2)
    assert torch.isfinite(mapped).all()
    assert (mapped >= -1.0).all() and (mapped <= 1.0).all()
    mapped.sum().backward()
    assert samples.grad is not None


def test_portable_camera_and_quaternion_output_buffers() -> None:
    intrinsics = torch.tensor([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    rays = get_projection_rays(2, 2, intrinsics)
    points = project_depth_using_rays(torch.ones(2, 2), rays)
    assert points.shape == (1, 4, 3)
    structured = project_depth_to_pointcloud(torch.ones(2, 2), intrinsics)
    assert structured.shape == (2, 2, 3)
    output = torch.empty(2, 2)
    torch.testing.assert_close(extract_depth_from_structured_pointcloud(structured, output), output)

    result_buffer = torch.empty(4)
    result = quat_multiply(torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0, 0.0]), result_buffer)
    assert result.data_ptr() == result_buffer.data_ptr()
    torch.testing.assert_close(result, torch.tensor([0.0, 1.0, 0.0, 0.0]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_graph_helpers_run_without_cpu_fallback_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    nodes = torch.tensor([[0.0], [1.0]], device="mps")
    unique, _ = DistanceNeighborCalculator.jit_get_unique_nodes(nodes, torch.ones(1, device="mps"), 0.0, 1)
    assert unique.device.type == "mps"
