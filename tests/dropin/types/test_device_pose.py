import pytest
import torch

from curobo.types import DeviceCfg, Pose


def test_device_cfg_conversion_and_no_silent_fallback() -> None:
    cfg = DeviceCfg(torch.device("cpu"), torch.float64)
    value = cfg.to_device([1, 2])
    assert value.device.type == "cpu"
    assert value.dtype == torch.float64
    assert cfg.is_same_torch_device(torch.device("cpu"))

    unavailable = DeviceCfg(torch.device("cuda", 0))
    if not torch.cuda.is_available():
        with pytest.raises((AssertionError, RuntimeError)):
            unavailable.to_device([1.0])


def test_pose_shapes_clone_index_and_mutating_to() -> None:
    pose = Pose.from_batch_list(
        [[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1]],
        q_xyzw=True,
    )
    assert pose.position.shape == (2, 3)
    assert pose.quaternion.shape == (2, 4)
    assert torch.equal(pose.quaternion[:, 0], pose.quaternion.new_ones(2))

    clone = pose.clone()
    assert isinstance(clone, Pose)
    assert clone.position.data_ptr() != pose.position.data_ptr()
    indexed = pose[1]
    assert isinstance(indexed, Pose)
    assert indexed.position.shape == (1, 3)

    returned = pose.to(device=torch.device("cpu"))
    assert returned is pose
    with pytest.raises(ValueError, match="requires device_cfg or device"):
        pose.to()


def test_pose_matrix_and_composition_behavior() -> None:
    left = Pose.from_list([1, 0, 0, 1, 0, 0, 0])
    right = Pose.from_list([0, 2, 0, 1, 0, 0, 0])
    result = left.multiply(right)

    assert isinstance(result, Pose)
    assert torch.allclose(result.position, result.position.new_tensor([[1.0, 2.0, 0.0]]))
    assert result.get_matrix().shape == (1, 4, 4)
    assert torch.allclose(result.inverse().multiply(result).position, result.position.new_zeros(1, 3))
