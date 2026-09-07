
from curobo.types import DeviceCfg
import inspect

import torch

from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.particle.mppi import MPPI, MPPICfg
from curobo._src.solver.solve_mode import SolveMode, parse_solve_mode
from curobo._src.types.tool_pose import GoalToolPose
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.optim import MultiStageOptimizer


class QuadraticRollout:
    action_horizon = 1
    action_dim = 2

    def __call__(self, value):
        return (value - 2).square().sum(dim=(-1, -2))


def test_public_optimizer_signatures_and_lbfgs_behavior():
    assert list(inspect.signature(LBFGSOpt.__init__).parameters) == [
        "self", "config", "rollout_list", "use_cuda_graph"
    ]
    rollout = QuadraticRollout()
    config = LBFGSOptCfg(num_iters=20, inner_iters=5, device_cfg=DeviceCfg("cpu"))
    optimizer = LBFGSOpt(config, [rollout, rollout])
    result = optimizer.optimize(torch.zeros((3, 1, 2)))
    torch.testing.assert_close(result, torch.full_like(result, 2), atol=1e-4, rtol=1e-4)
    optimizer.reset()
    assert optimizer._cache.generation == 1


def test_particle_and_multi_stage_lifecycle():
    rollout = QuadraticRollout()
    particle = MPPI(MPPICfg(num_iters=3, num_particles=16, device_cfg=DeviceCfg("cpu")), [rollout])
    gradient = LBFGSOpt(LBFGSOptCfg(num_iters=10, inner_iters=5, device_cfg=DeviceCfg("cpu")), [rollout, rollout])
    result = MultiStageOptimizer([particle, gradient]).optimize(torch.zeros((2, 1, 2)))
    torch.testing.assert_close(result, torch.full_like(result, 2), atol=2e-3, rtol=2e-3)
    assert parse_solve_mode("multi_env") is SolveMode.MULTI_ENV


def test_franka_inverse_kinematics_round_trip():
    config = InverseKinematicsCfg.create(
        "franka.yml",
        num_seeds=2,
        use_cuda_graph=True,
        position_tolerance=0.02,
        orientation_tolerance=0.1,
        device_cfg=DeviceCfg("cpu"),
    )
    solver = InverseKinematics(config)
    default = solver.default_joint_state
    pose = solver.compute_kinematics(default).tool_poses.get_link_pose("panda_hand")
    goal = GoalToolPose.from_poses(
        {"panda_hand": pose}, ordered_tool_frames=["panda_hand"]
    )
    result = solver.solve_pose(goal, seed_config=default.position.view(1, 1, -1))
    assert bool(result.success.item())
    torch.testing.assert_close(
        result.solution[0, 0], default.position, atol=1e-5, rtol=1e-5
    )


def test_cuda_graph_request_is_explicit_for_low_level_optimizer():
    rollout = QuadraticRollout()
    try:
        LBFGSOpt(LBFGSOptCfg(device_cfg=DeviceCfg("cpu")), [rollout, rollout], use_cuda_graph=True)
    except NotImplementedError as error:
        assert "CUDA Graph" in str(error)
    else:
        raise AssertionError("CUDA graph request must fail explicitly")
