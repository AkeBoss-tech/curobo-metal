"""Move a cuboid into a query sphere and restore it through update_world."""

import torch
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import DeviceCfg
from application_support import emit, tensor

def scene(x):
    return Scene(cuboid=[Cuboid(name="box", pose=[x, 0, 0, 1, 0, 0, 0], dims=[0.4, 0.4, 0.4])])

checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(scene_model=scene(3.0)))
sphere = DeviceCfg().to_device([[[[0.05, 0, 0, 0.05]]]])
far = checker.get_collision_constraint(sphere).clone()
checker.update_world(scene(0.0))
near = checker.get_collision_constraint(sphere).clone()
assert near.max().item() > far.max().item() + 0.01
checker.update_world(scene(3.0))
restored = checker.get_collision_constraint(sphere).clone()
assert torch.allclose(far, restored, atol=1e-6)
emit({"far": tensor(far), "near": tensor(near), "restored": tensor(restored)})
