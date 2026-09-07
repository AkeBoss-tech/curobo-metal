"""Query a sphere world, then move it away through the public world API."""

import torch
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.scene import Scene, Sphere
from curobo.types import DeviceCfg
from application_support import emit, tensor

def world(x):
    return Scene(sphere=[Sphere("ball", [x, 0, 0, 1, 0, 0, 0], radius=0.2)])

checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(scene_model=world(0.0)))
query = DeviceCfg().to_device([[[[0.1, 0, 0, 0.05]]]])
near = checker.get_collision_constraint(query).clone()
checker.update_world(world(2.0))
far = checker.get_collision_constraint(query).clone()
assert near.max() > far.max()
emit({"near": tensor(near), "far": tensor(far)})
