"""Construct a sphere obstacle and preserve its public calibrated values."""

from curobo.scene import Sphere
from curobo.types import DeviceCfg
from application_support import emit, tensor

sphere = Sphere(name="ball", pose=[0.1, 0.2, 0.3, 1, 0, 0, 0], radius=0.25)
assert sphere.name == "ball"
assert sphere.position == [0.1, 0.2, 0.3]
assert sphere.pose == [0.1, 0.2, 0.3, 1, 0, 0, 0]
emit({"position": tensor(DeviceCfg().to_device(sphere.position)), "radius": sphere.radius})
