"""Convert analytic obstacles through public conservative approximations."""

from curobo.scene import Cuboid, Sphere
from curobo.types import DeviceCfg
from application_support import emit, tensor

box = Cuboid(name="box", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.6, 0.4, 0.8])
sphere = box.get_sphere()
cube = Sphere(name="ball", pose=[0, 0, 0, 1, 0, 0, 0], radius=0.25).get_cuboid()
assert sphere.radius == 0.4
assert all(abs(value - 0.5) < 0.01 for value in cube.dims)
emit({"sentinel": tensor(DeviceCfg().to_device([sphere.radius])), "cube_valid": True})
