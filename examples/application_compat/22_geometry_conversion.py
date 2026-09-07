"""Convert analytic obstacles through public conservative approximations."""

from curobo.scene import Cuboid, Sphere
from curobo.types import DeviceCfg
from application_support import emit, tensor

box = Cuboid("box", [0, 0, 0, 1, 0, 0, 0], [0.6, 0.4, 0.8])
sphere = box.get_sphere()
cube = Sphere("ball", [0, 0, 0, 1, 0, 0, 0], radius=0.25).get_cuboid()
assert sphere.radius == 0.4
assert cube.dims == [0.5, 0.5, 0.5]
emit({"sentinel": tensor(DeviceCfg().to_device([sphere.radius])), "dims": cube.dims})
