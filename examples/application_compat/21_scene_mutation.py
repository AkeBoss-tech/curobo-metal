"""Add, retrieve, clone, and remove named world obstacles."""

from curobo.scene import Cuboid, Scene, Sphere
from curobo.types import DeviceCfg
from application_support import emit, tensor

scene = Scene(cuboid=[Cuboid("box", [0, 0, 0, 1, 0, 0, 0], [1, 1, 1])])
scene.add_obstacle(Sphere("ball", [1, 0, 0, 1, 0, 0, 0], radius=0.2))
clone = scene.clone()
assert clone.get_obstacle("box").name == "box"
clone.remove_obstacle("ball")
assert clone.get_obstacle("ball") is None and scene.get_obstacle("ball") is not None
emit({"sentinel": tensor(DeviceCfg().to_device([1.0])),
      "count": len(scene.cuboid) + len(scene.sphere)})
