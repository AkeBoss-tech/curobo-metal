# Portable robot-scene collision behavior

`RobotSceneCollisionCfg.load_from_config` now constructs real device-resident
Halton sampling, joint-bound, self-collision, scene-cost, and scene-constraint
components.  `RobotSceneCollision` shares their batch lifecycle with its public
collision buffer and accepts either one scene or one scene per configured
environment in `update_world`.

All scene-distance, constraints, self-collision, sampling, gradients, and
environment routing run through the portable CPU/MPS PyTorch collision backend.
The checker intentionally retains explicit boundaries for raw Warp kernel entry
points, Warp BVHs (meshes use vectorized triangle scans), and analytic
continuous collision detection (swept checks are sampled).
