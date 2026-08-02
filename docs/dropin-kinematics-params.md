# Portable KinematicsParams

`curobo._src.robot.types.kinematics_params.KinematicsParams` exposes the
pinned V2 metadata, collision-sphere, inertial-update, clone/copy, and
CPU/MPS migration surface over the portable robot value model. The module also
re-exports the same public type names used by V2 callers: `CSpaceParams`,
`DeviceCfg`, `JointLimits`, `JointState`, `JointType`, and
`RobotCollisionGeometry`.

Sphere configurations are a `[environment, sphere, x-y-z-radius]` bank.
`disable_link_spheres()` writes V2's `-100.0` disabled-radius sentinel;
`enable_link_spheres()` restores the reference radius and
`reset_link_spheres()` restores both center and radius. Unknown links are
rejected explicitly, while a known link with no spheres returns an empty,
device-resident index tensor.

`load_cspace_cfg_from_kinematics()` supplies missing c-space defaults from
active joint limits without overwriting caller configuration: centered finite
joint positions (zero for unbounded joints), unit c-space/null-space weights,
and portable acceleration/jerk defaults.

`export_to_urdf()` is a self-contained structural exporter. It emits the
current robot name, links, inertial values, fixed/revolute/prismatic joints,
limits, mimic metadata, and—when `include_spheres=True`—the currently enabled
collision spheres. It returns XML text and optionally writes the same text to
`output_path`; it does not need `yourdfpy`, CUDA, Warp, Isaac, or a source
URDF file. Negative-radius disabled spheres are intentionally omitted because
URDF has no disabled-geometry representation. Mesh visuals, USD/Isaac assets,
raw packed CUDA FK/RNEA buffers, and conversion to `yourdfpy.URDF` remain
explicit external/CUDA boundaries.
