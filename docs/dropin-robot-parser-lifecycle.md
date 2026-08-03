# Portable robot parser lifecycle

`UrdfRobotParser` now retains the V2 parent-map shape (`_parent_map`) alongside
the portable `link_parent` projection.  It exposes deterministic chains,
extra-link overlays, raw joint identifiers, continuous-joint normalization,
finite defaults for omitted velocity/effort limits, and canonical positive
axis records with the sign represented in `joint_offset`.

URDF primitive visual/collision geometry is usable without external runtime
dependencies: sphere, box, and cylinder records preserve their full `xyz +
wxyz` origin pose.  Static inertial metadata retains authored mass/COM/inertia
and uses V2's small finite defaults for zero or omitted inertial values.

File-backed meshes, scene graphs, USD/Isaac, and trimesh asset construction
remain deliberately unavailable.  Parser operations that require one raise a
specific `NotImplementedError` rather than pretending a missing mesh is a
portable collision object.
