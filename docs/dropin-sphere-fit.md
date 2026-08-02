# Portable sphere-fit lifecycle

`curobo.sphere_fit.fit_spheres_to_mesh` keeps the pinned V2 dispatcher:
`surface`, `voxel`, and `morphit` fit modes, automatic sphere counts, optional
quality metrics, clip planes, histories, and diagnostics all work with CPU and
Metal tensors.

`surface` uses deterministic farthest-point sampling. `voxel` uses the
portable vectorised signed triangle query for caller-declared watertight meshes.
`morphit` retains voxel seeding and result/history lifecycle, but it is a
portable approximation: CUDA/Warp MorphIt optimisation parity is not claimed.

Triangle-free vertex clouds are useful for attached primitive approximations;
they fall back deterministically to surface spheres and report the fallback in
`result.debug_info`. Coverage/protrusion metrics require triangle topology and
are intentionally not invented for those vertex clouds. Raw Warp mesh IDs,
BVH acceleration, and external MorphIt mesh workflows remain outside this
portable boundary.
