# Drop-in configuration and assets slice

This slice reproduces the packaged-content lookup behavior of cuRoboV2 commit
`8e734f3ced1df898990bcd92de40abce475907db`. `curobo.content` returns `pathlib.Path`
objects rooted at the installed package. The compatibility helpers in
`curobo.util_file` preserve the pinned path, YAML, file-copy, and platform helper
signatures and defaults. Missing YAML files retain Python's native
`FileNotFoundError`; unknown robot names retain the pinned inventory-bearing
error message.

## Included content

- `franka.yml`, `franka_panda.urdf`, and every OBJ/DAE mesh referenced by that
  URDF.
- The four pinned primitive scene YAML files: `collision_table.yml`,
  `collision_primitives_3d.yml`, `collision_test.yml`, and
  `collision_base_stand.yml`.
- The Franka asset license copied alongside the robot files. These files came
  from `franka_ros` 0.7.0 and are Apache-2.0 licensed; the YAMLs and helper code
  carry NVIDIA's Apache-2.0 SPDX notices.

## Deliberate limits

This first slice does not bundle the pinned Unitree or Universal Robots models,
task/solver YAMLs, XRDF files, neural weights, USD stages, voxel/ESDF maps, or
NVblox datasets. Consequently `get_task_configs_path()` is API-compatible but
its returned directory is not supplied yet, and the available robot inventory
is exactly `["franka"]`.

YAML lookup and parsing are supported through PyYAML, matching pinned cuRoboV2.
URDF, OBJ, and COLLADA/DAE files are packaged and path-resolvable, but this slice
does not claim to parse mesh geometry or XRDF/USD/NVblox formats. It imports no
CUDA, Isaac Sim, or Warp modules.

The repository's current setuptools metadata does not yet declare non-Python
package data, and changing that metadata is outside this slice's ownership.
Therefore a wheel built before the packaging slice is integrated contains the
Python helpers but omits YAML, URDF, OBJ, DAE, and license files. The packaging
owner must include `curobo/content/**/*` (and declare PyYAML, as pinned cuRoboV2
does) before claiming installed-wheel asset lookup; the source/editable layout
is complete and tested here.
