# Third-party notices

`curobo-metal` includes a deliberately limited subset of configuration and
robot-description files from NVIDIA cuRobo. The files were copied without
modification from cuRobo commit
`8e734f3ced1df898990bcd92de40abce475907db` and were verified byte-for-byte
against that revision on 2026-08-11.

## NVIDIA cuRobo configuration files

The following files retain their original NVIDIA copyright and SPDX headers
and are licensed under Apache-2.0:

- `src/curobo/content/configs/robot/franka.yml`
- `src/curobo/content/configs/scene/collision_base_stand.yml`
- `src/curobo/content/configs/scene/collision_primitives_3d.yml`
- `src/curobo/content/configs/scene/collision_table.yml`
- `src/curobo/content/configs/scene/collision_test.yml`

Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Source: <https://github.com/NVlabs/curobo/tree/8e734f3ced1df898990bcd92de40abce475907db/curobo/content/configs>

## Franka robot description

The URDF and OBJ/DAE mesh files under
`src/curobo/content/assets/robot/franka_description/` originate from
`franka_ros` 0.7.0 and are licensed under Apache-2.0.

Copyright 2017 Franka Emika GmbH

Source: <https://github.com/frankaemika/franka_ros/tree/0.7.0>

The complete Apache-2.0 license shipped with those assets remains adjacent to
them at `src/curobo/content/assets/robot/franka_description/LICENSE`. The
project-wide Apache-2.0 text is also distributed as `LICENSE`.

Franka, NVIDIA, and cuRobo names and marks belong to their respective owners.
Their inclusion identifies provenance and does not imply endorsement.
