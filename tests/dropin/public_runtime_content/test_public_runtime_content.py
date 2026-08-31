"""Consumer-level tests for the public config/content/runtime convenience API."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import curobo._src.runtime as internal_runtime
from curobo.config_io import copy_file_to_path, load_yaml, resolve_config, write_yaml
from curobo.content import (
    get_assets_path,
    get_configs_path,
    get_content_root,
    get_robot_configs_path,
    get_scene_configs_path,
    get_task_configs_path,
)
from curobo.profiling import CudaEventTimer
from curobo.scene import Cuboid, Scene
from curobo.sphere_fit import SphereFitType
from curobo.types import ContentPath, DeviceCfg
from curobo.viewer import UsdWriter, ViserVisualizer


def test_content_and_content_path_are_installed_package_relative() -> None:
    root = get_content_root()
    assert root.is_dir()
    assert get_assets_path() == root / "assets"
    assert get_configs_path() == root / "configs"
    assert get_robot_configs_path() == root / "configs" / "robot"
    assert get_scene_configs_path() == root / "configs" / "scene"
    assert get_task_configs_path() == root / "configs" / "task"
    assert (get_task_configs_path() / "metrics_base.yml").is_file()
    assert len(list(get_task_configs_path().rglob("*.yml"))) == 13
    assert ContentPath().robot_config_root_path == get_robot_configs_path()


def test_public_config_and_scene_types_support_normal_consumer_flow(tmp_path: Path) -> None:
    source = tmp_path / "scene.yml"
    write_yaml({"cuboid": {"table": {"dims": [1.0, 2.0, 0.1]}}}, str(source))
    copied = Path(copy_file_to_path(str(source), str(tmp_path / "copied")))
    parsed = load_yaml(str(copied))
    assert resolve_config(parsed) is parsed
    scene = Scene(
        cuboid=[
            Cuboid(
                name="table",
                dims=parsed["cuboid"]["table"]["dims"],
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            )
        ]
    )
    assert scene.cuboid[0].name == "table"
    assert DeviceCfg().device.type == "cpu"
    assert SphereFitType.MORPHIT.value == "morphit"


def test_public_timer_uses_seconds_and_honors_runtime_disable() -> None:
    prior = internal_runtime.cuda_event_timers
    try:
        internal_runtime.cuda_event_timers = True
        timer = CudaEventTimer().start()
        time.sleep(0.003)
        elapsed = timer.stop()
        assert 0.001 <= elapsed < 1.0
        assert timer.elapsed_seconds == elapsed

        internal_runtime.cuda_event_timers = False
        assert CudaEventTimer().start().stop() == 0.0
    finally:
        internal_runtime.cuda_event_timers = prior


@pytest.mark.parametrize("factory, dependency", [(UsdWriter, "usd-core"), (ViserVisualizer, "Viser")])
def test_optional_public_viewers_fail_explicitly_without_extra_dependencies(factory, dependency: str) -> None:
    with pytest.raises((ImportError, NotImplementedError), match=dependency):
        factory()
