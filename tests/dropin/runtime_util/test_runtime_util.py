from __future__ import annotations

import logging
from pathlib import Path

import pytest

import curobo.runtime as runtime
from curobo._src.context import CuroboRuntime, get_runtime
from curobo._version import __version__
from curobo.config_io import (
    get_filename,
    get_files_from_dir,
    join_path,
    load_yaml,
    merge_dict_a_into_b,
    resolve_config,
    write_yaml,
)
from curobo.logging import log_and_raise, log_info, setup_logger


def test_config_io_round_trip_and_pinned_helpers(tmp_path: Path):
    path = tmp_path / "config.yml"
    write_yaml({"outer": {"value": 2}}, str(path))
    assert load_yaml(str(path)) == {"outer": {"value": 2}}
    assert resolve_config({"ready": True}) == {"ready": True}
    assert get_filename(str(path), remove_extension=True) == "config"
    assert join_path(tmp_path, "config.yml") == str(path)
    assert get_files_from_dir(tmp_path, [".yml"], "config") == ["config.yml"]
    base = {"outer": {"value": 1, "kept": True}}
    assert merge_dict_a_into_b({"outer": {"value": 3}}, base) == {
        "outer": {"value": 3, "kept": True}
    }


def test_logging_and_exception_type(caplog: pytest.LogCaptureFixture):
    setup_logger("info", "curobo-test")
    with caplog.at_level(logging.INFO, logger="curobo-test"):
        log_info("portable", logger_name="curobo-test")
    assert "portable" in caplog.text
    with pytest.raises(RuntimeError, match="failure"):
        log_and_raise(
            "failure",
            logger_name="curobo-test",
            exc_info=False,
            exception_type=RuntimeError,
        )


def test_runtime_flags_context_and_version():
    assert isinstance(runtime.cuda_graphs, bool)
    assert isinstance(runtime.cache_dir, str)
    assert get_runtime() is get_runtime()
    assert isinstance(get_runtime(), CuroboRuntime)
    with pytest.raises(NotImplementedError, match="NVIDIA CUDA"):
        get_runtime().get_cuda_core_cache()
    assert isinstance(__version__, str) and __version__
