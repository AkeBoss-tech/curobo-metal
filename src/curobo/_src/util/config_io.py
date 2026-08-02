"""Pinned cuRoboV2 configuration I/O compatibility.

This module deliberately mirrors the historical internal import surface.  A
number of applications import the type aliases from here, even though the
actual implementations live in :mod:`curobo.util_file`.
"""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Union

import yaml
from yaml import CLoader as Loader

from curobo.util_file import (
    ConfigT,
    copy_file_to_path,
    create_dir_if_not_exists,
    file_exists,
    get_filename,
    get_files_from_dir,
    get_path_of_dir,
    is_file_xrdf,
    is_platform_linux,
    is_platform_windows,
    join_path,
    load_yaml,
    merge_dict_a_into_b,
    resolve_config,
    write_yaml,
)

__all__ = [
    "ConfigT",
    "copy_file_to_path",
    "create_dir_if_not_exists",
    "file_exists",
    "get_filename",
    "get_files_from_dir",
    "get_path_of_dir",
    "is_file_xrdf",
    "is_platform_linux",
    "is_platform_windows",
    "join_path",
    "load_yaml",
    "merge_dict_a_into_b",
    "resolve_config",
    "write_yaml",
]
