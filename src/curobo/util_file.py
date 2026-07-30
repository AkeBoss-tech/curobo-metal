# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility file and packaged-content helpers from pinned cuRoboV2."""

import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Union

import yaml
from yaml import CLoader as Loader

from curobo.content import (
    get_assets_path,
    get_configs_path,
    get_content_root,
    get_robot_configs_path,
    get_robot_path,
    get_scene_configs_path,
    get_task_configs_path,
    list_available_robots,
)

Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
    [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def join_path(path1: Union[str, Path], path2: Union[str, Path]) -> str:
    if isinstance(path1, Path):
        path1 = str(path1)
    if isinstance(path2, Path):
        path2 = str(path2)
    if not isinstance(path2, str):
        return path2
    return os.path.join(path1, path2)


ConfigT = TypeVar("ConfigT")


def resolve_config(config: Union[str, ConfigT]) -> Union[Dict, ConfigT]:
    if isinstance(config, str):
        with open(config) as file_p:
            return yaml.load(file_p, Loader=Loader)
    return config


def load_yaml(file_path: Union[str, Dict]) -> Dict:
    return resolve_config(file_path)


def write_yaml(data: Dict, file_path: str):
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def copy_file_to_path(source_file: str, destination_path: str) -> str:
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    _, file_name = os.path.split(source_file)
    new_path = join_path(destination_path, file_name)
    if not os.path.exists(new_path):
        shutil.copyfile(source_file, new_path)
    return new_path


def get_filename(file_path: str, remove_extension: bool = False) -> str:
    _, file_name = os.path.split(file_path)
    if remove_extension:
        file_name = os.path.splitext(file_name)[0]
    return file_name


def get_path_of_dir(file_path: str) -> str:
    dir_path, _ = os.path.split(file_path)
    return dir_path


def get_files_from_dir(dir_path, extension: List[str], contains: str) -> List[str]:
    file_names = [
        fn
        for fn in os.listdir(dir_path)
        if any(fn.endswith(ext) for ext in extension) and contains in fn
    ]
    file_names.sort()
    return file_names


def file_exists(path: str) -> bool:
    if path is None:
        return False
    return os.path.exists(path)


def merge_dict_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in a.items():
        if isinstance(value, dict):
            merge_dict_a_into_b(value, b[key])
        else:
            b[key] = value
    return b


def is_platform_windows() -> bool:
    return sys.platform == "win32"


def is_platform_linux() -> bool:
    return sys.platform == "linux"


def is_file_xrdf(file_path: str) -> bool:
    return file_path.endswith(".xrdf") or file_path.endswith(".XRDF")


def create_dir_if_not_exists(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
