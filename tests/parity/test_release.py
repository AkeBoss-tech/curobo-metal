import ast
import subprocess
import sys
import tomllib
from pathlib import Path


def test_distribution_metadata_and_license():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert metadata["license"] == "Apache-2.0"
    assert metadata["license-files"] == ["LICENSE"]
    assert metadata["readme"] == "README.md"
    assert Path("LICENSE").is_file()


def test_top_level_import_is_torch_lazy():
    code = "import sys, curobo_metal; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_modules_parse_and_only_approved_robot_assets_are_vendored():
    for path in Path("src/curobo_metal").rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    approved_roots = (
        "src/curobo/content/assets/robot/franka_description/",
        "src/curobo/content/configs/robot/",
    )
    asset_paths = [
        path
        for path in tracked
        if "/usd/" in path.lower()
        or "/robot/" in path.lower()
        or path.endswith((".stl", ".dae", ".obj"))
    ]
    assert asset_paths
    assert all(path.startswith(approved_roots) for path in asset_paths)
    assert f"{approved_roots[0]}LICENSE" in tracked
