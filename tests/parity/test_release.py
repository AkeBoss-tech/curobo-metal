import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    import tomli as tomllib


def test_distribution_metadata_and_license():
    metadata = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    assert metadata["license"] == "Apache-2.0"
    assert metadata["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    assert metadata["readme"] == "README.md"
    assert Path("LICENSE").is_file()
    assert Path("THIRD_PARTY_NOTICES.md").is_file()
    assert Path("CHANGELOG.md").is_file()
    assert Path("SECURITY.md").is_file()


def test_top_level_import_is_torch_lazy():
    code = "import sys, curobo_metal; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_public_modules_parse_and_only_approved_robot_assets_are_vendored():
    for path in Path("src/curobo_metal").rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    approved_roots = (
        "src/curobo/content/assets/robot/",
        "src/curobo/content/configs/robot/",
    )
    asset_paths = [
        path
        for path in tracked
        if "/usd/" in path.lower()
        or path.startswith("src/curobo/content/assets/robot/")
        or path.startswith("src/curobo/content/configs/robot/")
        or path.endswith((".stl", ".dae", ".obj"))
    ]
    assert asset_paths
    assert all(path.startswith(approved_roots) for path in asset_paths)
    assert "src/curobo/content/assets/robot/franka_description/LICENSE" in tracked


def test_vendored_asset_provenance_manifest_matches_bytes():
    manifest = json.loads(Path("artifacts/release/asset-provenance.json").read_text())
    assert manifest["upstream_revision"] == "8e734f3ced1df898990bcd92de40abce475907db"
    assert manifest["verification"] == "byte-for-byte-sha256"
    content_root = Path("src/curobo/content")
    distributed_content = {
        path.as_posix()
        for path in content_root.rglob("*")
        if path.is_file() and path.name != "__init__.py" and "__pycache__" not in path.parts
    }
    assert set(manifest["files"]) == distributed_content
    for relative_path, expected_hash in manifest["files"].items():
        payload = Path(relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_hash, relative_path
