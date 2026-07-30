from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[2]


def test_upstream_shaped_fk_example_runs_against_portable_fixture() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/upstream_forward_kinematics_mps.py"),
            str(ROOT / "tests/compat/types_config/fixtures/tiny_xrdf_robot.yml"),
            "--device",
            "cpu",
            "--batch-size",
            "32",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Reference comparison:" in result.stdout
    assert "  PASS" in result.stdout
