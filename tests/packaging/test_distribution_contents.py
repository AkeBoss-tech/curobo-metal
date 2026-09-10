from pathlib import Path

from tools.packaging.check_distribution_contents import validate_members


def test_repository_license_and_notices_are_complete() -> None:
    root = Path(__file__).resolve().parents[2]
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    notices = (root / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in license_text
    assert "9. Accepting Warranty or Additional Liability" in license_text
    assert "APPENDIX: How to apply the Apache License" in license_text
    assert "NVIDIA CORPORATION & AFFILIATES" in notices
    assert "Franka Emika GmbH" in notices


def test_wheel_requires_both_license_documents_and_rejects_leaks() -> None:
    wheel = Path("curobo_metal-1.0.1-py3-none-any.whl")
    valid = [
        "curobo/__init__.py",
        "curobo/types/math.py",
        "curobo/wrap/reacher/motion_gen.py",
        "curobo_metal/ops/perception/core.py",
        "curobo_metal-1.0.1.dist-info/licenses/LICENSE",
        "curobo_metal-1.0.1.dist-info/licenses/THIRD_PARTY_NOTICES.md",
    ]
    assert validate_members(wheel, valid) == []

    errors = validate_members(wheel, valid + ["tests/test_package.py", "curobo/__pycache__/x.pyc"])
    assert any("non-runtime tree leaked" in error for error in errors)
    assert any("generated or stale" in error for error in errors)


def test_sdist_requires_single_root_and_release_documents() -> None:
    sdist = Path("curobo_metal-1.0.1.tar.gz")
    valid = [
        "curobo_metal-1.0.1/LICENSE",
        "curobo_metal-1.0.1/THIRD_PARTY_NOTICES.md",
        "curobo_metal-1.0.1/README.md",
        "curobo_metal-1.0.1/pyproject.toml",
        "curobo_metal-1.0.1/src/curobo/__init__.py",
    ]
    assert validate_members(sdist, valid) == []

    errors = validate_members(sdist, valid + ["other-root/file.txt"])
    assert any("exactly one root" in error for error in errors)
