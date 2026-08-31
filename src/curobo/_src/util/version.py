"""Installed-package version detection."""


def get_version() -> str:
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    if (root / ".git").exists() and not (root / ".git/shallow").exists():
        try:
            import setuptools_scm

            return setuptools_scm.get_version(
                root=root,
                version_scheme="no-guess-dev",
                local_scheme="dirty-tag",
            )
        except (ImportError, LookupError):
            pass
    try:
        from importlib.metadata import version

        return version("curobo-metal")
    except Exception:
        return "v0.8.0-no-tag"
