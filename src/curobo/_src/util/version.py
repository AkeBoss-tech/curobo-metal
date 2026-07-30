"""Installed-package version detection."""


def get_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    for distribution in ("curobo-metal", "nvidia_curobo"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            pass
    return "0.0.0+source"
