"""Installed-package version detection."""


def get_version() -> str:
    from curobo_metal._version import __version__

    return __version__
