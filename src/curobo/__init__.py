"""Portable drop-in namespace for the supported cuRobo type slice."""

# Pinned cuRobo exposes its version at the package root and does not re-export
# the classes in ``curobo.types`` here.
from curobo._src.util.version import get_version

__version__ = get_version()
del get_version
