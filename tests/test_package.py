import curobo_metal
import curobo
from curobo_metal._version import __version__ as source_version


def test_package_version_matches_distribution_metadata() -> None:
    assert curobo_metal.__version__ == source_version
    assert curobo.__version__ == source_version
