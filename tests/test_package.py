from importlib.metadata import version

import curobo_metal


def test_package_version_matches_distribution_metadata() -> None:
    assert curobo_metal.__version__ == version("curobo-metal")
