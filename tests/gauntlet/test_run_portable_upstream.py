from __future__ import annotations

from tools.gauntlet.run_portable_upstream import PINNED_REVISION, selected_test_paths


def test_selects_every_substituted_bundled_test_as_a_safe_relative_path() -> None:
    census = {
        "entries": [
            {
                "module": "curobo.tests._src.types.test_pose",
                "surface": "bundled_test",
                "disposition": "platform_substituted",
            },
            {
                "module": "curobo.tests._src.types.test_device_cfg",
                "surface": "bundled_test",
                "disposition": "unchanged_upstream",
            },
        ]
    }
    assert PINNED_REVISION == "8e734f3ced1df898990bcd92de40abce475907db"
    assert selected_test_paths(census, "platform_substituted") == (
        "_src/types/test_pose.py",
    )
