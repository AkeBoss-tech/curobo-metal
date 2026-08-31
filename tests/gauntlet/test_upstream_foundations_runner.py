from tools.gauntlet.run_upstream_foundations import PINNED_REVISION, TESTS


def test_foundation_batch_is_bounded_and_pinned() -> None:
    assert PINNED_REVISION == "8e734f3ced1df898990bcd92de40abce475907db"
    assert len(TESTS) == 9
    assert len(set(TESTS)) == len(TESTS)
    assert all(path.endswith(".py") and ".." not in path for path in TESTS)
