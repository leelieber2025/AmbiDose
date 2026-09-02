from ambidose import __version__


def test_version_semver_shape():
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])
