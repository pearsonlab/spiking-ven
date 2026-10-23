"""Checksum manifest: recording artifacts, and catching a changed or missing one."""

from spiking_ven.manifest import sha256_file, verify_manifest, write_manifest

ARTIFACTS = ("outputs/a.npz", "outputs/b.npz")


def _root(tmp_path):
    (tmp_path / "outputs").mkdir()
    for i, rel in enumerate(ARTIFACTS):
        (tmp_path / rel).write_bytes(bytes([i]) * 100)
    return tmp_path


def test_write_then_verify_passes(tmp_path):
    root = _root(tmp_path)
    m = tmp_path / "MANIFEST.sha256"
    assert write_manifest(m, root, ARTIFACTS) == 0
    lines = m.read_text().splitlines()
    assert lines == [f"{sha256_file(root / rel)}  {rel}" for rel in ARTIFACTS]
    assert verify_manifest(m, root) == 0


def test_a_changed_artifact_fails_verification(tmp_path, capsys):
    root = _root(tmp_path)
    m = tmp_path / "MANIFEST.sha256"
    write_manifest(m, root, ARTIFACTS)
    (root / ARTIFACTS[1]).write_bytes(b"different")
    assert verify_manifest(m, root) == 1
    assert "CHANGED" in capsys.readouterr().out


def test_a_missing_artifact_fails_verification(tmp_path, capsys):
    root = _root(tmp_path)
    m = tmp_path / "MANIFEST.sha256"
    write_manifest(m, root, ARTIFACTS)
    (root / ARTIFACTS[0]).unlink()
    assert verify_manifest(m, root) == 1
    assert "MISSING" in capsys.readouterr().out


def test_absent_artifacts_are_not_recorded(tmp_path):
    root = _root(tmp_path)
    m = tmp_path / "MANIFEST.sha256"
    assert write_manifest(m, root, ARTIFACTS + ("outputs/absent.npz",)) == 0
    assert "absent" not in m.read_text()
    assert write_manifest(m, root, ("outputs/absent.npz",)) == 1   # nothing to record


def test_verify_without_a_manifest_fails(tmp_path):
    assert verify_manifest(tmp_path / "nope.sha256", tmp_path) == 1


def test_comments_and_blank_lines_are_ignored(tmp_path):
    root = _root(tmp_path)
    m = tmp_path / "MANIFEST.sha256"
    write_manifest(m, root, ARTIFACTS)
    m.write_text("# recorded by hand\n\n" + m.read_text())
    assert verify_manifest(m, root) == 0
