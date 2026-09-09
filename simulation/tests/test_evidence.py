"""Integrity-verifier behavior, not experiment outcome tests."""
import hashlib
import zipfile

import pytest

from simulation.verify_evidence import verify_files


def metadata(payload):
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_accepts_exact_file_and_rejects_tampering(tmp_path):
    path = tmp_path / "record.json"
    path.write_bytes(b"{}")
    expected = {path.name: metadata(path.read_bytes())}
    verify_files(tmp_path, expected)
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="changed"):
        verify_files(tmp_path, expected)


def test_checks_zip_members(tmp_path):
    path = tmp_path / "case.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("result.json", b"{}")
    row = dict(metadata(path.read_bytes()), kind="zip", members={"result.json": metadata(b"{}")})
    verify_files(tmp_path, {path.name: row})
    row["members"]["result.json"] = metadata(b"[]")
    with pytest.raises(ValueError, match="Archived record changed"):
        verify_files(tmp_path, {path.name: row})


def test_rejects_symlink(tmp_path):
    target = tmp_path / "real.json"
    target.write_bytes(b"{}")
    (tmp_path / "link.json").symlink_to(target)
    with pytest.raises(ValueError, match="Unsafe"):
        verify_files(tmp_path, {"link.json": metadata(b"{}")})
