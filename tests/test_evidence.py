"""Checksums and path validation for experiment records."""
import hashlib
import json
import zipfile

import pytest

from dream_sim.verify_evidence import verify_files, verify_comparison


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


def test_controller_comparison_requires_all_declared_attempt_archives(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"attempt_archives": 60, "files": {}}))
    with pytest.raises(ValueError, match="Incomplete controller comparison"):
        verify_comparison(tmp_path)


def test_controller_comparison_rejects_unqualified_reported_successes(tmp_path):
    manifest = dict(attempt_archives=60, all_60_outcomes_bound=True,
                    all_reported_successes_audited=True, all_counted_successes_passed_audits=False,
                    files={f"attempts/{index}.zip": {} for index in range(60)})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="all_counted_successes_passed_audits"):
        verify_comparison(tmp_path)
