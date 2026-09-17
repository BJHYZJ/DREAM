"""The public entry point must report the current cohort and reject inflated outcomes."""

import hashlib
import json
import shutil

import pytest

from dream_sim.evaluate import EVALUATION, verify_results


def test_default_evaluation_is_the_current_complete_cohort():
    result = verify_results()
    assert result["attempts"] == 50
    assert result["strict_successes"] == 38
    assert result["strict_success_rate"] == 0.76


def test_rehashed_failed_outcome_cannot_be_counted_as_success(tmp_path):
    shutil.copytree(EVALUATION, tmp_path / "evidence")
    root = tmp_path / "evidence"
    path = root / "results.json"
    results = json.loads(path.read_text())
    next(row for row in results["results"] if not row["task_success"])["strict_pass"] = True
    path.write_text(json.dumps(results))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["results.json"].update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Failure incorrectly counted"):
        verify_results(root)
