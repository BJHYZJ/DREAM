import json
from pathlib import Path

import pytest

from dream_sim.study import controller_overrides, materialize_controller, sha


def revision(root: Path, name="experiments/policy.py", content=b"candidate\n"):
    file = root / name
    file.parent.mkdir(parents=True)
    file.write_bytes(content)
    (root / "controller.json").write_text(json.dumps({
        "base_study_source_id": "base", "overrides": {name: sha(content)}}))


def test_revision_applies_only_to_a_copy_and_preserves_other_modules(tmp_path):
    source = tmp_path / "baseline"
    (source / "experiments").mkdir(parents=True)
    (source / "src/dream").mkdir(parents=True)
    (source / "experiments/policy.py").write_bytes(b"baseline\n")
    (source / "experiments/evaluation.py").write_bytes(b"evaluator\n")
    (source / "src/dream/memory.py").write_bytes(b"memory\n")
    directory = tmp_path / "revision"
    revision(directory)
    destination = tmp_path / "materialized"
    materialize_controller(source, destination, controller_overrides(directory, "base"))
    assert (source / "experiments/policy.py").read_bytes() == b"baseline\n"
    assert (destination / "experiments/policy.py").read_bytes() == b"candidate\n"
    assert (destination / "experiments/evaluation.py").read_bytes() == b"evaluator\n"
    assert (destination / "src/dream/memory.py").read_bytes() == b"memory\n"


def test_revision_rejects_a_different_base_version(tmp_path):
    revision(tmp_path / "revision")
    with pytest.raises(ValueError, match="different base"):
        controller_overrides(tmp_path / "revision", "another-base")


def test_revision_rejects_changed_source_bytes(tmp_path):
    directory = tmp_path / "revision"
    revision(directory)
    (directory / "experiments/policy.py").write_bytes(b"unrecorded change")
    with pytest.raises(ValueError, match="checksum mismatch"):
        controller_overrides(directory, "base")


def test_materialization_does_not_overwrite_an_existing_workspace(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        materialize_controller(tmp_path / "source", existing, {})
    assert (existing / "keep.txt").read_text() == "keep"
