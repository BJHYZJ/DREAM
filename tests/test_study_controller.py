import json
from pathlib import Path

import pytest

from dream_sim.study import controller_overrides, materialize_controller, sha


def revision(root: Path, name="experiments/policy.py", content=b"candidate\n"):
    file = root / name
    file.parent.mkdir(parents=True)
    file.write_bytes(content)
    (root / "controller.json").write_text(
        json.dumps({"base_study_source_id": "base", "overrides": {name: sha(content)}})
    )


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


def test_revision_can_include_a_new_experiment_module(tmp_path):
    source = tmp_path / "baseline"
    (source / "experiments").mkdir(parents=True)
    (source / "experiments/policy.py").write_bytes(b"import visual_search\n")
    directory = tmp_path / "revision"
    revision(directory, "experiments/visual_search.py", b"TILE_SIZE = 384\n")
    destination = tmp_path / "materialized"
    materialize_controller(source, destination, controller_overrides(directory, "base"))
    assert (destination / "experiments/visual_search.py").read_bytes() == b"TILE_SIZE = 384\n"
    assert not (source / "experiments/visual_search.py").exists()


def test_materialization_rejects_modules_outside_experiments(tmp_path):
    with pytest.raises(ValueError, match="experiment Python modules"):
        materialize_controller(
            tmp_path / "source", tmp_path / "copy", {"src/dream/new.py": b"unexpected\n"}
        )


def test_inherited_controller_keeps_parent_integrity_and_replaces_only_declared_files(tmp_path):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    revision(parent, content=b"parent\n")
    revision(child, name="experiments/return.py", content=b"return\n")
    manifest = json.loads((child / "controller.json").read_text())
    manifest["base_controller"] = "parent"
    (child / "controller.json").write_text(json.dumps(manifest))
    assert controller_overrides(child, "base") == {
        "experiments/policy.py": b"parent\n",
        "experiments/return.py": b"return\n",
    }
    (parent / "experiments/policy.py").write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        controller_overrides(child, "base")


def test_controller_inheritance_rejects_cycles_and_paths_outside_siblings(tmp_path):
    root = tmp_path / "controller"
    revision(root)
    manifest = json.loads((root / "controller.json").read_text())
    for name in ("controller", "../outside"):
        manifest["base_controller"] = name
        (root / "controller.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            controller_overrides(root, "base")
