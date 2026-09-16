import json
from pathlib import Path

import pytest

from dream_sim.study import load_task_manifest, sha


def fixture(tmp_path):
    room = tmp_path / "rooms.npz"
    room.write_bytes(b"geometry")
    task = tmp_path / "task.json"
    task.write_text(json.dumps(dict(scene="house-a", seed=42, room_map_file=room.name)))
    row = dict(
        id="01",
        scene="house-a",
        seed=42,
        variant="dynamic",
        file=task.name,
        sha256=sha(task.read_bytes()),
        room_map_sha256=sha(room.read_bytes()),
    )
    manifest = dict(planned_attempts=1, seed=42, variants=["dynamic"], tasks=[row])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, task, room


def test_cohort_binds_task_and_geometry(tmp_path):
    path, expected, task, _ = fixture(tmp_path)
    actual, tasks = load_task_manifest(path)
    assert actual == expected and tasks == [task]


def test_cohort_rejects_changed_geometry(tmp_path):
    path, _, _, room = fixture(tmp_path)
    room.write_bytes(b"other house")
    with pytest.raises(ValueError, match="Room map checksum"):
        load_task_manifest(path)


def test_cohort_rejects_duplicate_houses(tmp_path):
    path, manifest, _, _ = fixture(tmp_path)
    manifest["tasks"] *= 2
    manifest["planned_attempts"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="distinct house"):
        load_task_manifest(path)


def test_cohort_rejects_seed_metadata_changes(tmp_path):
    path, manifest, _, _ = fixture(tmp_path)
    manifest["tasks"][0]["seed"] = 43
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata differs"):
        load_task_manifest(path)


def test_cohort_rejects_task_escape(tmp_path):
    path, manifest, _, _ = fixture(tmp_path)
    manifest["tasks"][0]["file"] = "../outside.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_task_manifest(path)


def test_residential_summary_keeps_all_failures_in_denominator():
    from dream_sim.study_report import summarize_residential_outcomes

    rows = [
        dict(scene=f"house-{i}", seed=42, variant="dynamic", task_success=i < 37) for i in range(50)
    ]
    result = summarize_residential_outcomes(rows)
    assert result["variants"]["dynamic"] == dict(attempts=50, successes=37, success_rate=0.74)
    assert result["contrast"] is None
    with pytest.raises(ValueError, match="all 50"):
        summarize_residential_outcomes(rows[:-1])


def test_public_residential_tasks_use_fifty_houses_and_seed_42():
    from collections import Counter

    root = Path(__file__).resolve().parents[1] / "configs/residential50"
    manifest, tasks = load_task_manifest(root / "task_manifest.json")
    assert manifest["planned_attempts"] == 50 and len(tasks) == 50
    assert {r["seed"] for r in manifest["tasks"]} == {42}
    assert {r["variant"] for r in manifest["tasks"]} == {"dynamic"}
    assert (
        sorted(Counter(json.loads(p.read_text())["recipe"]["id"] for p in tasks).values())
        == [10] * 5
    )
    lock = json.loads((root / "assets.lock.json").read_text())
    assert lock["complete"] and not lock["failures"]
    assert {r["scene"] for r in lock["scenes"]} == {r["scene"] for r in manifest["tasks"]}
