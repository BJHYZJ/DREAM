"""Offline entrypoint checks: these tests do not claim task success."""
from pathlib import Path

import pytest

from simulation.run import CATALOG, build_command, load_catalog, preflight
from simulation.auditor import record_reviewer, sha


def test_catalog_without_runtime():
    report = preflight(Path("absent-assets"), Path("absent-models"), runtime=False)
    assert report["profiles_verified"] == 10
    assert report["source_snapshots_verified"] == 7
    assert not report["policy_executed"]


@pytest.mark.parametrize("index", range(1, 11))
def test_frozen_command(index, tmp_path):
    identifier = f"{index:02d}"
    _, data = load_catalog(CATALOG)
    case = next(row for row in data["cases"] if row["id"] == identifier)
    command, report = build_command(CATALOG, case_id=identifier, output=tmp_path / identifier)
    assert report["source_id"] == case["source_id"]
    assert report["case_ids"] == [identifier]
    assert ("--heading-navigation" in command) == (index >= 9)
    assert "run_instruction_frozen_batch.py" in command[1]
    assert "replay_instruction_actions.py" not in " ".join(command)


def test_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        build_command(CATALOG, case_id="01", output=tmp_path)


def test_refuses_unknown_case(tmp_path):
    with pytest.raises(ValueError):
        build_command(CATALOG, case_id="11", output=tmp_path / "new")


@pytest.mark.parametrize("index", range(1, 11))
def test_record_reviewer_matches_original_video(index):
    import json
    root, data = load_catalog(CATALOG)
    case = next(row for row in data["cases"] if row["id"] == f"{index:02d}")
    recorded = json.loads((root / case["records"]["record_review.json"]).read_text())
    assert sha(record_reviewer(case, root)) == recorded["review_script_sha256"]


@pytest.mark.parametrize("identifier", ["07", "10"])
def test_renderer_matches_original_video(identifier):
    import json
    from simulation.render import renderer_source
    root, data = load_catalog(CATALOG)
    case = next(row for row in data["cases"] if row["id"] == identifier)
    recorded = json.loads((root / case["records"]["spectator_video_review.json"]).read_text())
    assert sha(renderer_source(identifier) / "experiments" / "instruction_replay_video.py") == recorded["renderer_script_sha256"]
