import json
import tarfile

import pytest

from dream_sim.prune_recordings import prune_recording
from dream_sim.resume_study import digest


def fixture(tmp_path):
    live = tmp_path / "live"
    durable = tmp_path / "durable"
    name = "task01"
    (live / name).mkdir(parents=True)
    (durable / name).mkdir(parents=True)
    for filename, data in [
        ("observation_00001.npz", b"camera"),
        ("reviewer_view.mp4", b"video"),
        ("result.json", b"{}"),
        ("evaluator_room_map.npz", b"map"),
    ]:
        (live / name / filename).write_bytes(data)
    archive = durable / "recording.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(live / name, arcname=name)
    (durable / name / "recording_archive.json").write_text(
        json.dumps(dict(archive="recording.tar", archive_sha256=digest(archive)))
    )
    return live, durable, name


def test_prunes_verified_camera_copies_and_retains_summary_and_map(tmp_path):
    live, durable, name = fixture(tmp_path)
    plan = prune_recording(live, durable, name)
    assert plan["files"] == 2 and (live / name / "reviewer_view.mp4").exists()
    result = prune_recording(live, durable, name, execute=True)
    assert result["bytes"] == 11
    assert not (live / name / "reviewer_view.mp4").exists()
    assert (live / name / "result.json").exists()
    assert (live / name / "evaluator_room_map.npz").exists()
    assert (durable / "recording.tar").exists()


@pytest.mark.parametrize("change", ["live", "archive"])
def test_changed_bytes_abort_before_any_recording_is_removed(tmp_path, change):
    live, durable, name = fixture(tmp_path)
    if change == "live":
        (live / name / "observation_00001.npz").write_bytes(b"new evidence")
    else:
        with (durable / "recording.tar").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        prune_recording(live, durable, name, execute=True)
    assert (live / name / "observation_00001.npz").exists()
    assert (live / name / "reviewer_view.mp4").exists()
