import json

import pytest

from dream_sim.limited_worker import worker_environment
from dream_sim.resume_study import archive_recording, cancelled, completed_records, digest


def test_qualification_upper_bound_counts_unknown_and_pending_as_possible_successes():
    from dream_sim.resume_study import qualification_upper_bound

    results = {str(i): {"qualified": False} for i in range(15)}
    assert qualification_upper_bound(50, results) == 35
    results["pending"] = {}
    results["success"] = {"qualified": True}
    assert qualification_upper_bound(50, results) == 35
    results["sixteenth_failure"] = {"qualified": False}
    assert qualification_upper_bound(50, results) == 34


def test_staged_execution_preserves_cohort_and_never_retries_completed_outcomes():
    from dream_sim.resume_study import scheduled_attempts

    protocol = {"attempts": [{"name": str(i)} for i in range(50)]}
    original = json.dumps(protocol)
    priority = ["7", "18", "39", "40"]
    pilot = scheduled_attempts(protocol, {}, priority, 4)
    assert [job["name"] for job in pilot] == priority
    completed = {job["name"]: {"task_success": False} for job in pilot}
    remaining = scheduled_attempts(protocol, completed, priority, None)
    assert len(remaining) == 46
    assert {job["name"] for job in remaining}.isdisjoint(completed)
    assert json.dumps(protocol) == original
    for invalid in (["missing"], ["7", "7"]):
        with pytest.raises(ValueError):
            scheduled_attempts(protocol, {}, invalid, 4)
    with pytest.raises(ValueError):
        scheduled_attempts(protocol, {}, [], 0)


def test_worker_cannot_escape_gpu_allocation(monkeypatch):
    monkeypatch.setattr("dream_sim.limited_worker.ALLOWED_GPUS", ("4", "5", "6", "7"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("OMP_NUM_THREADS", "192")
    for gpu in ("4", "5", "6", "7"):
        env = worker_environment(gpu)
        assert env["CUDA_VISIBLE_DEVICES"] == gpu
        assert env["OMP_NUM_THREADS"] == "2"
    for gpu in ("0", "3", "4,5", "-1"):
        with pytest.raises(ValueError):
            worker_environment(gpu)


def test_explicit_two_gpu_deployment_overrides_stale_server_indices(tmp_path, monkeypatch):
    from dream_sim.limited_worker import configured_gpus

    config = tmp_path / "worker_limits.json"
    config.write_text(json.dumps({"allowed_gpu_ids": ["0", "1"]}))
    allocation = configured_gpus(config)
    monkeypatch.setattr("dream_sim.limited_worker.ALLOWED_GPUS", allocation)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    for gpu in allocation:
        assert worker_environment(gpu)["CUDA_VISIBLE_DEVICES"] == gpu
    with pytest.raises(ValueError):
        worker_environment("4")
    config.write_text(json.dumps({"allowed_gpu_ids": ["0", "0"]}))
    with pytest.raises(ValueError):
        configured_gpus(config)


def test_completed_failures_are_not_retried(tmp_path):
    finished = tmp_path / "failed_task"
    finished.mkdir()
    result = finished / "result.json"
    result.write_text(json.dumps({"evaluator_task_success": False}))
    (tmp_path / "interrupted").mkdir()
    protocol = {"attempts": [{"name": "failed_task"}, {"name": "interrupted"}]}
    records = [
        {"name": "failed_task", "task_success": False, "result_sha256": digest(result)},
        {"name": "interrupted", "status": "execution_failure"},
    ]
    assert list(completed_records(tmp_path, protocol, records)) == ["failed_task"]
    result.write_text("{}")
    with pytest.raises(ValueError, match="Unverified existing result"):
        completed_records(tmp_path, protocol, records)


def test_archive_retains_bytes_and_refuses_overwrite(tmp_path):
    run, durable = tmp_path / "run", tmp_path / "durable"
    source = run / "task"
    source.mkdir(parents=True)
    (durable / "raw").mkdir(parents=True)
    (source / "result.json").write_text('{"success": false}')
    (source / "sensor.bin").write_bytes(bytes(range(256)))
    receipt = archive_recording(run, durable, "task")
    assert receipt["all_members_verified"]
    assert receipt["member_sha256"]["sensor.bin"] == digest(source / "sensor.bin")
    assert receipt["archive_sha256"] == digest(durable / receipt["archive"])
    with pytest.raises(FileExistsError):
        archive_recording(run, durable, "task")


def test_cancelled_study_remains_stopped_from_either_record(tmp_path):
    run, durable = tmp_path / "run", tmp_path / "durable"
    run.mkdir()
    durable.mkdir()
    assert not cancelled(run, durable)
    for root in (run, durable):
        marker = root / "CANCELLED_BY_USER.json"
        marker.write_text("{}")
        assert cancelled(run, durable)
        marker.unlink()


def test_failed_task_audit_is_final_and_bound_to_frozen_inputs(tmp_path):
    from dream_sim.audit_recording import audit_recording

    run, durable = tmp_path / "run", tmp_path / "durable"
    (run / "task").mkdir(parents=True)
    durable.mkdir()
    (run / "protocol.json").write_text("{}")
    (run / "task/result.json").write_text('{"evaluator_task_success": false}')
    record = {"name": "task", "task_success": False}
    report = audit_recording(run, durable, record, "4")
    assert report["qualified"] is False
    assert report["reason"] == "task_evaluator_failed"
    assert audit_recording(run, durable, record, "4") == report
    (run / "protocol.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Audit inputs changed"):
        audit_recording(run, durable, record, "4")
