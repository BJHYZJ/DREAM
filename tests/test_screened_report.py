"""Completed-cohort binding and review orchestration without a simulator."""

import json
from pathlib import Path

import pytest

from dream_sim import fold_review, study_review
from dream_sim.screened_report import ENDPOINT_CHECKS, INTEGRITY_CHECKS
from dream_sim.study_records import load_completed_screen, sha
from dream_sim.study_report import analyze


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def cohort(root, successes=1):
    source = root / "frozen_workspace/DREAM_code/experiments/instruction_arm_return.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic cohort fixture; never executed.\n")
    hashes = {"experiments/instruction_arm_return.py": sha(source)}
    jobs = []
    for i in range(50):
        name = f"case_{i:02d}"
        task = dict(scene=f"house_{i}", seed=42)
        task_path = root / f"frozen_tasks/{name}/task.json"
        write(task_path, task)
        (task_path.parent / "evaluator_room_map.npz").write_bytes(b"synthetic map")
        job = dict(
            name=name,
            scene=task["scene"],
            seed=42,
            variant="dynamic",
            task=str(task_path.relative_to(root)),
            task_sha256=sha(task_path),
            room_map_sha256=sha(task_path.parent / "evaluator_room_map.npz"),
        )
        jobs.append(job)
        folder = root / name
        write(folder / "environment_task.json", task)
        (folder / "evaluator_room_map.npz").write_bytes(b"synthetic map")
        write(
            folder / "result.json",
            dict(
                evaluator_task_success=i < successes,
                source_files_unchanged=True,
                robot_action_seconds=120.0,
                robot_action_limit_seconds=900.0,
            ),
        )
        for filename in ["source_hashes_before.json", "source_hashes_after.json"]:
            write(folder / filename, hashes)
        write(
            root / f"screen_results/{name}.json",
            dict(
                **job,
                task_success=i < successes,
                result_sha256=sha(folder / "result.json"),
                source_verified_after=True,
                returncode=0,
                wall_timeout=False,
            ),
        )
    protocol = dict(
        mode="residential50_easy_grasp_seed42_v1",
        planned_attempts=50,
        simulation_budget_s=900.0,
        source_sha256=hashes,
        attempts=jobs,
    )
    write(root / "protocol.json", protocol)
    (root / "protocol.sha256").write_text(sha(root / "protocol.json") + "\n")
    write(root / "screen_protocol.json", dict(protocol_sha256=sha(root / "protocol.json")))
    write(
        root / "screen_complete.json",
        dict(
            all_50_recorded=True,
            planned=50,
            completed=50,
            task_successes=successes,
            active={},
            fatal_errors=[],
        ),
    )
    return root, protocol


def audit_fixture(run, protocol, audits, monkeypatch, contact_failure=False):
    name = protocol["attempts"][0]["name"]
    folder, destination = run / name, audits / name
    for filename in study_review.INPUT_FILES:
        path = folder / filename
        if not path.exists():
            write(path, [] if filename.endswith(".json") else {"event": "synthetic"})
    physical_path = destination / "physical/audit.json"
    physical = dict(
        source_run=str(folder.resolve()),
        physical_reexecution_passed=True,
        source_recording_unchanged=True,
        replay_environment_source_matches_recording=True,
        replay_source_unchanged=True,
        contact_audit=dict(native_environment_contact_control_steps=int(contact_failure)),
    )
    write(physical_path, physical)
    write(physical_path.parent / "replay_source_hashes.json", protocol["source_sha256"])
    fold = dict(passed=True, checks={"synthetic_fixture": True})
    monkeypatch.setattr(fold_review, "review_fold", lambda *args: fold)
    checks = dict.fromkeys(INTEGRITY_CHECKS | ENDPOINT_CHECKS, True)
    checks["no_native_contact_above_threshold"] = not contact_failure
    review = dict(
        kind="feedback_return_record_review",
        name=name,
        source_run=str(folder.resolve()),
        protocol_sha256=sha(run / "protocol.json"),
        input_sha256={filename: sha(folder / filename) for filename in study_review.INPUT_FILES},
        physical_audit_path=str(physical_path.resolve()),
        physical_audit_sha256=sha(physical_path),
        checks=checks,
        fold=fold,
        task_physics_and_observation_checks_passed=not contact_failure,
        fold_reviewer_sha256=sha(Path(fold_review.__file__)),
        reviewer_sha256=sha(Path(study_review.__file__)),
    )
    review_path = destination / "screen_physics_review.json"
    write(review_path, review)
    write(
        audits / "reviews_complete.json",
        dict(
            reviews_complete=True,
            protocol_sha256=sha(run / "protocol.json"),
            review_sha256={name: sha(review_path)},
        ),
    )
    return review_path


def test_parallel_outcomes_are_supported_without_a_serial_batch_result(tmp_path, monkeypatch):
    run, protocol = cohort(tmp_path / "run")
    audits = tmp_path / "audits"
    audit_fixture(run, protocol, audits, monkeypatch)
    report = analyze(run, audits, tmp_path / "summary")
    assert report["statistics"]["variants"]["dynamic"] == dict(
        attempts=50, successes=1, success_rate=0.02
    )
    assert len(report["attempts"]) == 50
    assert all(row["eligible_for_summary"] for row in report["attempts"])
    assert (tmp_path / "summary/attempts.csv").exists()


@pytest.mark.parametrize(
    "damage",
    ["missing", "extra", "protocol", "task", "environment", "source", "outcome", "incomplete"],
)
def test_missing_changed_or_inconsistent_outcomes_stop_reporting(tmp_path, damage):
    run, _ = cohort(tmp_path / "run")
    if damage == "missing":
        (run / "screen_results/case_49.json").unlink()
    elif damage == "extra":
        write(run / "screen_results/unexpected.json", {})
    elif damage == "protocol":
        (run / "protocol.json").write_text((run / "protocol.json").read_text() + " ")
    elif damage == "task":
        write(run / "frozen_tasks/case_00/task.json", {"changed": True})
    elif damage == "environment":
        write(run / "case_00/environment_task.json", {"scene": "other_house", "seed": 42})
    elif damage == "source":
        write(run / "case_00/source_hashes_after.json", {})
    elif damage == "outcome":
        path = run / "screen_results/case_00.json"
        row = json.loads(path.read_text())
        row["task_success"] = False
        write(path, row)
    else:
        path = run / "screen_complete.json"
        value = json.loads(path.read_text())
        value["all_50_recorded"] = False
        write(path, value)
    with pytest.raises(ValueError):
        analyze(run, tmp_path / "audits", tmp_path / "summary")
    assert not (tmp_path / "summary").exists()


@pytest.mark.parametrize(
    "damage", ["missing_review", "input", "physics", "checks", "reviewer", "completion"]
)
def test_a_review_cannot_certify_different_or_incomplete_evidence(tmp_path, monkeypatch, damage):
    run, protocol = cohort(tmp_path / "run")
    audits = tmp_path / "audits"
    path = audit_fixture(run, protocol, audits, monkeypatch)
    if damage == "missing_review":
        path.unlink()
    elif damage == "input":
        write(run / "case_00/actions.json", [{"changed": True}])
    elif damage == "physics":
        write(audits / "case_00/physical/audit.json", {"changed": True})
    elif damage == "completion":
        write(audits / "reviews_complete.json", {"reviews_complete": False})
    else:
        value = json.loads(path.read_text())
        if damage == "checks":
            del value["checks"]["review_inputs_unchanged"]
        else:
            value["fold_reviewer_sha256"] = "changed"
        write(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        analyze(run, audits, tmp_path / "summary")
    assert not (tmp_path / "summary").exists()


def test_contact_rejection_keeps_the_original_score_and_full_denominator(tmp_path, monkeypatch):
    run, protocol = cohort(tmp_path / "run")
    audits = tmp_path / "audits"
    audit_fixture(run, protocol, audits, monkeypatch, contact_failure=True)
    before = sha(run / "case_00/result.json")
    with pytest.raises(ValueError, match="include-contact-rejections"):
        analyze(run, audits, tmp_path / "summary")
    report = analyze(run, audits, tmp_path / "summary", include_contact_rejections=True)
    assert report["reported_task_statistics"]["variants"]["dynamic"]["successes"] == 1
    assert report["statistics"]["variants"]["dynamic"] == dict(
        attempts=50, successes=0, success_rate=0
    )
    assert sha(run / "case_00/result.json") == before


def test_review_plan_does_not_launch_simulations_or_create_output(tmp_path, monkeypatch):
    run, _ = cohort(tmp_path / "run", successes=3)
    monkeypatch.setattr(
        study_review.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected execution")
    )
    result = study_review.review_completed(run, tmp_path / "audits")
    assert result["planned_tasks"] == 50 and result["successful_tasks_to_review"] == 3
    assert result["new_policy_execution"] is False
    assert not (tmp_path / "audits").exists()


def test_loader_never_accepts_an_unfinished_parallel_run(tmp_path):
    with pytest.raises(RuntimeError, match="has not finished"):
        load_completed_screen(tmp_path)


def test_review_execution_launches_every_success_in_a_separate_process(tmp_path, monkeypatch):
    run, _ = cohort(tmp_path / "run", successes=3)
    commands = []

    def execute(command, **kwargs):
        assert kwargs["check"] is True and kwargs["timeout"] == 1200
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        name = command[command.index("--one") + 1]
        output = Path(command[command.index("--output") + 1])
        commands.append(command)
        write(
            output / "screen_physics_review.json",
            dict(
                name=name,
                task_physics_and_observation_checks_passed=True,
            ),
        )

    monkeypatch.setattr(study_review.subprocess, "run", execute)
    result = study_review.review_completed(run, tmp_path / "audits", execute=True)
    assert len(commands) == 3
    assert set(result["review_sha256"]) == {"case_00", "case_01", "case_02"}
    assert result["planned_tasks"] == 50 and result["new_policy_execution"] is False


def test_failed_review_process_leaves_the_cohort_unqualified(tmp_path, monkeypatch):
    run, _ = cohort(tmp_path / "run")

    def fail(command, **kwargs):
        raise study_review.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(study_review.subprocess, "run", fail)
    with pytest.raises(study_review.subprocess.CalledProcessError):
        study_review.review_completed(run, tmp_path / "audits", execute=True)
    assert not (tmp_path / "audits/reviews_complete.json").exists()
