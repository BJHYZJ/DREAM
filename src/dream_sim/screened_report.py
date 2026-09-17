"""Report a complete parallel cohort using bound independent review records."""

import csv
import json
from pathlib import Path

from dream_sim import fold_review, study_review
from dream_sim.sources import safe_member
from dream_sim.study_records import load_completed_screen, read, sha
from dream_sim.study_review import INPUT_FILES

INTEGRITY_CHECKS = {
    "source_hashes_match",
    "saved_source_matches_hashes",
    "actual_control_reexecution",
    "replay_sources_verified",
    "replay_refers_to_this_run",
    "complete_aligned_traces",
    "review_inputs_unchanged",
    "frozen_source_matches_protocol",
    "physical_audit_unchanged",
    "protocol_unchanged",
}
ENDPOINT_CHECKS = {
    "instruction_first",
    "no_native_contact_above_threshold",
    "identity_records_consistent_with_saved_masks_and_replayed_positions",
    "fresh_visual_discovery_at_standoff_during_translation",
    "original_task_scoring",
    "reexecuted_task_success",
    "correct_visual_moved_target_observation",
    "all_logged_updates_have_observed_free_depth",
    "all_fold_receipts_use_feedback_protocol",
}


def checked_review(run, name, audits, protocol):
    folder = run / name
    path = audits / name / "screen_physics_review.json"
    review = read(path)
    if (
        review.get("kind") != "feedback_return_record_review"
        or review.get("name") != name
        or Path(review.get("source_run", "")).resolve() != folder.resolve()
        or review.get("protocol_sha256") != sha(run / "protocol.json")
        or review.get("reviewer_sha256") != sha(Path(study_review.__file__))
        or review.get("fold_reviewer_sha256") != sha(Path(fold_review.__file__))
    ):
        raise ValueError(f"{name}: review refers to different inputs or a different arm reviewer")
    identities = review.get("input_sha256", {})
    if not set(INPUT_FILES) <= identities.keys() or any(
        sha(folder / safe_member(filename)) != digest for filename, digest in identities.items()
    ):
        raise ValueError(f"{name}: review inputs changed or are incomplete")
    physical_path = Path(review["physical_audit_path"])
    if sha(physical_path) != review.get("physical_audit_sha256"):
        raise ValueError(f"{name}: physical audit changed")
    physical = read(physical_path)
    if Path(physical["source_run"]).resolve() != folder.resolve():
        raise ValueError(f"{name}: physical replay refers to another task")
    if (
        any(
            physical.get(flag) is not True
            for flag in (
                "physical_reexecution_passed",
                "source_recording_unchanged",
                "replay_environment_source_matches_recording",
                "replay_source_unchanged",
            )
        )
        or read(physical_path.parent / "replay_source_hashes.json") != protocol["source_sha256"]
    ):
        raise ValueError(f"{name}: physical replay is incomplete or its sources differ")
    checks = review.get("checks", {})
    if (
        not (INTEGRITY_CHECKS | ENDPOINT_CHECKS) <= checks.keys()
        or any(type(value) is not bool for value in checks.values())
        or not all(checks[key] for key in INTEGRITY_CHECKS)
    ):
        raise ValueError(f"{name}: review integrity checks are missing or failed")
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    fold = fold_review.review_fold(
        events,
        read(folder / "actions.json"),
        physical["contact_audit"],
        physical.get("arm_joint_motion_reexecution"),
    )
    passed = all(checks.values()) and fold["passed"]
    if (
        review.get("fold") != fold
        or review.get("task_physics_and_observation_checks_passed") is not passed
    ):
        raise ValueError(f"{name}: stored verdict disagrees with its checks")
    return review, physical, sha(path)


def analyze_screened(run, audits, output, *, include_contact_rejections=False):
    from dream_sim.study_report import summarize_residential_outcomes

    run, audits, output = run.resolve(), audits.resolve(), output.resolve()
    protocol, outcomes = load_completed_screen(run)
    completed = read(audits / "reviews_complete.json")
    successes = {job["name"] for job, row, _ in outcomes if row["task_success"]}
    review_hashes = completed.get("review_sha256", {})
    if (
        completed.get("reviews_complete") is not True
        or completed.get("protocol_sha256") != sha(run / "protocol.json")
        or set(review_hashes) != successes
    ):
        raise ValueError("Independent reviews are incomplete")
    rows, rejections = [], []
    for job, record, result in outcomes:
        name = job["name"]
        row = dict(
            name=name,
            scene=job["scene"],
            seed=job["seed"],
            variant=job["variant"],
            task_success=record["task_success"],
            qualified_task_success=record["task_success"],
            strict_success=result.get("evaluator_protocol_success"),
            eligible_for_summary=True,
            runner_status="completed",
            result_sha256=record["result_sha256"],
            wall_time_s=record.get("execution_seconds"),
            robot_action_seconds=record.get("robot_action_seconds"),
            failure_category=record.get("failure_category"),
            error=record.get("error"),
            audit_qualification="task_failed",
            native_contact_steps=None,
        )
        if record["task_success"]:
            review, physical, digest = checked_review(run, name, audits, protocol)
            if digest != review_hashes[name]:
                raise ValueError(f"{name}: completed review checksum mismatch")
            contact_failed = not review["checks"]["no_native_contact_above_threshold"]
            if contact_failed and not include_contact_rejections:
                raise ValueError(
                    "Native contact rejected; use --include-contact-rejections to retain it as a failed outcome"
                )
            qualified = review["task_physics_and_observation_checks_passed"]
            reason = (
                "passed"
                if qualified
                else "native_contact_rejected"
                if contact_failed
                else "arm_return_rejected"
                if not review["fold"]["passed"]
                else "record_review_rejected"
            )
            row.update(
                qualified_task_success=qualified,
                audit_qualification=reason,
                physical_audit_sha256=review["physical_audit_sha256"],
                record_review_sha256=digest,
                arm_return_review=review["fold"],
                native_contact_steps=physical["contact_audit"][
                    "native_environment_contact_control_steps"
                ],
            )
            if not qualified:
                rejections.append(
                    dict(name=name, reason=reason, native_contact_steps=row["native_contact_steps"])
                )
        rows.append(row)
    raw_statistics = summarize_residential_outcomes(rows)
    statistics = summarize_residential_outcomes(
        [dict(row, task_success=row["qualified_task_success"]) for row in rows]
    )
    report = dict(
        protocol_sha256=sha(run / "protocol.json"),
        all_planned_outcomes_bound=True,
        planned_attempts=50,
        all_reported_successes_audited=True,
        all_counted_successes_passed_audits=True,
        successful_task_physics_and_record_checks_passed=not rejections,
        statistics_endpoint="audit-qualified task completion",
        audit_rejections=rejections,
        statistics=statistics,
        reported_task_statistics=raw_statistics,
        comparison_to_baseline={},
        attempts=rows,
        scope="All 50 declared development-cohort tasks; one seed-42 dynamic-memory task per house.",
        fold_reviewer_sha256=sha(Path(fold_review.__file__)),
    )
    report["statistics"]["analysis_plan"]["primary_endpoint"] = (
        "Independently verified task completion and arm return"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "study_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "name",
        "scene",
        "seed",
        "variant",
        "task_success",
        "qualified_task_success",
        "audit_qualification",
        "robot_action_seconds",
        "failure_category",
        "error",
    ]
    with (output / "attempts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return report
