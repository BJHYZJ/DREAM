"""Independently review every successful task in a completed parallel cohort."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dream_sim.io import configure_vulkan
from dream_sim.study_records import load_completed_screen, read, sha

INPUT_FILES = (
    "configuration.json",
    "result.json",
    "environment_task.json",
    "events.jsonl",
    "actions.json",
    "evaluator_trajectory.json",
    "evaluator_discoveries.json",
    "disturbance_forces.json",
    "initial_conditions.json",
    "evaluator_room_map.npz",
)


def review_one(run, name, output, physical_audit=None):
    """Run in a separate process so each review imports its frozen simulator."""
    run, output = run.resolve(), output.resolve()
    protocol, outcomes = load_completed_screen(run)
    matching = [record for job, record, _ in outcomes if job["name"] == name]
    if len(matching) != 1 or matching[0]["task_success"] is not True:
        raise ValueError("Choose a successful task from the declared cohort")
    if "experiments/instruction_arm_return.py" not in protocol["source_sha256"]:
        raise ValueError(
            "This entrypoint reviews controllers with measured intermediate arm returns"
        )
    source, record = run / "frozen_workspace/DREAM_code", run / name
    identities = {filename: sha(record / filename) for filename in INPUT_FILES}
    events = [json.loads(line) for line in (record / "events.jsonl").read_text().splitlines()]
    discoveries = read(record / "evaluator_discoveries.json")
    observation_ids = {row["observation_id"] for row in discoveries}
    observation_ids.update(
        event["evidence_observation_id"]
        for event in events
        if event["event"] == "observed_memory_update" and event.get("task_stage") == "pickup_search"
    )
    evidence_files = {f"observation_{number:05d}.npz" for number in observation_ids}
    evidence_files.update(
        f"evaluator_visibility/visibility_{row['observation_id']:05d}.npz" for row in discoveries
    )
    identities.update({filename: sha(record / filename) for filename in sorted(evidence_files)})
    protocol_digest = sha(run / "protocol.json")
    output.mkdir(parents=True, exist_ok=False)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        MS_ASSET_DIR=protocol["asset_root"],
        HF_HUB_CACHE=protocol["model_cache"],
    )
    configure_vulkan()
    sys.path[:0] = [str(source / "experiments"), str(source / "src")]
    if physical_audit is None:
        import replay_instruction_actions

        original_arguments = sys.argv
        try:
            sys.argv = [
                "replay_instruction_actions",
                "--source-run",
                str(record),
                "--output",
                str(output / "physical"),
            ]
            replay_instruction_actions.main()
        finally:
            sys.argv = original_arguments
        physical_audit = output / "physical/audit.json"
    physical_audit = physical_audit.resolve()
    physical_digest = sha(physical_audit)
    physical = read(physical_audit)
    from review_instruction_record import audit_record, primary_endpoint_checks

    from dream_sim import fold_review

    evidence, *_ = audit_record(record, physical_audit)
    checks = primary_endpoint_checks(evidence)
    fold = fold_review.review_fold(
        events,
        read(record / "actions.json"),
        physical["contact_audit"],
        physical.get("arm_joint_motion_reexecution"),
    )
    checks.update(
        all_fold_receipts_use_feedback_protocol=all(
            event.get("arm_return_protocol") == "feedback_gated"
            for event in events
            if event["event"] in ("carried_arm_folded", "idle_arm_folded")
        ),
        review_inputs_unchanged=all(
            sha(record / key) == value for key, value in identities.items()
        ),
        physical_audit_unchanged=sha(physical_audit) == physical_digest,
        protocol_unchanged=sha(run / "protocol.json") == protocol_digest,
        frozen_source_matches_protocol=all(
            sha(source / key) == value for key, value in protocol["source_sha256"].items()
        ),
    )
    result = dict(
        kind="feedback_return_record_review",
        name=name,
        source_run=str(record),
        protocol_sha256=protocol_digest,
        input_sha256=identities,
        physical_audit_path=str(physical_audit),
        physical_audit_sha256=physical_digest,
        checks=checks,
        fold=fold,
        record_evidence=evidence,
        task_physics_and_observation_checks_passed=all(checks.values()) and fold["passed"],
        fold_reviewer_sha256=sha(Path(fold_review.__file__)),
        reviewer_sha256=sha(Path(__file__)),
        new_policy_execution=False,
        existing_physical_replay_reused=physical_audit.parent != output / "physical",
    )
    (output / "screen_physics_review.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def review_completed(run, output, *, execute=False, physical_audits=None):
    """Review all raw successes once; retain every declared failure for reporting."""
    run, output = run.resolve(), output.resolve()
    protocol, outcomes = load_completed_screen(run)
    names = [job["name"] for job, row, _ in outcomes if row["task_success"]]
    plan = dict(
        run=str(run),
        planned_tasks=len(outcomes),
        successful_tasks_to_review=len(names),
        review_workers=1,
        new_policy_execution=False,
        execute=execute,
        reuse_physical_audits=str(physical_audits.resolve()) if physical_audits else None,
    )
    print(json.dumps(plan), flush=True)
    if not execute:
        return plan
    if "experiments/instruction_arm_return.py" not in protocol["source_sha256"]:
        raise ValueError(
            "This entrypoint requires a controller with measured intermediate arm returns"
        )
    output.mkdir(parents=True, exist_ok=False)
    for name in names:
        command = [
            sys.executable,
            "-m",
            "dream_sim.study_review",
            "--run",
            str(run),
            "--output",
            str(output / name),
            "--one",
            name,
        ]
        if physical_audits:
            command.extend(
                [
                    "--physical-audit",
                    str((physical_audits / name / "physical/audit.json").resolve()),
                ]
            )
        with (output / f"{name}.log").open("x") as log:
            subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
            )
        result = read(output / name / "screen_physics_review.json")
        print(
            json.dumps(
                dict(name=name, passed=result["task_physics_and_observation_checks_passed"])
            ),
            flush=True,
        )
    summary = dict(
        plan,
        reviews_complete=True,
        protocol_sha256=sha(run / "protocol.json"),
        review_sha256={name: sha(output / name / "screen_physics_review.json") for name in names},
    )
    (output / "reviews_complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Execute the printed review plan")
    parser.add_argument(
        "--physical-audits",
        type=Path,
        help="Reuse existing exact physical replays; recheck their saved observations",
    )
    parser.add_argument("--one", help=argparse.SUPPRESS)
    parser.add_argument("--physical-audit", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        result = review_one(args.run, args.one, args.output, args.physical_audit)
        print(
            json.dumps(
                dict(name=args.one, passed=result["task_physics_and_observation_checks_passed"])
            )
        )
    else:
        if args.physical_audit:
            parser.error("Use --physical-audits for a complete cohort")
        review_completed(
            args.run, args.output, execute=args.execute, physical_audits=args.physical_audits
        )


if __name__ == "__main__":
    main()
