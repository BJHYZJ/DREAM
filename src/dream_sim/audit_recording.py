"""Serial frozen-physics and record verification before freeing camera files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from dream_sim.fold_review import review_fold
from dream_sim.io import atomic_json, digest
from dream_sim.limited_worker import worker_environment


def audit_recording(run: Path, durable: Path, record: dict, gpu: str) -> dict:
    name = record["name"]
    output = durable / "audits" / name
    output.mkdir(parents=True, exist_ok=True)
    source = run / "frozen_workspace/DREAM_code"
    identity = dict(
        result_sha256=digest(run / name / "result.json"),
        protocol_sha256=digest(run / "protocol.json"),
    )
    provenance = output / "input_identity.json"
    if provenance.exists() and json.loads(provenance.read_text()) != identity:
        raise ValueError("Audit inputs changed since the previous execution")
    if not provenance.exists():
        atomic_json(provenance, identity)
    if (output / "qualification.json").exists():
        return json.loads((output / "qualification.json").read_text())
    report = dict(
        name=name,
        qualified=False,
        task_success=bool(record.get("task_success")),
        audit_adapter_sha256=digest(Path(__file__)),
        release_ready=False,
    )
    if not report["task_success"]:
        report["reason"] = "task_evaluator_failed"
        atomic_json(output / "qualification.json", report)
        return report
    env = worker_environment(gpu)
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    stages = [
        (
            "physical",
            "replay_instruction_actions.py",
            ["--source-run", str(run / name), "--output", str(output / "physical")],
            1200,
        ),
        (
            "record",
            "review_instruction_record.py",
            [
                "--run",
                str(run / name),
                "--physical-audit",
                str(output / "physical/audit.json"),
                "--output",
                str(output / "record"),
                "--annotate",
                "--endpoint",
                "task",
            ],
            1800,
        ),
    ]
    for stage, script, arguments, timeout in stages:
        report_path = (
            output / stage / ("audit.json" if stage == "physical" else "record_review.json")
        )
        if report_path.exists():
            continue
        if (output / stage).exists():
            raise RuntimeError(
                f"Incomplete {stage} audit requires inspection before resuming: {name}"
            )
        command = [
            sys.executable,
            "-m",
            "dream_sim.limited_worker",
            "--gpu",
            gpu,
            "--wait-for-slot-seconds",
            "7200",
            "--entrypoint",
            str(source / "experiments" / script),
            "--receipt",
            str(output / (stage + ".runtime.json")),
            *arguments,
        ]
        with (output / (stage + ".log")).open("x") as log:
            result = subprocess.run(
                command,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout + 7200,
            )
        report[stage + "_returncode"] = result.returncode
        if not report_path.is_file():
            raise RuntimeError(f"{stage} audit did not produce a report: {name}")
    physical = json.loads((output / "physical/audit.json").read_text())
    review = json.loads((output / "record/record_review.json").read_text())
    events = [json.loads(line) for line in (run / name / "events.jsonl").read_text().splitlines()]
    fold = review_fold(
        events, json.loads((run / name / "actions.json").read_text()), physical["contact_audit"]
    )
    status = dict(
        name=name,
        physical_reexecution_passed=bool(physical["physical_reexecution_passed"]),
        native_contact_steps=physical["contact_audit"]["native_environment_contact_control_steps"],
        automatic_checks_passed=bool(review["primary_task_record_review_passed"]),
        primary_task_checks_passed=bool(review["primary_task_record_review_passed"]),
        strict_checks_passed=bool(review["record_review_passed"]),
        visual_acceptance_pending=True,
        release_ready=False,
    )
    report.update(
        audit=status,
        fold=fold,
        qualified=bool(
            status["physical_reexecution_passed"]
            and status["automatic_checks_passed"]
            and status["native_contact_steps"] == 0
            and fold["passed"]
        ),
    )
    atomic_json(output / "status.json", status)
    atomic_json(output / "qualification.json", report)
    return report
