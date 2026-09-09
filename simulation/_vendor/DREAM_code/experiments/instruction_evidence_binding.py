"""Bind fresh physical/record audits to complete, unchanged saved inputs.

No learned-policy trial is launched, and no original run is changed. The frozen
study controller supplies both audit executables; this wrapper only records
before/after hashes and refuses mismatched source/configuration/attempt identity.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def child(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Path outside declared root: {relative}")
    return path


def tree_hashes(root):
    """Include all recorded input bytes, including sensor archives and videos."""
    root = Path(root).resolve(); values = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.relative_to(root).parts:
            continue  # Interpreter cache is not a recording or source input.
        if path.is_symlink():
            raise ValueError(f"Symlink inside recorded inputs: {path}")
        if path.is_file():
            values[path.relative_to(root).as_posix()] = digest(path)
    return values


def validate_attempt(batch, record):
    batch = Path(batch).resolve()
    protocol = json.loads((batch / "protocol.json").read_text())
    if (batch / "protocol.sha256").read_text().strip() != digest(batch / "protocol.json"):
        raise ValueError("Protocol digest mismatch")
    matches = [job for job in protocol["attempts"] if job["name"] == record["name"]]
    if len(matches) != 1 or any(record.get(k) != v for k, v in matches[0].items()):
        raise ValueError("Attempt differs from declared protocol")
    job = matches[0]; run = child(batch, job["name"])
    if record.get("status") != "completed" or record.get("returncode") != 0:
        raise ValueError("Only a completed saved task can be input-bound audited")
    if not all(record.get(k) is True for k in ("source_verified_before", "source_verified_after",
                                               "task_verified_before", "task_verified_after")):
        raise ValueError("Original frozen checks did not pass")
    if digest(run / "result.json") != record["result_sha256"]:
        raise ValueError("Result changed since task completion")
    frozen = batch / "frozen_workspace/DREAM_code"
    hashes = protocol["source_sha256"]
    if not hashes:
        raise ValueError("Missing frozen source manifest")
    for label in ("before", "after"):
        if json.loads((run / f"source_hashes_{label}.json").read_text()) != hashes:
            raise ValueError("Run source differs from study controller")
    for name, expected in hashes.items():
        if digest(child(frozen, name)) != expected or digest(child(run / "source_snapshot", name)) != expected:
            raise ValueError(f"Frozen/saved source mismatch: {name}")
    task_path = child(batch, job["task"]); task = json.loads(task_path.read_text())
    if digest(task_path) != job["task_sha256"] or digest(child(task_path.parent, task["room_map_file"])) != job["room_map_sha256"]:
        raise ValueError("Frozen task/room changed")
    saved = json.loads((run / "environment_task.json").read_text())
    if saved != task or digest(run / "evaluator_room_map.npz") != job["room_map_sha256"]:
        raise ValueError("Executed task/room differs from planned input")
    config = json.loads((run / "configuration.json").read_text())
    expected = dict(variant=job["variant"], threshold=protocol["threshold"],
        navigation_budget=protocol["navigation_budget"], heading_navigation=protocol["heading_navigation"],
        video=True, entrypoint="run_instruction_task.py")
    if any(type(config.get(k)) is not type(v) or config.get(k) != v for k, v in expected.items()):
        raise ValueError("Executed configuration differs from study")
    return protocol, run, frozen


def audit_bound_attempt(batch, record, output):
    """Return status and immutable-input proof for fresh saved-control audits."""
    batch = Path(batch).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(name=record["name"], requested_endpoint="task", input_binding_passed=False,
        primary_task_checks_passed=False, automatic_checks_passed=False,
        visual_acceptance_pending=True, release_ready=False)
    try:
        protocol, run, frozen = validate_attempt(batch, record)
        before = dict(schema_version=1, source_run=str(run), protocol_sha256=digest(batch / "protocol.json"),
            original_result_sha256=record["result_sha256"], wrapper_sha256=digest(__file__),
            run_files_sha256=tree_hashes(run), frozen_source_sha256=protocol["source_sha256"],
            created_unix_s=time.time(), boundary="Hashes captured BEFORE fresh control reexecution and observation review.")
        (output / "audit_inputs_before.json").write_text(json.dumps(before, indent=2)+"\n")
        steps = [([sys.executable, str(frozen / "experiments/replay_instruction_actions.py"),
                   "--source-run", str(run), "--output", str(output / "physical")], "physical", 1200),
                 ([sys.executable, str(frozen / "experiments/review_instruction_record.py"),
                   "--run", str(run), "--physical-audit", str(output / "physical/audit.json"),
                   "--output", str(output / "record"), "--annotate", "--endpoint", "task"], "record", 1800)]
        for command, name, timeout in steps:
            with (output / f"{name}.log").open("x") as stream:
                completed = subprocess.run(command, cwd=frozen, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=timeout)
            status[name+"_returncode"] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f"Fresh {name} audit failed; original record is retained")
        physical = json.loads((output / "physical/audit.json").read_text())
        review = json.loads((output / "record/record_review.json").read_text())
        after = tree_hashes(run)
        checks = dict(all_recorded_inputs_unchanged=after == before["run_files_sha256"],
            protocol_unchanged=digest(batch / "protocol.json") == before["protocol_sha256"],
            wrapper_unchanged=digest(__file__) == before["wrapper_sha256"],
            frozen_source_unchanged=all(digest(child(frozen,k)) == v for k,v in before["frozen_source_sha256"].items()),
            physical_passed=physical.get("physical_reexecution_passed") is True,
            physical_primary_success=physical["evaluation"].get("evaluator_task_success") is True,
            same_physical_run=Path(physical["source_run"]).resolve() == run,
            same_record_run=Path(review["source_run"]).resolve() == run,
            record_binds_physical=review["physical_audit_sha256"] == digest(output / "physical/audit.json"),
            record_primary_passed=review.get("primary_task_record_review_passed") is True,
            no_native_contact=physical["contact_audit"]["native_environment_contact_control_steps"] == 0)
        proof = dict(schema_version=1, source_run=str(run), checks=checks,
            input_binding_passed=all(checks.values()), before_manifest_sha256=digest(output / "audit_inputs_before.json"),
            physical_output_sha256=tree_hashes(output / "physical"),
            recording_output_sha256=tree_hashes(output / "record"),
            finished_unix_s=time.time(), release_ready=False,
            boundary="Same saved controls and full saved sensor inputs, not another policy trial or author approval.")
        (output / "evidence_input_binding.json").write_text(json.dumps(proof, indent=2)+"\n")
        status.update(input_binding_passed=proof["input_binding_passed"],
            primary_task_checks_passed=proof["input_binding_passed"],
            automatic_checks_passed=proof["input_binding_passed"] and review.get("record_review_passed") is True,
            native_contact_steps=physical["contact_audit"]["native_environment_contact_control_steps"],
            review_definition_version=review["review_definition_version"])
    except Exception as error:
        status["error"] = repr(error)
    (output / "status.json").write_text(json.dumps(status, indent=2)+"\n")
    return status
