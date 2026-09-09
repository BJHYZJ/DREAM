#!/usr/bin/env python3
"""Inventory explicit CLOSED development batches, without pooling success rates.

Reported controller/evaluator flags are not independent audit certificates.
Historical reevaluations, replay diagnostics, standalone runs and curated videos
are deliberately not silently merged into the original planned-attempt records.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_reference(path, root):
    return dict(path=str(path.relative_to(root)), sha256=sha256(path))


def safe_child(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path outside the declared batch: {relative}")
    return path


def summarize_batch(batch):
    batch = Path(batch).resolve()
    protocol_path = batch / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("mode") != "development":
        raise ValueError("This inventory is only for development, not the formal paired study")
    closed_path = batch / "batch_result.json"
    closed = json.loads(closed_path.read_text())  # Require a terminal manifest.
    if closed.get("all_planned_attempts_recorded") is not True:
        raise ValueError("Batch is not closed with all planned attempts recorded")
    planned = protocol["attempts"]
    names = [job["name"] for job in planned]
    if len(names) != len(set(names)) or protocol["planned_attempts"] != len(names):
        raise ValueError("Duplicate/missing planned attempt")
    records_path = batch / "attempts.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    record_names = [record["name"] for record in records]
    if len(record_names) != len(set(record_names)) or set(record_names) != set(names):
        raise ValueError("Finished records must match every planned attempt exactly once")
    by_name = {record["name"]: record for record in records}
    closed_records = closed["attempts"]
    if len(closed_records) != len(records) or any(by_name.get(r["name"]) != r for r in closed_records):
        raise ValueError("Terminal manifest and append-only attempt records disagree")
    if len({r["name"] for r in closed_records}) != len(records):
        raise ValueError("Duplicate terminal attempt")

    protocol_sha = sha256(protocol_path)
    protocol_ok = ((batch / "protocol.sha256").read_text().strip() == protocol_sha
                   and closed.get("protocol_sha256") == protocol_sha)
    frozen = batch / "frozen_workspace/DREAM_code"
    source_mismatches = []
    for relative, expected in protocol["source_sha256"].items():
        path = safe_child(frozen, relative)
        if not path.is_file() or sha256(path) != expected:
            source_mismatches.append(relative)

    rows = []
    for job in planned:
        record = by_name[job["name"]]
        run = safe_child(batch, job["name"])
        task = safe_child(batch, job["task"])
        task_data = json.loads(task.read_text())
        room = safe_child(task.parent, task_data["room_map_file"])
        task_matches = sha256(task) == job["task_sha256"]
        room_matches = room.is_file() and sha256(room) == job["room_map_sha256"]
        identity_matches = all(record.get(k) == job.get(k) for k in job)
        result_path = run / "result.json"
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
        result_matches = (sha256(result_path) == record.get("result_sha256")) if result is not None else None
        # A saved control trace is execution evidence, not proof of task success.
        trace = run / "evaluator_trajectory.jsonl"
        if trace.is_file():
            with trace.open("rb") as stream:
                control_steps = sum(bool(line.strip()) for line in stream)
            trace_ref = file_reference(trace, batch)
        else:
            actions = run / "actions.json"
            if actions.is_file():
                values = json.loads(actions.read_text())
                if not isinstance(values, list):
                    raise ValueError(f"Unexpected saved action format: {actions}")
                control_steps = len(values)
                trace_ref = file_reference(actions, batch)
            else:
                control_steps = 0
                trace_ref = None
        rows.append(dict(batch=batch.name, attempt=job["name"], scene=job["scene"],
            seed=job["seed"], recipe=job["recipe"], variant=job["variant"],
            runner_status=record["status"], recorded_control_steps=control_steps,
            has_saved_execution_evidence=control_steps > 0, control_trace=trace_ref,
            reported_task_success=None if result is None else result.get("evaluator_task_success"),
            reported_strict_success=None if result is None else result.get("evaluator_protocol_success"),
            evaluation_definition_version=None if result is None else result.get("evaluation_definition_version"),
            held_base_transport_m=None if result is None else result.get("held_base_transport_m"),
            stable_placement_s=None if result is None else result.get("stable_placement_s"),
            failed_original_criteria=[] if result is None else [k for k,v in result.get("criteria",{}).items() if v is False],
            source_verified_before=record.get("source_verified_before"),
            source_verified_after=record.get("source_verified_after"),
            planned_record_identity_matches=identity_matches,
            current_task_sha_matches=task_matches, current_room_sha_matches=room_matches,
            original_result_sha_matches=result_matches,
            original_result=None if result is None else file_reference(result_path,batch),
            runner_error=record.get("error"), independent_video_qualification="not_assessed_by_this_inventory"))
    counts = dict(planned=len(rows), saved_execution_traces=sum(r["has_saved_execution_evidence"] for r in rows),
        without_saved_execution_trace=sum(not r["has_saved_execution_evidence"] for r in rows),
        runner_status=dict(Counter(r["runner_status"] for r in rows)),
        reported_original_task_true=sum(r["reported_task_success"] is True for r in rows),
        reported_original_task_false=sum(r["reported_task_success"] is False for r in rows),
        reported_original_task_unavailable=sum(r["reported_task_success"] is None for r in rows),
        reported_original_strict_true=sum(r["reported_strict_success"] is True for r in rows))
    integrity = dict(protocol_sha_matches=protocol_ok,
        frozen_source_mismatches=source_mismatches,
        original_result_mismatches=[r["attempt"] for r in rows if r["original_result_sha_matches"] is False],
        task_or_identity_mismatches=[r["attempt"] for r in rows if not (
            r["current_task_sha_matches"] and r["current_room_sha_matches"] and r["planned_record_identity_matches"])])
    return dict(batch=str(batch), input_files=[file_reference(p,batch) for p in
        (protocol_path,closed_path,records_path)], source_files=len(protocol["source_sha256"]),
        counts=counts, integrity=integrity, attempts=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; earlier inventories remain immutable")
    resolved=[p.resolve() for p in args.batches]
    if len(set(resolved)) != len(resolved):
        raise ValueError("A batch must not be inventoried twice")
    batches=[summarize_batch(p) for p in resolved]
    report=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
        scope="Only the explicitly listed closed instruction development batches",
        boundary="Different development controllers/seeds are NOT pooled into a success rate. Original flags are not independent task/video certificates. Standalone trials, local diagnostics, replays, later rescoring and live batches are outside this inventory. No formal-study inference.",
        release_ready=False,batches=batches)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"inventory.json").write_text(json.dumps(report,indent=2)+"\n")
    fields=["batch","attempt","scene","seed","recipe","variant","runner_status",
        "recorded_control_steps","reported_task_success","reported_strict_success",
        "source_verified_before","source_verified_after","original_result_sha_matches"]
    with (args.output/"attempts.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for batch in batches:
            for row in batch["attempts"]:writer.writerow({k:row[k] for k in fields})
    lines=["# Closed development-batch inventory (not a success-rate study)","",report["boundary"],"",
        "| Batch | Planned | Saved execution traces | No saved trace | Original primary flags (not audited) | Original strict=true |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for batch in batches:
        c=batch["counts"]
        primary=(f"{c['reported_original_task_true']} true; {c['reported_original_task_false']} false; "
                 f"{c['reported_original_task_unavailable']} unavailable")
        lines.append(f"| {Path(batch['batch']).name} | {c['planned']} | {c['saved_execution_traces']} | {c['without_saved_execution_trace']} | {primary} | {c['reported_original_strict_true']} |")
    lines.extend(["","Flags are retained exactly as originally recorded, including older evaluator definitions.",
        "Unavailable primary flags are NOT treated as failed primary tasks; older scorers did not store this endpoint.",
        "A later independently checked reassessment does not overwrite a failed original result.",
        "No saved trace means no saved control evidence was found, not proof that a simulator never started.",
        "See inventory.json for all source/input mismatches and per-attempt failure criteria.",
        "See attempts.csv for the complete row-level inventory of these declared batches.",""])
    for batch in batches:
        check=batch["integrity"]
        if not check["protocol_sha_matches"] or any(check[k] for k in (
                "frozen_source_mismatches","original_result_mismatches","task_or_identity_mismatches")):
            lines.extend([f"## Retained integrity issue: {Path(batch['batch']).name}","",
                json.dumps(check,ensure_ascii=False),
                "This report records the discrepancy; it does not repair the old source or certify its results.",""])
    (args.output/"README.md").write_text("\n".join(lines))
    (args.output/"SHA256.json").write_text(json.dumps({p.name:sha256(p) for p in sorted(args.output.iterdir()) if p.is_file()},indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output),batches=len(batches),
        planned=sum(b["counts"]["planned"] for b in batches),release_ready=False)))


if __name__=="__main__":
    main()
