#!/usr/bin/env python3
"""Input-bound summary of the declared 60-attempt paired instruction study.

Incomplete, technically failed or unaudited outcomes never become successes or
silently disappear from the denominator. Partial inspection produces NO rate/CI.
This checker does not grant author acceptance or public release permission.
"""
import argparse
from collections import Counter
import csv
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import instruction_evaluation
import instruction_study_statistics
import learned_evaluation
from instruction_evidence_binding import child, digest, tree_hashes, validate_attempt
from run_instruction_frozen_batch import validate_design


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON: {value}")))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_values(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(same_values(actual[k],v) for k,v in expected.items())
    if isinstance(expected, float):
        return type(actual) in (int,float) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-8)
    return type(actual) is type(expected) and actual == expected


def validate_result_endpoint(original, record):
    """Keep valid frozen task failures, including the declared sim-time limit.

The runner records normal budget/grasp aborts in `error`; that is not a missing
endpoint. Such failures remain in the denominator after source/trace validation.
Evaluator exceptions or absent endpoints remain unresolved, never imputed.
"""
    require(not original.get("evaluator_error"), "Recorded evaluator exception")
    require(type(original.get("evaluator_task_success")) is bool and type(original.get("evaluator_protocol_success")) is bool,
            "Missing/nonboolean outcome")
    require(original["evaluator_task_success"] is record.get("task_success") and
            original["evaluator_protocol_success"] is record.get("protocol_success"), "Result/attempt flags disagree")
    require(original.get("evaluation_definition_version") == 4, "Different evaluation definition")


def recorded_output_path(record):
    """Resolve a stored identifier, never read from the old server location."""
    command=record.get("command",[])
    require(isinstance(command,list) and all(isinstance(v,str) for v in command)
            and command.count("--output")==1, "Missing/ambiguous recorded output argument")
    index=command.index("--output")
    require(index+1<len(command), "Missing recorded output path")
    value=Path(command[index+1])
    require(value.is_absolute() and ".." not in value.parts and value.name==record["name"], "Recorded output identity mismatch")
    return value


def source_identifier_matches(value, current_run, recorded_run):
    # Archives retain original audit bytes/hashes. Exact historical identifiers
    # come from the saved command; all actual data is read under --batch/--audits.
    return Path(value) in (Path(current_run),Path(recorded_run))


def validate_study_protocol(batch):
    batch = Path(batch).resolve(); protocol = read(batch / "protocol.json")
    require(protocol.get("mode") == "prespecified_60_attempt_paired_study", "Not the formal paired study")
    require((batch / "protocol.sha256").read_text().strip() == digest(batch / "protocol.json"), "Protocol hash mismatch")
    require(protocol.get("analysis_plan") == instruction_study_statistics.study_analysis_plan(), "Analysis plan differs from prespecification")
    jobs = protocol["attempts"]
    require(protocol.get("planned_attempts") == len(jobs) == 60, "Expected all sixty declared attempts")
    require(len({j["name"] for j in jobs}) == 60, "Duplicate attempt names")
    keys = {(j["scene"],j["seed"],j["variant"]) for j in jobs}
    houses = {j["scene"] for j in jobs}; seeds = protocol["seeds"]
    require(keys == {(h,s,v) for h in houses for s in seeds for v in ("dynamic","static")} and len(keys) == 60,
            "Missing, duplicate or mismatched house/seed/variant pairing")
    require(protocol.get("simulation_budget_s") == 1200 and protocol.get("heading_navigation") is True,
            "Study budget/navigation differs from declared protocol")
    task_by_house = {}; task_by_pair = {}
    for job in jobs:
        task_path = child(batch,job["task"]); task = read(task_path)
        require(digest(task_path) == job["task_sha256"], "Task digest mismatch")
        require(digest(child(task_path.parent,task["room_map_file"])) == job["room_map_sha256"], "Room digest mismatch")
        require(task["scene"] == job["scene"] and type(task["seed"]) is int and task["seed"] == job["seed"]
                and task["recipe"]["id"] == job["recipe"], "Task identity mismatch")
        pair = (job["scene"],job["seed"])
        require(pair not in task_by_pair or task_by_pair[pair] == (job["task_sha256"],job["room_map_sha256"]),
                "Memory variants do not share the same task/room")
        task_by_pair[pair] = (job["task_sha256"],job["room_map_sha256"])
        fixed = {k:v for k,v in task.items() if k not in ("seed","initial_sample_preview")}
        require(job["scene"] not in task_by_house or task_by_house[job["scene"]][0] == fixed,
                "Fixture configuration differs across paired start seeds")
        task_by_house[job["scene"]] = (fixed,task)
    validate_design([v[1] for v in task_by_house.values()],seeds,protocol["variants"],True)
    frozen = batch / "frozen_workspace/DREAM_code"; hashes = protocol["source_sha256"]
    require(bool(hashes), "Missing source manifest")
    for name,value in hashes.items():
        require(digest(child(frozen,name)) == value, f"Frozen source changed: {name}")
    # Read-only rescoring and statistics must use the exact prespecified bytes,
    # not an evaluator silently changed after seeing the outcomes.
    for module in (instruction_evaluation, learned_evaluation, instruction_study_statistics):
        path = Path(inspect.getfile(module)).resolve()
        require(digest(path) == hashes.get("experiments/"+path.name), "Current scorer/statistics differs from frozen source")
    return protocol


def score_trace(run, original, *, physical_rows=None):
    rows = read(run / "evaluator_trajectory.json") if physical_rows is None else physical_rows
    actions = read(run / "actions.json")
    require(bool(rows) and len(rows) == len(actions), "Missing/incomplete physics controls")
    for i,(row,action) in enumerate(zip(rows,actions),1):
        require(type(row["step"]) is int and row["step"] == action["step"] == i, "Noncontiguous control/trajectory steps")
        require(type(row["bilateral_contact"]) is bool and type(row["inside_receptacle"]) is bool, "Invalid physical booleans")
        for key in ("base_xyyaw","target_xyz","target_velocity"):
            require(len(row[key]) == 3 and np.isfinite(row[key]).all(), "Invalid physical state")
        require(np.isfinite(action["action"]).all() and math.isfinite(row["receptacle_contact_force_n"]), "Nonfinite action/contact")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines() if line.strip()]
    require(events and events[0]["instruction"]["text"] == read(run / "environment_task.json")["instruction"], "Wrong executed instruction")
    move = SimpleNamespace(observation_checks=read(run / "evaluator_discoveries.json"),
        gate=SimpleNamespace(started_step=original.get("relocation_start_step")),
        done_step=original.get("relocation_done_step"), trace=read(run / "disturbance_forces.json"),
        initial=np.asarray(read(run / "initial_conditions.json")["fixtures"]["target"]["initial_origin_xyz"]))
    with np.load(run / "evaluator_room_map.npz",allow_pickle=False) as data:
        room_map = {key:data[key] for key in data.files}
    return instruction_evaluation.evaluate_instruction(rows,events,actions,move,room_map,hz=20), len(rows)


def validate_bound_audit(batch, run, directory, protocol, original, control_steps, current_inputs, recorded_run):
    before = read(directory / "audit_inputs_before.json"); proof = read(directory / "evidence_input_binding.json")
    require(proof.get("input_binding_passed") is True and bool(proof.get("checks"))
            and all(v is True for v in proof["checks"].values()), "Input-bound audit did not pass")
    require(digest(directory / "audit_inputs_before.json") == proof["before_manifest_sha256"], "Before-manifest changed")
    require(source_identifier_matches(before["source_run"],run,recorded_run) and
            source_identifier_matches(proof["source_run"],run,recorded_run), "Binding refers to another run")
    require(before["protocol_sha256"] == digest(batch / "protocol.json") and before["original_result_sha256"] == digest(run / "result.json"), "Wrong protocol/result binding")
    require(before["run_files_sha256"] == current_inputs and before["frozen_source_sha256"] == protocol["source_sha256"], "Audit inputs changed")
    require(tree_hashes(directory / "physical") == proof["physical_output_sha256"] and
            tree_hashes(directory / "record") == proof["recording_output_sha256"], "Audit outputs changed")
    physical = read(directory / "physical/audit.json"); recorded = read(directory / "record/record_review.json")
    require(source_identifier_matches(physical["source_run"],run,recorded_run) and
            source_identifier_matches(recorded["source_run"],run,recorded_run), "Audit source identity mismatch")
    require(physical.get("physical_reexecution_passed") is True and physical["steps"] == control_steps, "Incomplete physical replay")
    require(type(physical["tolerance"]) in (int,float) and 0 <= physical["tolerance"] <= 1e-4, "Relaxed replay tolerance")
    errors = read(directory / "physical/control_errors.json")
    require(len(errors) == control_steps, "Incomplete per-control errors")
    for key in ("base_xy_m","yaw_rad","tcp_xyz_m","target_xyz_m"):
        require(all(type(e["step"]) is int and e["step"] == i+1 and math.isfinite(e[key]) and e[key] >= 0 for i,e in enumerate(errors)), "Invalid replay errors")
        maximum = max(e[key] for e in errors)
        require(maximum <= physical["tolerance"] and same_values(physical["maximum_error"][key],maximum), "Replay state mismatch")
    require(all(physical.get(k) is True for k in ("source_recording_unchanged","replay_source_unchanged","replay_environment_source_matches_recording")), "Replay source not verified")
    require(read(directory / "physical/replay_source_hashes.json") == protocol["source_sha256"], "Replay used another source snapshot")
    require(physical["contact_audit"]["native_environment_contact_control_steps"] == 0, "Above-threshold native-house contact")
    evaluation,_ = score_trace(run,original,physical_rows=read(directory / "physical/evaluator_trajectory.json"))
    require(same_values(physical["evaluation"],evaluation) and evaluation["evaluator_task_success"] is True, "Physical rescoring does not certify task completion")
    require(recorded.get("review_definition_version") == 4 and recorded.get("scoring_reassessment") is None,
            "Formal outcomes cannot use a changed/legacy scorer")
    require(recorded.get("primary_task_record_review_passed") is True and bool(recorded.get("primary_task_checks"))
            and all(v is True for v in recorded["primary_task_checks"].values()), "Primary recording audit failed")
    require(recorded["memory_variant"] == read(run / "configuration.json")["variant"], "Wrong memory-variant recording")
    require(recorded["physical_audit_sha256"] == digest(directory / "physical/audit.json")
            and recorded["events_sha256"] == digest(run / "events.jsonl"), "Recording audit is stale")
    require(recorded["review_script_sha256"] == protocol["source_sha256"]["experiments/review_instruction_record.py"], "Changed recording reviewer")
    for name,value in recorded["review_dependency_sha256"].items():
        require(protocol["source_sha256"].get("experiments/"+name) == value, "Changed review dependency")
    for name,value in recorded["source_video_sha256"].items():
        require(digest(child(run,name+".mp4")) == value, "Raw recording changed")
    require(digest(directory / "record/reviewer_view.mp4") == recorded["annotated_video_sha256"], "Annotated video changed")
    require(recorded["decoded_frames"] == control_steps // 4, "Video/control duration mismatch")
    return dict(binding_sha256=digest(directory / "evidence_input_binding.json"),
        primary_verified=True, strict_record_verified=recorded.get("record_review_passed") is True)


def inspect_attempt(batch, audits, protocol, record):
    row = dict(name=record["name"],scene=record["scene"],seed=record["seed"],variant=record["variant"],
        runner_status=record["status"],task_success=None,reported_task_success=record.get("task_success"),
        reported_strict_success=record.get("protocol_success"),eligible_for_summary=False,issues=[])
    if record["status"] != "completed" or record.get("returncode") != 0:
        row.update(technical_failure=True,issues=["Technical execution/orchestration failure, not imputed as task failure"])
        return row
    try:
        _,run,_ = validate_attempt(batch,record)
        original = read(run / "result.json")
        row["reported_task_error"] = original.get("error")
        validate_result_endpoint(original,record)
        inputs = tree_hashes(run)
        evaluation,steps = score_trace(run,original)
        require(all(k in original and same_values(original[k],v) for k,v in evaluation.items()), "Saved score differs from frozen evaluator")
        append_rows = [json.loads(line) for line in (run / "evaluator_trajectory.jsonl").read_text().splitlines() if line.strip()]
        require(append_rows == read(run / "evaluator_trajectory.json"), "Saved and append-only trajectories disagree")
        row.update(control_steps=steps,simulation_seconds=steps/20,
            initial_conditions_sha256=inputs["initial_conditions.json"],input_sha256=inputs,
            failed_task_criteria=[k for k,v in evaluation["task_criteria"].items() if v is False],
            frozen_rescoring_matched=True,technical_failure=False)
        if original["evaluator_task_success"]:
            row["independent_audit"] = validate_bound_audit(batch,run,child(audits,record["name"]),
                protocol,original,steps,inputs,recorded_output_path(record))
        require(tree_hashes(run) == inputs, "Completed input changed during summary")
        row.update(task_success=original["evaluator_task_success"],eligible_for_summary=True)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        row["issues"].append(repr(error))
    return row


def summarize(batch, audits, *, allow_partial=False):
    batch=Path(batch).resolve(); audits=Path(audits).resolve()
    protocol=validate_study_protocol(batch); protocol_sha=digest(batch / "protocol.json")
    records_path=batch / "attempts.jsonl"
    data=records_path.read_bytes() if records_path.exists() else b""
    require(not data or data.endswith(b"\n"), "Concurrent incomplete attempt record; retry without restarting tasks")
    records=[json.loads(line) for line in data.splitlines() if line.strip()]
    jobs={j["name"]:j for j in protocol["attempts"]}
    require(len({r["name"] for r in records}) == len(records), "Duplicate completed attempt")
    for record in records:
        require(record["name"] in jobs and all(record.get(k) == v for k,v in jobs[record["name"]].items()), "Undeclared/changed attempt")
        require(protocol["created_unix_s"] <= record["started_unix_s"] <= record["finished_unix_s"], "Protocol was not declared before execution")
    closed_path=batch / "batch_result.json"; closed=read(closed_path) if closed_path.exists() else None
    complete=closed is not None and closed.get("all_planned_attempts_recorded") is True
    if complete:
        require(len(records) == 60 and closed["protocol_sha256"] == protocol_sha, "Incomplete terminal study")
        require(sorted(closed["attempts"],key=lambda r:r["name"]) == sorted(records,key=lambda r:r["name"]), "Terminal/append-only records disagree")
    require(complete or allow_partial, "Study still running: use --allow-partial for evidence inspection WITHOUT a rate/CI")
    rows=[]
    for record in sorted(records,key=lambda r:r["name"]):
        row=inspect_attempt(batch,audits,protocol,record); rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ("input_sha256",)}),flush=True)
    by_pair={}
    for row in rows:
        pair=(row["scene"],row["seed"])
        if "initial_conditions_sha256" in row:
            by_pair.setdefault(pair,[]).append(row)
    for paired in by_pair.values():
        if len(paired)==2 and paired[0]["initial_conditions_sha256"] != paired[1]["initial_conditions_sha256"]:
            for row in paired:
                row["eligible_for_summary"]=False;row["task_success"]=None
                row["issues"].append("Memory variants did not share identical initial conditions")
    latest=records_path.read_bytes() if records_path.exists() else b""
    require(latest.startswith(data) and (not complete or latest == data), "Attempt history changed rather than appended")
    require(digest(batch / "protocol.json") == protocol_sha, "Protocol changed during inspection")
    available=complete and len(rows)==60 and all(r["eligible_for_summary"] for r in rows)
    report=dict(schema_version=1,scope="complete_frozen_study" if complete else "partial_evidence_inspection_no_aggregate",
        protocol_sha256=protocol_sha,analysis_plan=protocol["analysis_plan"],planned_attempts=60,
        inspected_completed_attempts=len(rows),all_planned_attempts_recorded=complete,
        status_counts=dict(Counter(r["runner_status"] for r in rows)),
        missing_attempts=sorted(set(jobs)-{r["name"] for r in records}),attempts=rows,
        aggregate_available=available,statistics=None,
        analyzer_sha256=digest(__file__),technical_failures=[r["name"] for r in rows if r.get("technical_failure")],
        recorded_task_exceptions=[dict(name=r["name"],error=r["reported_task_error"])
            for r in rows if r.get("reported_task_error")],
        unresolved_evidence=[dict(name=r["name"],issues=r["issues"]) for r in rows if not r["eligible_for_summary"]],
        research_release_ready=False,author_acceptance_pending=True,
        boundary="Development-selected houses, not held-out generalization. All declared attempts retained. Curated videos and audit replays are not this denominator. Missing/technical/unaudited outcomes are not imputed. No public release approval.")
    if available:
        report["statistics"]=instruction_study_statistics.summarize_paired_outcomes(rows)
        # Keep the kernel's own limited scope explicit; validation is supplied
        # by this surrounding evidence report, not attributed to the kernel.
        report["input_and_success_audits_validated_for_all_outcomes"]=True
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=Path,required=True)
    parser.add_argument("--audits",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-partial",action="store_true")
    args=parser.parse_args()
    require(not args.output.exists(), "Use a new report directory; preserve older inspections")
    report=summarize(args.batch,args.audits,allow_partial=args.allow_partial)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output / "study_analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    columns=("name","scene","seed","variant","runner_status","reported_task_success","reported_strict_success",
        "task_success","eligible_for_summary","simulation_seconds","technical_failure")
    with (args.output / "attempts.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in report["attempts"]:writer.writerow({k:row.get(k) for k in columns})
    print(json.dumps({k:v for k,v in report.items() if k not in ("attempts",)},indent=2),flush=True)
    if not args.allow_partial and not report["aggregate_available"]:
        raise SystemExit(2)


if __name__ == "__main__":main()
