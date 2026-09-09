#!/usr/bin/env python3
"""Audit completed successful batch tasks, optionally watching the finite batch.

Fresh-physics replay and saved-record checks are not additional policy trials.
Automatic checks leave visual acceptance and research release explicitly pending.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--watch",action="store_true")
    p.add_argument("--all-task-successes",action="store_true",
        help="Also verify continuous-tracking task successes; never relabel them as strict success videos")
    p.add_argument("--record-review-script",type=Path,
        help="Optional explicitly versioned stricter review script; original task/physics source is never changed")
    p.add_argument("--input-bound",action="store_true",
        help="Hash all saved inputs before/after fresh frozen-source physics and primary-record audits")
    args=p.parse_args();batch=args.batch.resolve();output=args.output.resolve()
    if args.input_bound and (args.record_review_script or not args.all_task_successes):
        p.error("--input-bound requires --all-task-successes and the study's unchanged review script")
    output.mkdir(parents=True,exist_ok=False)
    frozen=batch/"frozen_workspace/DREAM_code"
    if not (frozen/"experiments/replay_instruction_actions.py").is_file():raise ValueError("Missing frozen instruction source")
    review_script=(args.record_review_script or frozen/"experiments/review_instruction_record.py").resolve()
    if not review_script.is_file():raise FileNotFoundError(review_script)
    status={};futures={}

    def audit(record):
        if args.input_bound:
            from instruction_evidence_binding import audit_bound_attempt
            report=audit_bound_attempt(batch,record,output/record["name"])
            print(json.dumps(report),flush=True)
            return report
        name=record["name"];directory=output/name;directory.mkdir()
        report=dict(name=name,task_protocol_success=record["protocol_success"],
            requested_endpoint="task" if args.all_task_successes else "strict",
            automatic_checks_passed=False,visual_acceptance_pending=True,release_ready=False)
        try:
            command=[sys.executable,str(frozen/"experiments/replay_instruction_actions.py"),
                "--source-run",str(batch/name),"--output",str(directory/"physical")]
            with (directory/"physical.log").open("x") as stream:
                result=subprocess.run(command,cwd=frozen,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
            report["physical_returncode"]=result.returncode
            if result.returncode!=0:raise RuntimeError("Physical replay command failed")
            physical=json.loads((directory/"physical/audit.json").read_text())
            report.update(physical_reexecution_passed=physical["physical_reexecution_passed"],
                native_contact_steps=physical["contact_audit"]["native_environment_contact_control_steps"])
            report["record_review_script"]=str(review_script)
            command=[sys.executable,str(review_script),
                "--run",str(batch/name),"--physical-audit",str(directory/"physical/audit.json"),
                "--output",str(directory/"record"),"--annotate"]
            if args.all_task_successes:command.extend(["--endpoint","task"])
            with (directory/"record.log").open("x") as stream:
                result=subprocess.run(command,cwd=frozen,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
            report["record_returncode"]=result.returncode
            if result.returncode!=0:raise RuntimeError("Recorded-video/source/identity review did not pass")
            review=json.loads((directory/"record/record_review.json").read_text())
            report["automatic_checks_passed"]=bool(review["record_review_passed"])
            report["primary_task_checks_passed"]=bool(review.get("primary_task_record_review_passed",False))
            report["review_definition_version"]=review.get("review_definition_version",1)
        except Exception as error:report["error"]=repr(error)
        (directory/"status.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report),flush=True);return report

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            attempts=batch/"attempts.jsonl"
            if attempts.exists():
                for line in attempts.read_text().splitlines():
                    try:record=json.loads(line)
                    except json.JSONDecodeError:continue  # Last concurrent write may not be complete yet.
                    name=record["name"]
                    if name in status:continue
                    eligible=record.get("protocol_success") or (args.all_task_successes and record.get("task_success"))
                    if eligible and record.get("source_verified_after"):
                        status[name]=dict(status="audit_running",release_ready=False)
                        futures[pool.submit(audit,record)]=name
                    else:status[name]=dict(status="task_failed_or_unverified_not_a_success_video",release_ready=False)
            for future,name in list(futures.items()):
                if future.done():status[name]=future.result();del futures[future]
            (output/"audit_status.json").write_text(json.dumps(dict(batch=str(batch),tasks=status,
                visual_acceptance_pending=True,release_ready=False),indent=2)+"\n")
            batch_complete=(batch/"batch_result.json").is_file()
            if (batch_complete or not args.watch) and not futures:break
            time.sleep(10)


if __name__=="__main__":main()
