#!/usr/bin/env python3
"""Versioned offline reassessment; never overwrite or promote an original run.

This checks recorded trajectories with the current evaluator. It is NOT a new
policy episode, physical reexecution, video review, or release qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from instruction_evaluation import evaluate_instruction


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def reassess(source,output):
    source=Path(source);output=Path(output)
    configuration=json.loads((source/"configuration.json").read_text())
    if configuration.get("entrypoint")!="run_instruction_task.py":
        raise ValueError("Only full instruction-task recordings can be reassessed")
    names=("configuration.json","result.json","environment_task.json","events.jsonl","actions.json",
           "evaluator_trajectory.json","evaluator_discoveries.json","disturbance_forces.json",
           "initial_conditions.json","evaluator_room_map.npz")
    inputs={name:sha(source/name) for name in names}
    original=json.loads((source/"result.json").read_text())
    rows=json.loads((source/"evaluator_trajectory.json").read_text())
    actions=json.loads((source/"actions.json").read_text())
    if not rows or len(rows)!=len(actions) or any(
            a["step"]!=r["step"] or a["step"]!=i+1 for i,(a,r) in enumerate(zip(actions,rows))):
        raise ValueError("Incomplete or unaligned full control/evaluator trace")
    if original.get("source_files_unchanged") is not True:
        raise ValueError("The original run did not pass its source-unchanged check")
    events=[json.loads(line) for line in (source/"events.jsonl").read_text().splitlines()]
    move=SimpleNamespace(
        observation_checks=json.loads((source/"evaluator_discoveries.json").read_text()),
        gate=SimpleNamespace(started_step=original.get("relocation_start_step")),
        done_step=original.get("relocation_done_step"),
        trace=json.loads((source/"disturbance_forces.json").read_text()),
        initial=np.asarray(json.loads((source/"initial_conditions.json").read_text())["fixtures"]["target"]["initial_origin_xyz"]))
    evaluator=Path(__file__).with_name("instruction_evaluation.py")
    evaluator_hash=sha(evaluator)
    with np.load(source/"evaluator_room_map.npz") as data:room_map={key:data[key] for key in data.files}
    result=evaluate_instruction(rows,events,actions,move,room_map,hz=20)
    changes={key:dict(original=value,reassessed=result["criteria"].get(key))
             for key,value in original.get("criteria",{}).items()
             if value!=result["criteria"].get(key)}
    report=dict(source_run=str(source.resolve()),status="offline_recorded_trajectory_reassessment_only",
        original_evaluation_version=original.get("evaluation_definition_version",1),
        original_protocol_success=original.get("evaluator_protocol_success"),
        original_task_success=original.get("evaluator_task_success"),criteria_changes=changes,
        evaluation=result,input_sha256=inputs,evaluator_sha256=evaluator_hash,
        reassessment_script_sha256=sha(Path(__file__)),
        inputs_unchanged=all(sha(source/name)==expected for name,expected in inputs.items()),
        evaluator_unchanged=sha(evaluator)==evaluator_hash,
        new_policy_execution=False,independent_physical_audit_pending=True,
        video_review_pending=True,release_ready=False,
        boundary="Original records/classification retained. A changed offline score does not certify a successful release case.")
    if not report["inputs_unchanged"] or not report["evaluator_unchanged"]:
        raise RuntimeError("Inputs or scorer changed during reassessment")
    output.mkdir(parents=True,exist_ok=False)
    (output/"reassessment.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(reassess(args.source_run,args.output),indent=2))


if __name__=="__main__":main()
