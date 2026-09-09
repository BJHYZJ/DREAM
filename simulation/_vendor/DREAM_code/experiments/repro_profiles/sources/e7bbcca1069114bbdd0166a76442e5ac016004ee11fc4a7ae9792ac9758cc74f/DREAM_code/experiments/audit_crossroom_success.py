#!/usr/bin/env python3
"""Independently check one real run and physically re-execute its controls.

This is an evidence check, not another learned-policy success. All reports are
new files outside the immutable source run, including failed audits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

from make_crossroom_gallery import audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=audit(args.run)
    report.pop("events")
    report.pop("frames")
    (args.output/"record_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    replay=args.output/"physical_control_reexecution.json"
    command=[sys.executable,str(Path(__file__).with_name("replay_learned_actions.py")),
             "--source-run",str(args.run.resolve()),"--output",str(replay.resolve()),"--contact-audit"]
    with (args.output/"reexecution.log").open("w") as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    control=json.loads(replay.read_text())
    contact=control["contact_audit"]
    report["control_reexecution"]=control
    report["checks"]["physical_control_reexecution"]=control["passed"]
    report["checks"]["no_native_environment_contact_above_threshold"]=contact["native_environment_contact_control_steps"]==0
    batch=args.run.parent
    pre=batch/f"{args.run.name}.preexecution_source_check.json"
    post=batch/f"{args.run.name}.postexecution_source_check.json"
    if pre.is_file() and post.is_file():
        protocol=json.loads((batch/"protocol.json").read_text())
        mismatches={key:dict(expected=expected,actual=hashlib.sha256((batch/"frozen_source"/key).read_bytes()).hexdigest())
            for key,expected in protocol["source_sha256"].items()
            if hashlib.sha256((batch/"frozen_source"/key).read_bytes()).hexdigest()!=expected}
        index=int(args.run.name.split("_",1)[0])-1
        task=protocol["tasks"][index]
        # Resolve within the restored batch, not the original server's path.
        task_file=batch/"frozen_tasks"/f"{index+1:02d}"/Path(task["file"]).name
        task_values=json.loads(task_file.read_text())
        task_checks=dict(task_hash=hashlib.sha256(task_file.read_bytes()).hexdigest()==task["sha256"],
            room_hash=hashlib.sha256((task_file.parent/task_values["room_map_file"]).read_bytes()).hexdigest()==task["room_map_sha256"],
            recorded_task=task_values==json.loads((args.run/"environment_task.json").read_text()),
            recorded_room=hashlib.sha256((args.run/"evaluator_room_map.npz").read_bytes()).hexdigest()==task["room_map_sha256"])
        provenance=dict(preexecution=json.loads(pre.read_text()),postexecution=json.loads(post.read_text()),
                        final_source_mismatches=mismatches,task_checks=task_checks)
        provenance["passed"]=bool(provenance["preexecution"]["passed"] and provenance["postexecution"]["passed"]
                                  and not mismatches and all(task_checks.values()))
        report["batch_provenance"]=provenance
        report["checks"]["checked_frozen_batch_provenance"]=provenance["passed"]
    report["passed"]=all(report["checks"].values())
    report["audit_source_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(__file__).with_name("physics_contact_audit.py"),
         Path(__file__).with_name("replay_learned_actions.py"),Path(__file__).with_name("make_crossroom_gallery.py")]}
    frozen=args.output/"audit_source"
    frozen.mkdir()
    for name in report["audit_source_sha256"]:
        shutil.copy2(Path(__file__).with_name(name),frozen/name)
    (args.output/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(run=str(args.run),passed=report["passed"],checks=report["checks"])),flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
