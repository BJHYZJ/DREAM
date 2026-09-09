#!/usr/bin/env python3
"""Freeze inputs, run every specified scene and both variants, retain failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpus",nargs="+",default=["0","1"])
    parser.add_argument("--variants",nargs="+",choices=["dynamic","static"],default=["dynamic","static"])
    parser.add_argument("--timeout",type=float,default=1800.)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1]
    sources=sorted(list((repo/"src/dream").rglob("*.py"))+list((repo/"experiments").glob("*.py")))
    hashes={str(p.relative_to(repo)):digest(p) for p in sources}
    for path in sources:
        destination=args.output/"frozen_source"/path.relative_to(repo)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    tasks=[]
    for directory in args.tasks:
        for path in sorted(directory.glob("[0-9][0-9]_*.json")):
            value=json.loads(path.read_text())
            if any(t["scene"]==value["scene"] for t in tasks):
                raise ValueError(f"Duplicate scene: {value['scene']}")
            destination=args.output/"frozen_tasks"/path.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
            tasks.append(dict(scene=value["scene"],path=str(destination),sha256=digest(destination)))
    tasks.sort(key=lambda t:Path(t["path"]).name)
    versions={name:importlib.metadata.version(name) for name in (
        "torch","numpy","scipy","mani_skill","sapien","transformers","huggingface-hub","trimesh","scikit-learn")}
    protocol=dict(status="frozen_before_suite_execution",source_sha256=hashes,tasks=tasks,
        variants=args.variants,model_set="production",navigation_budget=20,wrist_scans=True,
        dynamic_video=True,static_video=False,wall_timeout_s=args.timeout,gpus=args.gpus,
        versions=versions,created_unix_s=time.time(),
        scope="Development scene suite, not held-out generalization. All ten interiors have been inspected in prior integration attempts; v2 follows disclosed v1 failures and common controller/setup fixes. All scheduled outcomes are retained; do not pool tuned retries into a success-rate claim.")
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    jobs=queue.Queue()
    for index,task in enumerate(tasks):
        for variant in args.variants:
            jobs.put((index,task,variant))
    guard=threading.Lock()
    records=[]

    def worker(gpu):
        while True:
            try:
                index,task,variant=jobs.get_nowait()
            except queue.Empty:
                return
            name=f"{index+1:02d}_{task['scene']}_{variant}"
            destination=args.output/name
            record=dict(name=name,scene=task["scene"],variant=variant,gpu=gpu,started_unix_s=time.time())
            changed=[name for name,expected in hashes.items() if digest(repo/name)!=expected]
            if changed:
                record.update(status="not_run_source_changed",changed=changed)
            else:
                command=[sys.executable,str(repo/"experiments/maniskill_learned_dynamic.py"),
                    "--output",str(destination),"--task-json",task["path"],"--variant",variant,
                    "--model-set","production","--navigate","--grasp","--deliver"]
                if variant=="dynamic":
                    command.append("--video")
                record["command"]=command
                env=os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
                print(json.dumps(dict(event="start",**record)),flush=True)
                with (args.output/f"{name}.stdout.log").open("w") as stream:
                    process=subprocess.Popen(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    try:
                        code=process.wait(timeout=args.timeout)
                        record.update(status="completed" if code==0 else "execution_failure",returncode=code)
                    except subprocess.TimeoutExpired:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            process.wait(timeout=15)
                        record.update(status="timeout",returncode=process.returncode)
                if (destination/"result.json").is_file():
                    result=json.loads((destination/"result.json").read_text())
                    record.update(task_success=result.get("evaluator_task_success",False),
                        protocol_success=result.get("evaluator_protocol_success",False),
                        initial_target_observed=result.get("evaluator_initial_target_observed",False),
                        disturbance_valid=result.get("evaluator_disturbance_valid",False),
                        base_travel_m=result.get("evaluator_base_travel_m"),env_steps=result.get("env_steps"))
                if (destination/"failure.json").is_file():
                    record["failure"]=json.loads((destination/"failure.json").read_text())
            record["finished_unix_s"]=time.time()
            with guard:
                records.append(record)
                with (args.output/"attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(record)+"\n")
            print(json.dumps(dict(event="finished",**record)),flush=True)
            jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        list(executor.map(worker,args.gpus))
    summary=dict(scene_count=len(tasks),attempts=sorted(records,key=lambda r:r["name"]),
        counts={v:dict(scheduled=len(tasks),completed=sum(r["variant"]==v and r["status"]=="completed" for r in records),
            task_success=sum(r["variant"]==v and r.get("task_success",False) for r in records),
            protocol_success=sum(r["variant"]==v and r.get("protocol_success",False) for r in records)) for v in args.variants})
    (args.output/"suite_result.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
