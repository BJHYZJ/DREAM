#!/usr/bin/env python3
"""Run all supplied cross-room tasks from an immutable local source copy.

All outcomes are retained. Selecting successful examples afterward does not
turn this development batch into a 100% success-rate benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
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
    parser.add_argument("--timeout",type=float,default=3600.)
    parser.add_argument("--navigation-budget",type=int,default=110)
    parser.add_argument("--wrist-scans",action=argparse.BooleanOptionalAction,default=False,
                        help="Enable only to reproduce the archived arm-camera protocol")
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    root=repo.parent
    # Reject typos before creating an apparently scheduled/frozen batch.
    for path in args.tasks:
        task=json.loads(path.read_text())
        if "room_map_file" not in task:
            raise ValueError(f"Cross-room evaluation map missing: {path}")
        if not (path.parent/task["room_map_file"]).is_file():
            raise FileNotFoundError(path.parent/task["room_map_file"])
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    frozen=output/"frozen_source"
    hashes={}
    for path in sorted(list((repo/"src/dream").rglob("*.py"))+list((repo/"experiments").glob("*.py"))):
        destination=frozen/path.relative_to(repo)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        hashes[str(path.relative_to(repo))]=digest(destination)
        destination.chmod(0o444)
    tasks=[]
    for index,path in enumerate(args.tasks):
        task=json.loads(path.read_text())
        folder=output/"frozen_tasks"/f"{index+1:02d}"
        folder.mkdir(parents=True)
        shutil.copy2(path,folder/path.name)
        if "room_map_file" not in task:
            raise ValueError(f"Cross-room evaluation map missing: {path}")
        room=folder/task["room_map_file"]
        room.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path.parent/task["room_map_file"],room)
        tasks.append(dict(scene=task["scene"],file=str(folder/path.name),
                          sha256=digest(folder/path.name),room_map_sha256=digest(room)))
    protocol=dict(boundary="Development batch, all attempts retained; successful gallery entries are curated examples, not an unbiased rate.",
        source_sha256=hashes,tasks=tasks,created_unix_s=time.time(),navigation_version="astar",
        navigation_budget=args.navigation_budget,wrist_scans=args.wrist_scans,timeout_s=args.timeout,gpus=args.gpus)
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    jobs=queue.Queue()
    for index,task in enumerate(tasks):
        jobs.put((index,task))
    lock=threading.Lock()
    records=[]

    def source_check():
        return {key:dict(expected=expected,actual=digest(frozen/key) if (frozen/key).is_file() else None)
                for key,expected in hashes.items()
                if not (frozen/key).is_file() or digest(frozen/key)!=expected}

    def worker(gpu):
        while True:
            try:
                index,task=jobs.get_nowait()
            except queue.Empty:
                return
            name=f"{index+1:02d}_{task['scene']}"
            run=output/name
            before=source_check()
            (output/f"{name}.preexecution_source_check.json").write_text(json.dumps(dict(
                checked_unix_s=time.time(),passed=not before,mismatches=before),indent=2)+"\n")
            if before:
                raise RuntimeError(f"Frozen source changed before execution: {name}: {before}")
            command=[sys.executable,str(frozen/"experiments/maniskill_learned_dynamic.py"),
                "--output",str(run),"--task-json",task["file"],"--video","--navigate","--grasp","--deliver",
                "--navigation-version","astar","--navigation-budget",str(args.navigation_budget),
                "--wrist-scans" if args.wrist_scans else "--no-wrist-scans"]
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu,MS_ASSET_DIR=str(root/".maniskill_assets"),
                HF_HUB_CACHE=str(root/".dream_model_cache"),HF_HUB_OFFLINE="1",
                VK_ICD_FILENAMES=os.environ.get("VK_ICD_FILENAMES","/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"),
                OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
            record=dict(name=name,scene=task["scene"],started_unix_s=time.time(),gpu=gpu,command=command)
            print(json.dumps(dict(event="start",**record)),flush=True)
            with (output/f"{name}.stdout.log").open("w") as stream:
                process=subprocess.Popen(command,cwd=frozen,env=env,stdout=stream,stderr=subprocess.STDOUT)
                try:
                    process.wait(timeout=args.timeout)
                    record["status"]="completed" if process.returncode==0 else "execution_failure"
                except subprocess.TimeoutExpired:
                    record["status"]="timeout"
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=15)
                record["returncode"]=process.returncode
            if (run/"result.json").exists():
                r=json.loads((run/"result.json").read_text())
                record["protocol_success"]=bool(r.get("evaluator_protocol_success",False))
                record["cross_room_success"]=bool(r.get("evaluator_cross_room_success",False))
            record["finished_unix_s"]=time.time()
            after=source_check()
            record["source_unchanged"]=not after
            (output/f"{name}.postexecution_source_check.json").write_text(json.dumps(dict(
                checked_unix_s=time.time(),passed=not after,mismatches=after),indent=2)+"\n")
            with lock:
                records.append(record)
                with (output/"attempts.jsonl").open("a") as stream:
                    stream.write(json.dumps(record)+"\n")
            print(json.dumps(dict(event="finish",**record)),flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker,args.gpus))
    (output/"batch_result.json").write_text(json.dumps(dict(attempts=records),indent=2)+"\n")


if __name__=="__main__":
    main()
