#!/usr/bin/env python3
"""Execute instruction tasks from an isolated read-only source snapshot.

Development batches and the prespecified ten-house paired study are explicitly
different modes. All planned attempts, including execution failures, are kept.
Passing task scoring alone never marks a video or research release ready.
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


def validate_design(tasks,seeds,variants,benchmark):
    if len(set(variants))!=len(variants) or not set(variants)<={"dynamic","static"}:
        raise ValueError("Duplicate or invalid variants")
    if seeds is not None and len(set(seeds))!=len(seeds):
        raise ValueError("Duplicate start seeds")
    if benchmark:
        if len(tasks)!=10 or len({t["scene"] for t in tasks})!=10:
            raise ValueError("The paired study requires exactly ten distinct houses")
        if seeds is None or len(seeds)!=3 or set(variants)!={"dynamic","static"}:
            raise ValueError("The paired study requires three seeds and both memory variants")
        if len({t["recipe"]["id"] for t in tasks})<5:
            raise ValueError("The paired study requires at least five task recipes")
        if any(t.get("maximum_sim_seconds")!=1200 for t in tasks):
            raise ValueError("The paired study uses the declared 1,200 s simulation budget")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+")
    p.add_argument("--variants",nargs="+",choices=("dynamic","static"),default=["dynamic"])
    p.add_argument("--benchmark",action="store_true")
    p.add_argument("--gpus",nargs="+",default=["0","1"],help="One entry per worker; repeats allow multiple workers per GPU")
    p.add_argument("--wall-timeout",type=float,default=7200.)
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=repo.parent
    task_data=[json.loads(path.read_text()) for path in args.tasks]
    validate_design(task_data,args.seeds,args.variants,args.benchmark)
    for path,task in zip(args.tasks,task_data):
        if not (path.parent/task["room_map_file"]).is_file():
            raise FileNotFoundError(path.parent/task["room_map_file"])
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    workspace=output/"frozen_workspace";frozen=workspace/"DREAM_code"
    source_hashes={}
    sources=sorted([*(repo/"experiments").glob("*.py"),*(repo/"src/dream").rglob("*.py")])
    for source in sources:
        relative=source.relative_to(repo);destination=frozen/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination);destination.chmod(0o444)
        source_hashes[str(relative)]=digest(destination)
    # Only read-only input caches are shared. Policy/source imports resolve
    # inside the isolated DREAM_code tree, not the mutable working checkout.
    for name in (".maniskill_assets",".dream_model_cache"):
        (workspace/name).symlink_to(root/name,target_is_directory=True)
    jobs=[]
    for index,(path,task) in enumerate(zip(args.tasks,task_data)):
        for seed in args.seeds if args.seeds is not None else [task["seed"]]:
            folder=output/"frozen_tasks"/f"{index+1:02d}_{task['scene']}_seed{seed}"
            folder.mkdir(parents=True)
            task=dict(task,seed=seed)
            # The runner resamples from the valid cells with this seed. A
            # preview belonging to a different seed must not be carried over.
            task.pop("initial_sample_preview",None)
            room=folder/"evaluator_room_map.npz"
            shutil.copy2(path.parent/task["room_map_file"],room)
            task["room_map_file"]=room.name
            task_path=folder/"task.json"
            task_path.write_text(json.dumps(task,indent=2)+"\n")
            room.chmod(0o444);task_path.chmod(0o444)
            for variant in args.variants:
                jobs.append(dict(name=f"{index+1:02d}_{task['scene']}_seed{seed}_{variant}",
                    scene=task["scene"],seed=seed,recipe=task["recipe"]["id"],variant=variant,
                    task=str(task_path.relative_to(output)),task_sha256=digest(task_path),
                    room_map_sha256=digest(room)))
    protocol=dict(schema_version=1,mode="prespecified_60_attempt_paired_study" if args.benchmark else "development",
        created_unix_s=time.time(),planned_attempts=len(jobs),attempts=jobs,source_sha256=source_hashes,
        seeds=args.seeds,variants=args.variants,gpu_workers=args.gpus,wall_timeout_s=args.wall_timeout,
        simulation_budget_s=1200,threshold=.15,navigation_budget=110,
        boundary="All planned outcomes retained. Curated videos have a separate denominator. Physical replay is not another policy trial. No inferential claim follows from this manifest alone.")
    manifest=output/"protocol.json";manifest.write_text(json.dumps(protocol,indent=2)+"\n");manifest.chmod(0o444)
    (output/"protocol.sha256").write_text(digest(manifest)+"\n")
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    records=[];lock=threading.Lock()

    def check_source():
        return all((frozen/key).is_file() and digest(frozen/key)==expected for key,expected in source_hashes.items())

    def worker(gpu):
        while True:
            try:job=pending.get_nowait()
            except queue.Empty:return
            run=output/job["name"];task=output/job["task"]
            record=dict(**job,gpu=gpu,started_unix_s=time.time(),protocol_success=False,release_ready=False)
            source_before=check_source()
            task_before=digest(task)==job["task_sha256"] and digest(task.parent/"evaluator_room_map.npz")==job["room_map_sha256"]
            record.update(source_verified_before=source_before,task_verified_before=task_before)
            try:
                if not(source_before and task_before):raise RuntimeError("Frozen input hash mismatch")
                command=[sys.executable,str(frozen/"experiments/run_instruction_task.py"),"--task-json",str(task),
                    "--output",str(run),"--variant",job["variant"],"--video"]
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,
                    MS_ASSET_DIR=str(root/".maniskill_assets"),HF_HUB_CACHE=str(root/".dream_model_cache"),
                    HF_HUB_OFFLINE="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
                record["command"]=command
                print(json.dumps(dict(event="started",**record)),flush=True)
                with (output/(job["name"]+".log")).open("x") as stream:
                    process=subprocess.Popen(command,cwd=frozen,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    try:
                        process.wait(timeout=args.wall_timeout)
                        record["status"]="completed" if process.returncode==0 else "execution_failure"
                    except subprocess.TimeoutExpired:
                        record["status"]="wall_timeout";process.send_signal(signal.SIGINT)
                        try:process.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            try:process.wait(timeout=15)
                            except subprocess.TimeoutExpired:process.kill();process.wait()
                    record["returncode"]=process.returncode
                if (run/"result.json").is_file():
                    result=json.loads((run/"result.json").read_text())
                    record.update(protocol_success=bool(result.get("evaluator_protocol_success")),
                        result_sha256=digest(run/"result.json"),criteria=result.get("criteria",{}))
            except Exception as error:
                record.update(status="orchestration_failure",error=repr(error))
            record.update(finished_unix_s=time.time(),source_verified_after=check_source(),
                task_verified_after=digest(task)==job["task_sha256"])
            with lock:
                records.append(record)
                with (output/"attempts.jsonl").open("a") as stream:stream.write(json.dumps(record)+"\n")
                (output/"progress.json").write_text(json.dumps(dict(planned=len(jobs),completed=len(records),
                    protocol_successes=sum(r["protocol_success"] for r in records),release_ready=False),indent=2)+"\n")
            print(json.dumps(dict(event="finished",**record)),flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:list(pool.map(worker,args.gpus))
    (output/"batch_result.json").write_text(json.dumps(dict(protocol_sha256=digest(manifest),
        all_planned_attempts_recorded=len(records)==len(jobs),attempts=records,release_ready=False),indent=2)+"\n")


if __name__=="__main__":main()
