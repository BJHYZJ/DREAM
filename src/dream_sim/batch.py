"""Run every frozen scene once with a robot action budget and a separate watchdog.

The batch records every task outcome, including failures, with immutable task
and controller inputs. Physics replay evaluates successful executions separately.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dream_sim.io import atomic_json, digest
from dream_sim.limited_worker import ALLOWED_GPUS, worker_environment


def worker_cpu_command(command, gpu, config):
    """Let a child reach its configured CPU pool before it claims four cores."""
    cpus = sorted(
        {
            cpu
            for key, values in config.get("cpu_affinity_by_worker", {}).items()
            if key.startswith(str(gpu) + ":")
            for cpu in values
        }
    )
    if not cpus:
        return list(command)
    return ["taskset", "--cpu-list", ",".join(str(cpu) for cpu in cpus), *command]


def initial_worker_assignments(jobs, gpus, workers_per_gpu):
    """Reserve initial jobs across GPUs before concurrent workers can take them."""
    assignments = []
    for slot in range(workers_per_gpu):
        for gpu in gpus:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return assignments
            assignments.append((f"{gpu}:{slot}", gpu, job))
    return assignments


def wait_for_execution(process, started_path, limit, *, grace=30, poll_seconds=0.5):
    """The execution cap begins at slot acquisition, excluding the queue."""
    started = None
    timed_out = False
    sent = None
    while process.poll() is None:
        if started is None and started_path.exists():
            started = json.loads(started_path.read_text())["started_unix_s"]
        now = time.time()
        if started is not None and now - started >= limit:
            timed_out = True
            if sent is None:
                os.killpg(process.pid, signal.SIGINT)
                sent = now
            elif now - sent >= grace:
                os.killpg(process.pid, signal.SIGKILL)
        time.sleep(poll_seconds)
    if started is None and started_path.exists():
        started = json.loads(started_path.read_text())["started_unix_s"]
    finished_path = started_path.with_name(
        started_path.name.replace(".started.json", ".finished.json")
    )
    finished = (
        json.loads(finished_path.read_text())["finished_unix_s"]
        if finished_path.exists()
        else time.time()
    )
    elapsed = None if started is None else finished - started
    timed_out = timed_out or (elapsed is not None and elapsed >= limit)
    return process.returncode, timed_out, elapsed


def classify_failure(result, timeout, stage, execution_failure=False):
    if timeout:
        return "runtime_watchdog_timeout"
    if result.get("robot_action_timeout"):
        return f"{stage or 'task'}_action_timeout"
    if execution_failure:
        return "infrastructure_failure"
    if result.get("evaluator_task_success"):
        return None
    text = str(result.get("error", "")).lower()
    if any(word in text for word in ("empty", "lost", "grip")):
        return "grasp_or_payload_loss"
    if result.get("pickup_held") is False or stage == "grasp":
        return "grasp_failure"
    if result.get("placement_approached") and not result.get("placement_motion_completed"):
        return "placement_failure"
    return f"{stage or 'task'}_failure"


def action_time_success(result, returncode, watchdog_timeout, robot_limit):
    """A server watchdog is a runtime failure; robot duration is measured independently."""
    seconds = result.get("robot_action_seconds")
    return bool(
        returncode == 0
        and not watchdog_timeout
        and result.get("evaluator_task_success")
        and not result.get("robot_action_timeout")
        and isinstance(seconds, (int, float))
        and 0 <= seconds <= robot_limit
        and result.get("robot_action_limit_seconds") == robot_limit
    )


def latest_stage(folder):
    path = folder / "evaluator_trajectory.jsonl"
    if not path.exists():
        return "initialization"
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16384))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line).get("task_stage", "unknown")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gpus", nargs="+", choices=ALLOWED_GPUS, default=list(ALLOWED_GPUS))
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument(
        "--time-limit-seconds",
        type=float,
        default=2700,
        help="Separate server watchdog, including loading and saving",
    )
    p.add_argument(
        "--robot-time-limit-seconds",
        type=float,
        default=900,
        help="Executed robot controls divided by actual control frequency",
    )
    p.add_argument("--raster-threads", type=int, choices=(2, 4), default=2)
    p.add_argument(
        "--wait-for-slot-seconds",
        type=float,
        default=120,
        help="Wait for a shared worker slot; excluded from execution and robot clocks",
    )
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    run = args.run.resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    if len(set(args.gpus)) != len(args.gpus) or args.workers_per_gpu not in (1, 2, 4):
        p.error("Invalid GPU slots")
    if len(args.gpus) * args.workers_per_gpu > 8 or not 0 < args.time_limit_seconds <= 3600:
        p.error("Batch exceeds eight workers or the server watchdog cap")
    if not 0 < args.robot_time_limit_seconds <= 900:
        p.error("Robot action duration must be positive and at most 15 minutes")
    if not 0 <= args.wait_for_slot_seconds <= 3600:
        p.error("Worker slot wait must be between zero and one hour")
    if protocol["simulation_budget_s"] != args.robot_time_limit_seconds:
        p.error("Frozen robot duration differs from requested duration")
    if protocol["wall_timeout_s"] != args.time_limit_seconds:
        p.error("Frozen watchdog differs from requested watchdog")
    source = run / "frozen_workspace/DREAM_code"

    def verify():
        for name, expected in protocol["source_sha256"].items():
            if digest(source / name) != expected:
                raise ValueError(f"Frozen source changed: {name}")

    verify()
    config_path = Path(__file__).resolve().parents[2] / ".runtime/worker_resource_limits.json"
    config = json.loads(config_path.read_text())
    assert config["workers_per_gpu"] == args.workers_per_gpu
    plan = dict(
        kind="residential_batch",
        planned_attempts=len(protocol["attempts"]),
        protocol_sha256=digest(run / "protocol.json"),
        execution_wall_limit_seconds=args.time_limit_seconds,
        robot_action_limit_seconds=args.robot_time_limit_seconds,
        worker_slot_wait_seconds=args.wait_for_slot_seconds,
        deadline_starts="Robot clock: executed controls; separate server watchdog: worker slot acquired",
        cleanup_grace_seconds=30,
        gpus=args.gpus,
        workers_per_gpu=args.workers_per_gpu,
        maximum_concurrent_tasks=len(args.gpus) * args.workers_per_gpu,
        cpu_affinity_by_worker=config["cpu_affinity_by_worker"],
        combined_ram_limit_gib=config["total_memory_gib"],
        worker_cpu_launch="Configured per-GPU CPU pool before the existing per-slot CPU guard",
        initial_gpu_assignment="Round robin across requested GPUs; later jobs use the first available worker",
        worker_ram_limit_gib=config["worker_memory_gib"],
        worker_nice=10,
        per_task_video=False,
        raster_threads=args.raster_threads,
        runtime_source_sha256={
            name: digest(Path(__file__).with_name(name))
            for name in ["batch.py", "worker_limits.py", "limited_worker.py"]
        },
        resource_config_sha256=digest(config_path),
        created_unix_s=time.time(),
        scoring="Task evaluator success AND recorded robot action duration <=15minutes. Loading/inference/saving waits do not advance the robot clock. Runtime watchdog aborts are reported separately. Every declared task contributes to the result.",
        qualification="Task completion is provisional until the independent physics and recording checks pass.",
    )
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    lock = (run / "screen.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "screen_protocol.json").exists():
        raise FileExistsError(
            "Use the existing screen records; never silently rerun completed failures"
        )
    atomic_json(run / "screen_protocol.json", plan)
    runtime = run / "screen_runtime"
    runtime.mkdir()
    results = run / "screen_results"
    results.mkdir()
    jobs = queue.Queue()
    for job in protocol["attempts"]:
        jobs.put(job)
    mutex = threading.Lock()
    completed = {}
    active = {}
    fatal = []
    started = time.time()

    def progress():
        ordered = [completed[j["name"]] for j in protocol["attempts"] if j["name"] in completed]
        categories = Counter(r["failure_category"] for r in ordered if r["failure_category"])
        summary = dict(
            planned=len(protocol["attempts"]),
            completed=len(ordered),
            task_successes=sum(r["task_success"] for r in ordered),
            failures=sum(not r["task_success"] for r in ordered),
            timeouts=sum(r["wall_timeout"] for r in ordered),
            qualified_successes=0,
            qualification_pending=True,
            robot_action_timeouts=sum(bool(r.get("robot_action_timeout")) for r in ordered),
            active=dict(active),
            failure_categories=dict(categories),
            fatal_errors=fatal,
            wall_limit_seconds=args.time_limit_seconds,
            parallel_tasks=plan["maximum_concurrent_tasks"],
            robot_action_limit_seconds=args.robot_time_limit_seconds,
            elapsed_minutes=(time.time() - started) / 60,
            updated_unix_s=time.time(),
        )
        atomic_json(run / "screen_progress.json", summary)
        atomic_json(
            run / "screen_failure_summary.json", dict(counts=dict(categories), attempts=ordered)
        )
        return summary

    def worker(worker_id, gpu, initial_job):
        job = initial_job
        while True:
            if job is None:
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    return
            name = job["name"]
            folder = run / name
            receipt = runtime / (name + ".runtime.json")
            row = dict(job, worker_id=worker_id, gpu=gpu, queued_unix_s=time.time())
            try:
                if shutil.disk_usage(run).free < 128 * 1024**3:
                    raise RuntimeError("Insufficient disk headroom")
                verify()
                task = run / job["task"]
                task_data = json.loads(task.read_text())
                if (
                    digest(task) != job["task_sha256"]
                    or digest(task.parent / task_data["room_map_file"]) != job["room_map_sha256"]
                ):
                    raise ValueError("Frozen task or room map changed")
                cmd = [
                    sys.executable,
                    "-m",
                    "dream_sim.limited_worker",
                    "--gpu",
                    gpu,
                    "--entrypoint",
                    str(source / "experiments/run_instruction_task.py"),
                    "--raster-threads",
                    str(args.raster_threads),
                    "--wait-for-slot-seconds",
                    str(args.wait_for_slot_seconds),
                    "--receipt",
                    str(receipt),
                    "--task-json",
                    str(task),
                    "--output",
                    str(folder),
                    "--variant",
                    job["variant"],
                    "--heading-navigation",
                ]
                cmd.extend(["--simulation-time-limit-seconds", str(args.robot_time_limit_seconds)])
                env = worker_environment(gpu, raster_threads=args.raster_threads)
                env.update(
                    PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                    MS_ASSET_DIR=protocol["asset_root"],
                    HF_HUB_CACHE=protocol["model_cache"],
                )
                with (runtime / (name + ".log")).open("x") as log:
                    process = subprocess.Popen(
                        worker_cpu_command(cmd, gpu, config),
                        cwd=source,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    with mutex:
                        active[worker_id] = dict(
                            name=name, pid=process.pid, gpu=gpu, queued_unix_s=row["queued_unix_s"]
                        )
                        progress()
                    code, timeout, elapsed = wait_for_execution(
                        process, receipt.with_suffix(".started.json"), args.time_limit_seconds
                    )
                result_path = folder / "result.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                verify()
                stage = latest_stage(folder)
                row.update(
                    finished_unix_s=time.time(),
                    execution_seconds=elapsed,
                    returncode=code,
                    result_sha256=digest(result_path) if result_path.exists() else None,
                    wall_timeout=timeout,
                    task_success=action_time_success(
                        result, code, timeout, args.robot_time_limit_seconds
                    ),
                    robot_action_seconds=result.get("robot_action_seconds"),
                    robot_action_timeout=bool(result.get("robot_action_timeout")),
                    robot_action_limit_seconds=args.robot_time_limit_seconds,
                    evaluator_task_success=bool(result.get("evaluator_task_success")),
                    last_stage=stage,
                    failure_category=classify_failure(
                        result, timeout, stage, code != 0 and not timeout
                    ),
                    error=result.get("error"),
                    criteria=result.get("criteria", {}),
                    source_verified_after=True,
                    recording_directory=str(folder),
                    qualified=False,
                    qualification_pending=True,
                )
                if row["task_success"]:
                    row["failure_category"] = None
                if not row["task_success"] and row["failure_category"] is None:
                    row["failure_category"] = "execution_failure"
            except Exception as error:
                row.update(
                    finished_unix_s=time.time(),
                    wall_timeout=False,
                    task_success=False,
                    failure_category="infrastructure_failure",
                    error=repr(error),
                    qualified=False,
                )
            atomic_json(results / (name + ".json"), row)
            with mutex:
                completed[name] = row
                active.pop(worker_id, None)
                summary = progress()
            print(
                json.dumps(
                    dict(
                        event="attempt_finished",
                        name=name,
                        completed=summary["completed"],
                        successes=summary["task_successes"],
                        failure_category=row["failure_category"],
                    )
                ),
                flush=True,
            )
            jobs.task_done()
            job = None

    assignments = initial_worker_assignments(jobs, args.gpus, args.workers_per_gpu)
    with ThreadPoolExecutor(max_workers=max(1, len(assignments))) as pool:
        futures = [pool.submit(worker, *assignment) for assignment in assignments]
        for future in futures:
            future.result()
    summary = progress()
    summary["all_50_recorded"] = len(completed) == len(protocol["attempts"])
    atomic_json(run / "screen_complete.json", summary)
    print(json.dumps(dict(event="screen_complete", **summary)), flush=True)


if __name__ == "__main__":
    main()
