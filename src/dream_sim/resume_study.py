"""Resume infrastructure-interrupted tasks under a frozen study protocol.

Completed task failures are final outcomes too and are never retried. Previous
execution records and incomplete recordings are retained separately. Each GPU
has one worker; completed raw recordings are verified on durable storage before
that worker starts another task.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dream_sim.io import atomic_json, digest
from dream_sim.limited_worker import ALLOWED_GPUS, worker_environment


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cancelled(run: Path, durable: Path) -> bool:
    return any((root / "CANCELLED_BY_USER.json").exists() for root in (run, durable))


def completed_records(run: Path, protocol: dict, records: list[dict]) -> dict[str, dict]:
    by_name = {row["name"]: row for row in records}
    completed = {}
    for job in protocol["attempts"]:
        result = run / job["name"] / "result.json"
        if not result.is_file():
            continue
        row = by_name.get(job["name"])
        if not row or row.get("result_sha256") != digest(result):
            raise ValueError(f"Unverified existing result: {job['name']}")
        completed[job["name"]] = row
    return completed


def scheduled_attempts(
    protocol: dict, completed: dict, priority: list[str], maximum: int | None
) -> list[dict]:
    """Order a bounded stage without changing the cohort or retrying outcomes."""
    names = {job["name"] for job in protocol["attempts"]}
    if len(priority) != len(set(priority)) or not set(priority) <= names:
        raise ValueError("Priority attempts must be unique names in the frozen protocol")
    if maximum is not None and maximum < 1:
        raise ValueError("Maximum new attempts must be positive")
    rank = {name: index for index, name in enumerate(priority)}
    pending = [job for job in protocol["attempts"] if job["name"] not in completed]
    pending.sort(key=lambda job: rank.get(job["name"], len(priority)))
    return pending if maximum is None else pending[:maximum]


def qualification_upper_bound(planned: int, qualifications: dict) -> int:
    """Unknown outcomes remain possible successes; only final audits exclude them."""
    return planned - sum(row.get("qualified") is False for row in qualifications.values())


def archive_recording(run: Path, durable: Path, name: str) -> dict:
    source = run / name
    summary = durable / name
    summary.mkdir(exist_ok=True)
    receipt = summary / "recording_archive.json"
    if receipt.exists():
        raise FileExistsError(f"Recording already archived: {name}")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    hashes = {p.relative_to(source).as_posix(): digest(p) for p in files}
    destination = durable / "raw" / (name + ".tar")
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".tar.partial")
    with tarfile.open(temporary, "w", copybufsize=1024 * 1024) as archive:
        archive.add(source, arcname=name)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    with tarfile.open(temporary, "r") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        if len(members) != len(hashes):
            raise ValueError(f"Archive member count mismatch: {name}")
        for member in members:
            with archive.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != hashes[Path(member.name).relative_to(name).as_posix()]:
                raise ValueError(f"Archive content mismatch: {member.name}")
    temporary.replace(destination)
    for p in source.iterdir():
        if p.is_file() and p.suffix in (".json", ".jsonl"):
            shutil.copy2(p, summary / p.name)
    record = dict(
        name=name,
        archive=str(destination.relative_to(durable)),
        archive_bytes=destination.stat().st_size,
        archive_sha256=digest(destination),
        member_sha256=hashes,
        all_members_verified=True,
        recording_directory_present=True,
    )
    atomic_json(receipt, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--durable", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", choices=ALLOWED_GPUS, default=list(ALLOWED_GPUS[:2]))
    parser.add_argument(
        "--priority-attempts",
        nargs="*",
        default=[],
        help="Run these frozen attempt names first; retain the entire cohort denominator",
    )
    parser.add_argument(
        "--max-new-attempts",
        type=int,
        help="Finish a bounded stage, then leave remaining attempts available for resume",
    )
    parser.add_argument(
        "--minimum-qualified-total",
        type=int,
        help="Stop launching when final failures make this target impossible; finish active work",
    )
    parser.add_argument("--minimum-free-gib", type=int, default=128)
    parser.add_argument(
        "--audit-and-prune",
        action="store_true",
        help="Serially audit task successes, then remove verified temporary camera files",
    )
    parser.add_argument(
        "--allow-ram-recordings",
        action="store_true",
        help="Explicitly opt into temporary RAM recordings instead of disk storage",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("At most one inference worker per physical GPU")
    if len(args.gpus) > 2:
        parser.error("The current memory budget allows at most two concurrent workers")
    run, durable = args.run.resolve(), args.durable.resolve()
    if args.execute and run.is_relative_to("/dev/shm") and not args.allow_ram_recordings:
        parser.error(
            "Use a disk-backed recording directory; RAM recordings need --allow-ram-recordings"
        )
    if cancelled(run, durable):
        raise RuntimeError(
            "This study was cancelled; prepare a separate frozen study to run repaired code"
        )
    protocol = json.loads((run / "protocol.json").read_text())
    if args.minimum_qualified_total is not None and not 1 <= args.minimum_qualified_total <= len(
        protocol["attempts"]
    ):
        parser.error("Qualified target must be within the frozen cohort size")
    records = read_records(run / "attempts.jsonl")
    completed = completed_records(run, protocol, records)
    pending = scheduled_attempts(protocol, completed, args.priority_attempts, args.max_new_attempts)
    source = run / "frozen_workspace/DREAM_code"

    def verify():
        for name, expected in protocol["source_sha256"].items():
            if digest(source / name) != expected:
                raise ValueError(f"Frozen controller changed: {name}")

    verify()
    plan = dict(
        planned=len(protocol["attempts"]),
        completed=len(completed),
        remaining=[job["name"] for job in protocol["attempts"] if job["name"] not in completed],
        scheduled_this_stage=[job["name"] for job in pending],
        priority_attempts=args.priority_attempts,
        max_new_attempts=args.max_new_attempts,
        gpus=args.gpus,
        minimum_qualified_total=args.minimum_qualified_total,
        max_inference_workers=len(args.gpus),
        minimum_free_gib=args.minimum_free_gib,
        controller_sha256=digest(run / "protocol.json"),
        adapter_sha256=digest(Path(__file__).with_name("limited_worker.py")),
        scheduler_sha256=digest(Path(__file__)),
        runtime_dependencies_sha256={
            name: digest(Path(__file__).with_name(name))
            for name in [
                "worker_limits.py",
                "audit_recording.py",
                "prune_recordings.py",
                "fold_review.py",
            ]
        },
        audit_and_prune=args.audit_and_prune,
        changes="Bounded workers; frozen source and tasks verified before every attempt",
        created_unix_s=time.time(),
    )
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    lockfile = (durable / "recovery.lock").open("a")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    recovery = durable / ("recovery_" + time.strftime("%Y%m%d_%H%M%S"))
    recovery.mkdir()
    atomic_json(recovery / "plan.json", plan)
    atomic_json(recovery / "process.json", dict(pid=os.getpid(), started_unix_s=time.time()))
    for name in ("attempts.jsonl", "progress.json", "protocol.json", "protocol.sha256"):
        shutil.copy2(run / name, recovery / ("before_" + name))
    interrupted = run / recovery.name
    interrupted.mkdir()
    for job in pending:
        previous = run / job["name"]
        if previous.exists():
            previous.rename(interrupted / job["name"])
    jobs = queue.Queue()
    for job in pending:
        jobs.put(job)
    mutex = threading.Lock()
    archive_slots = threading.Semaphore(1)
    stop_launching = threading.Event()
    active = {}
    errors = []
    qualifications = {}
    archived = {name for name in completed if (durable / name / "recording_archive.json").is_file()}
    environment = dict(MS_ASSET_DIR=protocol["asset_root"], HF_HUB_CACHE=protocol["model_cache"])

    def target_unreachable():
        return (
            args.minimum_qualified_total is not None
            and qualification_upper_bound(len(protocol["attempts"]), qualifications)
            < args.minimum_qualified_total
        )

    def progress():
        ordered = [completed[j["name"]] for j in protocol["attempts"] if j["name"] in completed]
        state = dict(
            planned=len(protocol["attempts"]),
            completed=len(completed),
            protocol_successes=sum(bool(r.get("protocol_success")) for r in ordered),
            task_successes=sum(bool(r.get("task_success")) for r in ordered),
            archived=len(archived),
            active=active,
            errors=errors,
            audited=len(qualifications),
            qualified_successes=sum(r["qualified"] for r in qualifications.values()),
            maximum_possible_qualified=qualification_upper_bound(
                len(protocol["attempts"]), qualifications
            ),
            qualification_target_unreachable=target_unreachable(),
            gpu_workers=args.gpus,
            release_ready=False,
            recovery_directory=str(recovery),
            updated_unix_s=time.time(),
        )
        for root in (run, durable):
            temporary = root / "attempts.jsonl.tmp"
            temporary.write_text("".join(json.dumps(r) + "\n" for r in ordered))
            temporary.replace(root / "attempts.jsonl")
            atomic_json(root / "progress.json", state)
        atomic_json(recovery / "progress.json", state)

    def worker(gpu):
        while not stop_launching.is_set():
            if cancelled(run, durable):
                stop_launching.set()
                return
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            name = job["name"]
            try:
                # A full RAM filesystem produces SIGBUS in native simulators.
                # Wait before allocating another scene, rather than filling it.
                while shutil.disk_usage(run).free < args.minimum_free_gib * 1024**3:
                    if stop_launching.is_set() or cancelled(run, durable):
                        stop_launching.set()
                        return
                    with mutex:
                        active[gpu] = dict(name=name, status="waiting_for_storage_headroom")
                        progress()
                    time.sleep(15)
                task = run / job["task"]
                if digest(task) != job["task_sha256"]:
                    raise ValueError("Frozen task changed")
                task_data = json.loads(task.read_text())
                if digest(task.parent / task_data["room_map_file"]) != job["room_map_sha256"]:
                    raise ValueError("Frozen room map changed")
                verify()
                command = [
                    sys.executable,
                    "-m",
                    "dream_sim.limited_worker",
                    "--gpu",
                    gpu,
                    "--entrypoint",
                    str(source / "experiments/run_instruction_task.py"),
                    "--wait-for-slot-seconds",
                    str(protocol["wall_timeout_s"]),
                    "--receipt",
                    str(recovery / (name + ".runtime.json")),
                    "--task-json",
                    str(task),
                    "--output",
                    str(run / name),
                    "--variant",
                    job["variant"],
                    "--video",
                    "--heading-navigation",
                ]
                env = worker_environment(gpu)
                env.update(environment)
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
                record = dict(
                    job,
                    gpu=gpu,
                    started_unix_s=time.time(),
                    protocol_success=False,
                    task_success=False,
                    release_ready=False,
                    source_verified_before=True,
                    task_verified_before=True,
                    command=command,
                    infrastructure_retry=any(row["name"] == name for row in records),
                    previous_execution_records=str(recovery / "before_attempts.jsonl"),
                )
                with (recovery / (name + ".log")).open("x") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=source,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        stdin=subprocess.DEVNULL,
                    )
                    with mutex:
                        active[gpu] = dict(
                            name=name,
                            pid=process.pid,
                            status="inference",
                            started_unix_s=record["started_unix_s"],
                        )
                        progress()
                    try:
                        returncode = process.wait(timeout=protocol["wall_timeout_s"])
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            returncode = process.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            returncode = process.wait()
                result_path = run / name / "result.json"
                record.update(
                    returncode=returncode,
                    finished_unix_s=time.time(),
                    status="completed"
                    if result_path.exists() and returncode == 0
                    else "execution_failure",
                )
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    record.update(
                        task_success=result.get("evaluator_task_success", False),
                        protocol_success=result.get("evaluator_protocol_success", False),
                        result_sha256=digest(result_path),
                        criteria=result.get("criteria", {}),
                    )
                else:
                    with (recovery / "bootstrap_failures.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    raise RuntimeError(
                        f"Worker exited before a task result was written (exit {returncode})"
                    )
                verify()
                record.update(
                    source_verified_after=True,
                    task_verified_after=digest(task) == job["task_sha256"],
                )
                with mutex:
                    completed[name] = record
                    active[gpu] = dict(name=name, status="archiving")
                    with (recovery / "attempts.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    progress()
                with archive_slots:
                    archive_recording(run, durable, name)
                    if args.audit_and_prune:
                        from dream_sim.audit_recording import audit_recording
                        from dream_sim.prune_recordings import prune_recording

                        with mutex:
                            active[gpu] = dict(name=name, status="auditing_and_pruning")
                            progress()
                        qualification = audit_recording(run, durable, record, gpu)
                        prune_recording(run, durable, name, execute=True)
                        with mutex:
                            qualifications[name] = qualification
                            if target_unreachable():
                                stop_launching.set()
                with mutex:
                    archived.add(name)
                    active.pop(gpu, None)
                    progress()
                print(
                    json.dumps(
                        dict(event="archived", name=name, task_success=record["task_success"])
                    ),
                    flush=True,
                )
            except Exception as error:
                stop_launching.set()
                with mutex:
                    errors.append(dict(name=name, gpu=gpu, error=repr(error)))
                    active.pop(gpu, None)
                    progress()
                # Preserve the partial attempt for inspection; do not retry it in a loop.
                print(
                    json.dumps(dict(event="worker_error", name=name, error=repr(error))), flush=True
                )
            finally:
                jobs.task_done()

    with mutex:
        progress()
    # A task result is final even if the process stopped during archival or
    # review. Resume that remaining work without another policy execution.
    for name, record in completed.items():
        if cancelled(run, durable):
            stop_launching.set()
            break
        if name not in archived:
            archive_recording(run, durable, name)
            archived.add(name)
        if args.audit_and_prune:
            from dream_sim.audit_recording import audit_recording
            from dream_sim.prune_recordings import prune_recording

            qualifications[name] = audit_recording(run, durable, record, args.gpus[0])
            if target_unreachable():
                stop_launching.set()
            if not (durable / name / "recording_pruned.json").exists():
                prune_recording(run, durable, name, execute=True)
        with mutex:
            progress()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, args.gpus))
    ordered = [completed[j["name"]] for j in protocol["attempts"] if j["name"] in completed]
    final = dict(
        protocol_sha256=digest(run / "protocol.json"),
        attempts=ordered,
        all_planned_attempts_recorded=len(ordered) == len(protocol["attempts"]),
        all_recordings_archived=len(archived) == len(protocol["attempts"]),
        audit_and_prune=args.audit_and_prune,
        qualifications=qualifications,
        qualified_successes=sum(r["qualified"] for r in qualifications.values()),
        maximum_possible_qualified=qualification_upper_bound(
            len(protocol["attempts"]), qualifications
        ),
        qualification_target_unreachable=target_unreachable(),
        cancelled=cancelled(run, durable),
        infrastructure_retry_records=str(recovery / "before_attempts.jsonl"),
        release_ready=False,
        errors=errors,
    )
    for root in (run, durable):
        atomic_json(root / "batch_result.json", final)
    atomic_json(recovery / "execution_complete.json", final)


if __name__ == "__main__":
    main()
