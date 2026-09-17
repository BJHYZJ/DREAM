"""Bind a completed parallel residential cohort to its frozen inputs."""

import hashlib
import json
from pathlib import Path

from dream_sim.batch import action_time_success
from dream_sim.sources import safe_member

RESIDENTIAL_MODES = {
    "residential50_easy_grasp_seed42_v1",
    "residential50_seed42_dynamic_v1",
    "residential50_seed42_diverse_v2",
}


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_completed_screen(run: Path):
    """Read every declared outcome; absent or changed evidence is an error."""
    run = run.resolve()
    if not (run / "screen_complete.json").exists():
        raise RuntimeError("The declared parallel experiment has not finished")
    protocol = read(run / "protocol.json")
    complete = read(run / "screen_complete.json")
    launch = read(run / "screen_protocol.json")
    digest = sha(run / "protocol.json")
    if (
        launch.get("protocol_sha256") != digest
        or (run / "protocol.sha256").read_text().strip() != digest
    ):
        raise ValueError("Protocol checksum mismatch")
    jobs = protocol.get("attempts", [])
    if (
        protocol.get("mode") not in RESIDENTIAL_MODES
        or protocol.get("planned_attempts") != 50
        or len(jobs) != 50
        or len({job["name"] for job in jobs}) != 50
        or len({job["scene"] for job in jobs}) != 50
        or any(job["seed"] != 42 or job["variant"] != "dynamic" for job in jobs)
    ):
        raise ValueError("Expected the complete declared seed-42 residential cohort")
    if (
        complete.get("all_50_recorded") is not True
        or complete.get("planned") != 50
        or complete.get("completed") != 50
        or complete.get("active")
        or complete.get("fatal_errors")
    ):
        raise ValueError("Incomplete parallel experiment")
    source = run / "frozen_workspace/DREAM_code"
    hashes = protocol.get("source_sha256", {})
    if not hashes or any(
        sha(source / safe_member(name)) != value for name, value in hashes.items()
    ):
        raise ValueError("Frozen controller source changed")
    files = {path.stem: path for path in (run / "screen_results").glob("*.json")}
    if set(files) != {job["name"] for job in jobs}:
        raise ValueError("Missing or unexpected parallel outcomes")
    outcomes = []
    for job in jobs:
        name = job["name"]
        if len(safe_member(name).parts) != 1:
            raise ValueError("Invalid attempt directory")
        record = read(files[name])
        for key in ("name", "scene", "seed", "variant", "task", "task_sha256", "room_map_sha256"):
            if record.get(key) != job.get(key):
                raise ValueError(f"{name}: recorded task differs from the protocol")
        task = run / safe_member(job["task"])
        if (
            sha(task) != job["task_sha256"]
            or sha(task.parent / "evaluator_room_map.npz") != job["room_map_sha256"]
        ):
            raise ValueError(f"{name}: task input changed")
        folder = run / name
        task_data = read(task)
        if (
            task_data.get("scene") != job["scene"]
            or task_data.get("seed") != job["seed"]
            or read(folder / "environment_task.json") != task_data
            or sha(folder / "evaluator_room_map.npz") != job["room_map_sha256"]
        ):
            raise ValueError(f"{name}: executed environment differs from the frozen task")
        result = read(folder / "result.json")
        if sha(folder / "result.json") != record.get("result_sha256"):
            raise ValueError(f"{name}: result checksum mismatch")
        if (
            record.get("source_verified_after") is not True
            or result.get("source_files_unchanged") is not True
        ):
            raise ValueError(f"{name}: source verification failed")
        for filename in ("source_hashes_before.json", "source_hashes_after.json"):
            if read(folder / filename) != hashes:
                raise ValueError(f"{name}: controller differs from the declared source")
        if (
            type(result.get("evaluator_task_success")) is not bool
            or type(record.get("task_success")) is not bool
        ):
            raise ValueError(f"{name}: missing task outcome")
        if (
            type(record.get("wall_timeout")) is not bool
            or type(record.get("returncode")) is not int
        ):
            raise ValueError(f"{name}: missing execution outcome")
        success = action_time_success(
            result,
            record.get("returncode"),
            record.get("wall_timeout"),
            protocol["simulation_budget_s"],
        )
        if success is not record["task_success"]:
            raise ValueError(f"{name}: task outcome disagrees with its execution limits")
        outcomes.append((job, record, result))
    if sum(record["task_success"] for _, record, _ in outcomes) != complete.get("task_successes"):
        raise ValueError("Completed-batch count differs from its task outcomes")
    return protocol, outcomes
