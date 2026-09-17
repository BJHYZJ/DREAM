"""Run a common controller over a declared set of houses, seeds, and memory variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from dream_sim.limited_worker import ALLOWED_GPUS
from dream_sim.sources import PROJECT, engine_root, safe_member, verify_public_configs


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def controller_overrides(
    directory: Path, base_source_id: str, *, _parents: frozenset[Path] = frozenset()
) -> dict[str, bytes]:
    directory = directory.resolve()
    if directory in _parents:
        raise ValueError("Controller inheritance contains a cycle")
    manifest = json.loads((directory / "controller.json").read_text())
    if manifest["base_study_source_id"] != base_source_id:
        raise ValueError("Controller revision requires a different base implementation")
    files = {}
    if parent_name := manifest.get("base_controller"):
        parent_relative = safe_member(parent_name)
        if len(parent_relative.parts) != 1:
            raise ValueError("Base controller must be a sibling directory")
        parent = directory.parent / parent_relative
        if parent.is_symlink() or parent.resolve().parent != directory.parent:
            raise ValueError("Base controller escapes the controller directory")
        files.update(controller_overrides(parent, base_source_id, _parents=_parents | {directory}))
    for name, digest in manifest["overrides"].items():
        relative = safe_member(name)
        if relative.parts[0] != "experiments" or relative.suffix != ".py":
            raise ValueError("Controller overrides must be experiment Python modules")
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Controller override escapes its directory")
        content = path.read_bytes()
        if sha(content) != digest:
            raise ValueError(f"Controller revision checksum mismatch: {name}")
        files[name] = content
    if not files:
        raise ValueError("Controller revision contains no implementation")
    return files


def materialize_controller(source: Path, destination: Path, overrides: dict[str, bytes]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    paths = [*(source / "experiments").glob("*.py"), *(source / "src/dream").rglob("*.py")]
    names = {path.relative_to(source).as_posix() for path in paths}
    for name in overrides:
        relative = safe_member(name)
        if relative.parts[0] != "experiments" or relative.suffix != ".py":
            raise ValueError("Controller overrides must be experiment Python modules")
    for path in paths:
        name = path.relative_to(source).as_posix()
        content = overrides.get(name)
        if content is None:
            content = path.read_bytes()
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        if sha(output.read_bytes()) != sha(content):
            raise OSError(f"Controller copy verification failed: {name}")
    for name in sorted(set(overrides) - names):
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(overrides[name])
        if sha(output.read_bytes()) != sha(overrides[name]):
            raise OSError(f"Controller copy verification failed: {name}")


def load_task_manifest(path: Path) -> tuple[dict, list[Path]]:
    """Validate an explicit cohort and bind every task to its room geometry."""
    path = path.resolve()
    manifest = json.loads(path.read_text())
    rows = manifest["tasks"]
    if not rows or manifest["planned_attempts"] != len(rows):
        raise ValueError("Manifest attempt count does not match its tasks")
    tasks = []
    houses = set()
    for row in rows:
        task_path = (path.parent / safe_member(row["file"])).resolve()
        if not task_path.is_relative_to(path.parent):
            raise ValueError("Task path escapes manifest directory")
        content = task_path.read_bytes()
        if sha(content) != row["sha256"]:
            raise ValueError("Task checksum mismatch")
        task = json.loads(content)
        room = (task_path.parent / safe_member(task["room_map_file"])).resolve()
        if not room.is_relative_to(path.parent) or sha(room.read_bytes()) != row["room_map_sha256"]:
            raise ValueError("Room map checksum mismatch")
        if task["scene"] != row["scene"] or task["seed"] != row["seed"]:
            raise ValueError("Task metadata differs from manifest")
        if row["scene"] in houses:
            raise ValueError("Expected one task per distinct house")
        if row["seed"] != manifest["seed"] or row["variant"] != "dynamic":
            raise ValueError("Explicit cohort requires a common seed and dynamic memory")
        houses.add(row["scene"])
        tasks.append(task_path)
    if manifest.get("variants") != ["dynamic"]:
        raise ValueError("Explicit cohort requires dynamic memory")
    return manifest, tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=[
            "baseline",
            "recovery",
            "recovery_v2",
            "recovery_v6",
            "compact_v1",
            "staged_return",
            "continuous_return",
        ],
        default="staged_return",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[f"{i:02d}" for i in range(1, 11)],
        default=[f"{i:02d}" for i in range(1, 11)],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument(
        "--variants", nargs="+", choices=["dynamic", "static"], default=["dynamic", "static"]
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="Checksum-locked cohort; supplies tasks, seed and memory variant",
    )
    parser.add_argument("--gpus", nargs="+", choices=ALLOWED_GPUS, default=list(ALLOWED_GPUS[:1]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, default=PROJECT / ".runtime/assets")
    parser.add_argument("--model-cache", type=Path, default=PROJECT / ".runtime/models")
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the locked cohort with a recorded robot action limit and a separate server watchdog",
    )
    parser.add_argument(
        "--robot-time-limit-seconds",
        type=int,
        default=900,
        help="Parallel task action budget, frozen before execution (default: 900; maximum: 1800)",
    )
    parser.add_argument(
        "--wall-timeout-seconds",
        type=int,
        default=2700,
        help="Parallel execution watchdog; 0 disables the server deadline (maximum: 5400)",
    )
    parser.add_argument(
        "--wait-for-slot-seconds",
        type=int,
        default=120,
        help="Parallel worker queue wait, excluded from robot and execution clocks (maximum: 3600)",
    )
    parser.add_argument(
        "--raster-threads",
        type=int,
        choices=(2, 4),
        default=2,
        help="CPU rendering threads per parallel batch worker; match the deployment CPU affinity",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        choices=(2, 4),
        help="Parallel batch concurrency; match the verified deployment resource profile",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 0 < args.robot_time_limit_seconds <= 1800:
        parser.error("Robot action duration must be positive and at most 30 minutes")
    if args.robot_time_limit_seconds != 900 and not args.parallel:
        parser.error("--robot-time-limit-seconds is a parallel batch setting")
    if not 0 <= args.wall_timeout_seconds <= 5400:
        parser.error("Server watchdog must be zero (disabled) or at most 90 minutes")
    if args.wall_timeout_seconds != 2700 and not args.parallel:
        parser.error("--wall-timeout-seconds is a parallel batch setting")
    if not 0 <= args.wait_for_slot_seconds <= 3600:
        parser.error("Worker slot wait must be between zero and one hour")
    if args.wait_for_slot_seconds != 120 and not args.parallel:
        parser.error("--wait-for-slot-seconds is a parallel batch setting")
    if args.workers_per_gpu is None:
        profile = PROJECT / ".runtime/worker_resource_limits.json"
        args.workers_per_gpu = (
            json.loads(profile.read_text()).get("workers_per_gpu", 2)
            if args.parallel and profile.exists()
            else 2
        )
        if args.workers_per_gpu not in (2, 4):
            parser.error("Parallel batch requires a two- or four-worker GPU deployment profile")
    if args.parallel and not args.task_manifest:
        parser.error("--parallel requires a checksum-locked --task-manifest")
    if args.raster_threads != 2 and not args.parallel:
        parser.error("--raster-threads is a parallel batch setting")
    if args.workers_per_gpu != 2 and not args.parallel:
        parser.error("--workers-per-gpu is a parallel batch setting")
    if args.task_manifest and any(
        option.split("=")[0] in {"--cases", "--seeds", "--variants"} for option in sys.argv[1:]
    ):
        parser.error("--task-manifest supplies cases, seeds and variants; do not override them")
    for name in ("cases", "seeds", "variants", "gpus"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate {name}")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory")

    engine = engine_root()
    sys.path.insert(0, str(engine / "experiments"))
    from run_instruction_profile import load_catalog, verify_case, verify_source

    verify_public_configs()
    root, catalog = load_catalog(engine / "experiments/repro_profiles/profiles.json")
    source_id = catalog["study"]["source_id"]
    source = verify_source(root, catalog["sources"][source_id])
    by_id = {case["id"]: case for case in catalog["cases"]}
    manifest = None
    if args.task_manifest:
        manifest, tasks = load_task_manifest(args.task_manifest)
        args.seeds = [manifest["seed"]]
        args.variants = manifest["variants"]
        args.cases = [row.get("id", row["scene"]) for row in manifest["tasks"]]
    else:
        tasks = [verify_case(root, by_id[key]) for key in args.cases]
    directory = PROJECT / "controllers" / args.controller
    if args.controller != "baseline" and not (directory / "controller.json").is_file():
        parser.error(f"Controller revision is not installed: {args.controller}")
    overrides = controller_overrides(directory, source_id) if args.controller != "baseline" else {}
    full_comparison = (
        manifest is None
        and len(tasks) == 10
        and len(args.seeds) == 3
        and set(args.variants) == {"dynamic", "static"}
    )
    print(
        json.dumps(
            {
                "controller": args.controller,
                "base_source_id": source_id,
                "source_overrides": {key: sha(value) for key, value in overrides.items()},
                "task_manifest_sha256": sha(args.task_manifest.read_bytes())
                if args.task_manifest
                else None,
                "cases": args.cases,
                "seeds": args.seeds,
                "variants": args.variants,
                "planned_attempts": len(tasks) * len(args.seeds) * len(args.variants),
                "parallel": args.parallel,
                "workers_per_gpu": args.workers_per_gpu if args.parallel else 1,
                "parallel_tasks": len(args.gpus) * (args.workers_per_gpu if args.parallel else 1),
                "whole_task_wall_timeout_s": args.wall_timeout_seconds if args.parallel else 14400,
                "worker_slot_wait_limit_s": args.wait_for_slot_seconds if args.parallel else None,
                "robot_action_limit_seconds": args.robot_time_limit_seconds
                if args.parallel
                else None,
                "execute": args.execute,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.execute:
        return
    cache = args.model_cache.resolve()
    expected = json.loads((root / catalog["model_lock"]["file"]).read_text())
    if json.loads((cache / "dream_models.lock.json").read_text()) != expected:
        raise ValueError("Prepared models differ from the recorded production versions")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dream-controller-", dir=output.parent) as temporary:
        revised = Path(temporary) / "DREAM_code"
        materialize_controller(source, revised, overrides)
        command = [
            sys.executable,
            str(revised / "experiments/run_instruction_frozen_batch.py"),
            "--source-repo",
            str(revised),
            "--tasks",
            *map(str, tasks),
            "--output",
            str(output),
            "--heading-navigation",
            "--seeds",
            *map(str, args.seeds),
            "--variants",
            *args.variants,
            "--gpus",
            *args.gpus,
            "--wall-timeout",
            str(args.wall_timeout_seconds) if args.parallel else "14400",
            "--asset-dir",
            str(args.asset_dir.resolve()),
            "--model-cache",
            str(cache),
        ]
        if manifest is not None:
            if args.controller not in (
                "recovery_v6",
                "compact_v1",
                "staged_return",
                "continuous_return",
            ):
                raise ValueError(
                    "Explicit residential cohort requires a supported residential controller"
                )
            command.extend(["--cohort-id", manifest["id"]])
        if full_comparison:
            command.append("--benchmark")
        if args.parallel:
            command.extend(
                [
                    "--prepare-only",
                    "--simulation-time-limit-seconds",
                    str(args.robot_time_limit_seconds),
                ]
            )
        subprocess.run(command, cwd=engine, check=True)
        if args.parallel:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dream_sim.batch",
                    "--run",
                    str(output),
                    "--gpus",
                    *args.gpus,
                    "--workers-per-gpu",
                    str(args.workers_per_gpu),
                    "--raster-threads",
                    str(args.raster_threads),
                    "--time-limit-seconds",
                    str(args.wall_timeout_seconds),
                    "--robot-time-limit-seconds",
                    str(args.robot_time_limit_seconds),
                    "--wait-for-slot-seconds",
                    str(args.wait_for_slot_seconds),
                    "--execute",
                ],
                cwd=PROJECT,
                check=True,
            )


if __name__ == "__main__":
    main()
