"""Run the ten released profiles using their original learned controllers.

This entrypoint does not feed saved actions to a policy. Independent physics
replay is an explicitly separate audit after each newly executed episode.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SIMULATION = Path(__file__).resolve().parent
ENGINE = SIMULATION / "_vendor" / "DREAM_code"
HELPERS = ENGINE / "experiments"
CATALOG = HELPERS / "repro_profiles" / "profiles.json"
sys.path.insert(0, str(HELPERS))
from run_instruction_profile import (  # noqa: E402
    build_command, child, load_catalog, verify_case, verify_source,
)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def preflight(asset_dir: Path, model_cache: Path, *, runtime: bool = True) -> dict:
    """Verify source/task locks; optionally check installed runtime and assets."""
    root, catalog = load_catalog(CATALOG)
    for entry in catalog["sources"].values():
        verify_source(root, entry)
    for case in catalog["cases"]:
        verify_case(root, case)
        from simulation.auditor import record_reviewer
        record_reviewer(case, root)
    report = {
        "catalog_sha256": digest(CATALOG), "profiles_verified": 10,
        "source_snapshots_verified": len(catalog["sources"]),
        "original_video_review_versions_verified": 10,
        "runtime_checked": runtime, "policy_executed": False,
    }
    if not runtime:
        return report
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use the documented Python 3.11 environment.")
    versions = {}
    for name in ("requirements.txt", "requirements-maniskill.txt", "requirements-learned.txt"):
        for line in (HELPERS / name).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if "==" not in line:
                continue
            package, expected = line.split("==", 1)
            installed = importlib.metadata.version(package)
            if installed.split("+", 1)[0] != expected:
                raise RuntimeError(f"{package}: expected {expected}, installed {installed}")
            versions[package] = installed
    expected_models = json.loads(child(root, catalog["model_lock"]["file"]).read_text())
    if json.loads((model_cache / "dream_models.lock.json").read_text()) != expected_models:
        raise ValueError("Model cache revisions differ from the release lock.")
    for model, revision in expected_models.items():
        snapshot = model_cache / ("models--" + model.replace("/", "--")) / "snapshots" / revision
        required = [snapshot / "config.json", *snapshot.glob("*.safetensors")]
        if len(required) < 2 or any(not p.is_file() or p.stat().st_size == 0 for p in required):
            raise FileNotFoundError(f"Incomplete model snapshot: {model}@{revision}")
    lock = json.loads(child(root, catalog["asset_lock"]["file"]).read_text())
    for relative, expected in lock["artifacts"].items():
        path = asset_dir / "data" / "scene_datasets" / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Missing or mismatched locked asset: {relative}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This supported configuration requires CUDA inference.")
    icd = os.environ.get("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
    if not all(Path(path).is_file() for path in icd.split(":")):
        raise FileNotFoundError("Set VK_ICD_FILENAMES to an installed Vulkan ICD.")
    report.update(
        python=sys.version, packages=versions, asset_files_verified=len(lock["artifacts"]),
        asset_dir=str(asset_dir), model_cache=str(model_cache),
        model_revisions_verified=True, model_contents_sha256_verified=False,
        cuda_devices=torch.cuda.device_count(), vulkan_icd=icd,
        renderer_execution_tested=False,
        boundary="Preflight is not a task result; rendering and behavior are tested by actual execution.",
    )
    return report


def execute(command: list[str], log: Path, env: dict, *, cwd: Path = ENGINE) -> int:
    with log.open("x") as stream:
        return subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                              stderr=subprocess.STDOUT).returncode


def audit(case: dict, batch: Path, run: Path, destination: Path, env: dict) -> dict:
    """Preserve the historical definition and the declared case-05 correction."""
    destination.mkdir()
    frozen = batch / "frozen_workspace" / "DREAM_code" / "experiments"
    # Only case 05 needs the already documented evaluator-v4 correction.
    # Never substitute a new scorer in response to a failed new attempt.
    helpers = frozen
    if case["scoring_reassessment"] or case["id"] == "07":
        from run_instruction_frozen_batch import freeze_sources
        audit_workspace = destination / "workspace"
        helpers = audit_workspace / "DREAM_code" / "experiments"
        freeze_sources(ENGINE, audit_workspace / "DREAM_code")
        for name, key in ((".maniskill_assets", "MS_ASSET_DIR"), (".dream_model_cache", "HF_HUB_CACHE")):
            (audit_workspace / name).symlink_to(env[key], target_is_directory=True)
    report = {"new_policy_execution": False, "scoring_reassessment": case["scoring_reassessment"]}
    command = [sys.executable, str(helpers / "replay_instruction_actions.py"),
               "--source-run", str(run), "--output", str(destination / "physical")]
    code = execute(command, destination / "physical.log", env)
    report["physical_returncode"] = code
    if code != 0:
        return dict(report, passed=False)
    physical = json.loads((destination / "physical" / "audit.json").read_text())
    report["physical_reexecution_passed"] = physical["physical_reexecution_passed"]
    from simulation.auditor import record_reviewer
    reviewer = record_reviewer(case, CATALOG.parent)
    report["record_review_script_sha256"] = digest(reviewer)
    report["record_review_matches_original_video"] = True
    command = [sys.executable, str(reviewer),
               "--run", str(run), "--physical-audit", str(destination / "physical" / "audit.json"),
               "--output", str(destination / "record"), "--annotate"]
    if case["scoring_reassessment"]:
        reassessment = [sys.executable, str(helpers / "reassess_instruction_evaluation.py"),
                        "--source-run", str(run), "--output", str(destination / "reassessment")]
        code = execute(reassessment, destination / "reassessment.log", env)
        report["reassessment_returncode"] = code
        if code != 0:
            return dict(report, passed=False)
        command.extend(["--scoring-reassessment", str(destination / "reassessment" / "reassessment.json")])
    code = execute(command, destination / "record.log", env)
    report["record_returncode"] = code
    record_path = destination / "record" / "record_review.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    report.update(
        record_review_passed=record.get("record_review_passed", False),
        record_definition_version=record.get("review_definition_version", 1),
        native_contact_control_steps=physical["contact_audit"]["native_environment_contact_control_steps"],
        passed=code == 0 and physical["physical_reexecution_passed"] and record.get("record_review_passed", False),
    )
    return report


def run_case(case: dict, output: Path, gpu: str, asset_dir: Path, model_cache: Path) -> dict:
    directory = output / ("case_" + case["id"])
    directory.mkdir()
    report = {"case": case["id"], "scene": case["scene"], "seed": case["seed"],
              "source_id": case["source_id"], "started_unix_s": time.time(), "passed": False}
    env = os.environ.copy()
    env.update(MS_ASSET_DIR=str(asset_dir), HF_HUB_CACHE=str(model_cache), HF_HUB_OFFLINE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
    try:
        command, validated = build_command(
            CATALOG, case_id=case["id"], output=directory / "execution", gpus=[gpu],
            asset_dir=asset_dir, model_cache=model_cache,
        )
        report["validated_configuration"] = validated
        report["policy_returncode"] = execute(command, directory / "policy.log", env)
        terminal = directory / "execution" / "batch_result.json"
        if not terminal.is_file():
            raise RuntimeError("Policy launcher did not save its terminal attempt record.")
        attempts = json.loads(terminal.read_text())["attempts"]
        if len(attempts) != 1:
            raise ValueError("Expected exactly one attempt, with no automatic retries.")
        attempt = attempts[0]
        report["attempt"] = attempt
        run = directory / "execution" / attempt["name"]
        report["original_scoring_protocol_success"] = attempt.get("protocol_success")
        if report["policy_returncode"] == 0 and attempt["status"] == "completed":
            report["audit"] = audit(case, directory / "execution", run, directory / "audit", env)
            report["passed"] = report["audit"]["passed"] and all(attempt.get(key) for key in (
                "source_verified_before", "source_verified_after", "task_verified_before", "task_verified_after"))
            if report["passed"] and case["same_controls_spectator_rerender"]:
                render_command = [sys.executable, "-m", "simulation.render", "--execution", str(output),
                                  "--case", case["id"], "--output", str(directory / "spectator")]
                code = execute(render_command, directory / "spectator_launcher.log", env, cwd=SIMULATION.parent)
                rendered = directory / "spectator" / "render_acceptance.json"
                report["spectator_render"] = json.loads(rendered.read_text()) if rendered.is_file() else {"passed": False}
                report["passed"] = code == 0 and report["spectator_render"]["passed"]
        report["saved_sha256"] = {name: digest(run / name) for name in (
            "result.json", "actions.json", "evaluator_trajectory.json") if (run / name).is_file()}
        if report["passed"]:
            report["review_video"] = str(directory / ("spectator" if case["same_controls_spectator_rerender"] else "audit/record") / "reviewer_view.mp4")
            report["video_frame_record"] = str(run / "video_frames.json")
            report["review_video_playback_speed"] = 1
    except Exception as error:
        report["error"] = repr(error)
    report["finished_unix_s"] = time.time()
    report["boundary"] = "One fresh learned-policy attempt and separate audit; all outcomes retained. Not a population success rate."
    (directory / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--case", choices=[f"{index:02d}" for index in range(1, 11)])
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asset-dir", type=Path, default=SIMULATION / ".runtime" / "assets")
    parser.add_argument("--model-cache", type=Path, default=SIMULATION / ".runtime" / "models")
    parser.add_argument("--gpus", nargs="+", default=["0"], help="One entry per worker; default: one worker")
    parser.add_argument("--dry-run", action="store_true", help="Validate profiles without importing the simulator")
    args = parser.parse_args()
    assets, models = args.asset_dir.resolve(), args.model_cache.resolve()
    checked = preflight(assets, models, runtime=not args.dry_run)
    print(json.dumps(checked, indent=2), flush=True)
    if args.preflight:
        return
    if args.output is None:
        parser.error("--output is required for task execution or dry-run planning")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing attempts are never overwritten")
    _, catalog = load_catalog(CATALOG)
    cases = [case for case in catalog["cases"] if args.all or case["id"] == args.case]
    if args.dry_run:
        for case in cases:
            _, report = build_command(CATALOG, case_id=case["id"], output=output / ("case_" + case["id"]) / "execution",
                                      gpus=[args.gpus[0]], asset_dir=assets, model_cache=models)
            print(json.dumps(report), flush=True)
        return
    output.mkdir(parents=True)
    (output / "preflight.json").write_text(json.dumps(checked, indent=2) + "\n")
    (output / "planned.json").write_text(json.dumps({
        "case_ids": [case["id"] for case in cases], "attempts_per_case": 1,
        "gpu_workers": args.gpus, "automatic_retries": False,
        "scoring_correction_case_ids": [case["id"] for case in cases if case["scoring_reassessment"]],
    }, indent=2) + "\n")
    pending = iter(cases)
    lock = threading.Lock()
    completed = []

    def worker(gpu: str) -> None:
        while True:
            with lock:
                case = next(pending, None)
            if case is None:
                return
            print(json.dumps({"event": "started", "case": case["id"], "gpu": gpu}), flush=True)
            report = run_case(case, output, gpu, assets, models)
            with lock:
                completed.append(report)
                (output / "progress.json").write_text(json.dumps({
                    "planned": len(cases), "completed": len(completed),
                    "passed": sum(row["passed"] for row in completed),
                    "cases": [{key: row[key] for key in ("case", "passed")} for row in completed],
                }, indent=2) + "\n")
            print(json.dumps({"event": "finished", "case": case["id"], "passed": report["passed"]}), flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        list(pool.map(worker, args.gpus))
    result = {"planned": len(cases), "completed": len(completed),
              "all_passed": len(completed) == len(cases) and all(row["passed"] for row in completed),
              "cases": sorted(completed, key=lambda row: row["case"]),
              "public_release_performed": False, "author_visual_review_pending": True}
    (output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
