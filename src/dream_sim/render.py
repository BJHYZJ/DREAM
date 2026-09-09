"""Reproduce a selected spectator view by replaying the same saved controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from dream_sim.run import CATALOG, ENGINE, execute, digest, load_catalog, verify_source
from dream_sim.sources import audit_root


def renderer_source(case_id: str) -> Path:
    root, catalog = load_catalog(CATALOG)
    case = next(row for row in catalog["cases"] if row["id"] == case_id)
    recorded = json.loads((root / case["records"]["spectator_video_review.json"]).read_text())
    expected = recorded["renderer_script_sha256"]
    if digest(ENGINE / "experiments" / "instruction_replay_video.py") == expected:
        return ENGINE
    for entry in catalog["sources"].values():
        candidate = root / entry["repository"]
        renderer = candidate / "experiments" / "instruction_replay_video.py"
        if renderer.is_file() and digest(renderer) == expected:
            return verify_source(root, entry)
    candidate = audit_root() / "record_v3" / "DREAM_code"
    if digest(candidate / "experiments" / "instruction_replay_video.py") == expected:
        return candidate
    raise ValueError(f"Missing original spectator-renderer version for case {case_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True, help="Root containing case_XX/execution")
    parser.add_argument("--audits", type=Path, help="Separate audit root containing case_XX/record; defaults to execution's own audits")
    parser.add_argument("--case", choices=("07", "10"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case_name = "case_" + args.case
    batch = args.execution.resolve() / case_name / "execution"
    reviewed = (args.audits.resolve() / case_name if args.audits
                else args.execution.resolve() / case_name / "audit")
    record_path = reviewed / "record" / "record_review.json"
    record = json.loads(record_path.read_text())
    if not record["record_review_passed"]:
        raise ValueError("A passing record review is required before spectator rendering")
    attempt, = json.loads((batch / "batch_result.json").read_text())["attempts"]
    run = batch / attempt["name"]
    if Path(record["source_run"]).resolve() != run:
        raise ValueError("Review refers to another policy execution")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((batch / "protocol.json").read_text())
    from run_instruction_frozen_batch import freeze_sources
    source = renderer_source(args.case)
    workspace = output.parent / (output.name + "_source")
    helpers = workspace / "DREAM_code" / "experiments"
    hashes = freeze_sources(source, workspace / "DREAM_code")
    (workspace / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    for name, target in ((".maniskill_assets", protocol["asset_root"]), (".dream_model_cache", protocol["model_cache"])):
        (workspace / name).symlink_to(target, target_is_directory=True)
    env = os.environ.copy()
    env.update(MS_ASSET_DIR=protocol["asset_root"], HF_HUB_CACHE=protocol["model_cache"],
               HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
    command = [sys.executable, str(helpers / "replay_instruction_actions.py"), "--source-run", str(run),
               "--output", str(output), "--render-review", str(record_path),
               "--spectator-mode", "tool-front-placement"]
    result = execute(command, output.with_suffix(".log"), env)
    if result != 0:
        raise SystemExit(result)
    physical = json.loads((output / "audit.json").read_text())
    display = json.loads((output / "spectator_video_review.json").read_text())
    passed = physical["physical_reexecution_passed"] and display["render_checks_passed"]
    report = {"case": args.case, "passed": passed, "new_policy_execution": False,
              "renderer_script_sha256": digest(helpers / "instruction_replay_video.py"),
              "renderer_matches_original_video": True,
              "same_controls_spectator_rerender": True, "mode": "tool-front-placement"}
    (output / "render_acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
