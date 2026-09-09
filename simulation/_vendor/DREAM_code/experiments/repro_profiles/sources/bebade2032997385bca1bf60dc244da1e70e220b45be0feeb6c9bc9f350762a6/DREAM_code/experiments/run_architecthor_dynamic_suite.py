#!/usr/bin/env python3
"""Run the frozen ten-episode ArchitecTHOR dynamic DREAM protocol.

Every episode is a separate simulator process so physics state cannot leak
between scenes.  A trial is counted only when its own manifest says
``status=success``; failures remain on disk with diagnostics and are listed in
the aggregate report rather than silently replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


SUITE = (
    ("ArchitecTHOR-Test-03", 3),
    ("ArchitecTHOR-Val-02", 5),
    ("ArchitecTHOR-Val-01", 11),
    ("ArchitecTHOR-Test-02", 7),
    ("ArchitecTHOR-Val-04", 13),
    ("ArchitecTHOR-Val-03", 17),
    ("ArchitecTHOR-Test-04", 19),
    ("ArchitecTHOR-Test-00", 23),
    ("ArchitecTHOR-Test-03", 29),
    ("ArchitecTHOR-Val-01", 31),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(root: Path, command_rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in command_rows:
        episode = root / row["episode"]
        manifest_path = episode / "manifest.json"
        failure_path = episode / "failure.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "success":
                successes.append(
                    {
                        "episode": row["episode"],
                        "scene_id": manifest["spec"]["scene_id"],
                        "seed": manifest["spec"]["seed"],
                        "control_steps": manifest["control_steps"],
                        "simulated_duration_s": manifest["simulated_duration_s"],
                        "base_travel_m": manifest["base_travel_m"],
                        "frontier_replans": manifest["frontier_replans"],
                        "measured_relocation_m": manifest["measured_relocation_m"],
                        "measured_object_lift_m": manifest["measured_object_lift_m"],
                        "stale_region_voxels_before": manifest.get(
                            "stale_region_voxels_before"
                        ),
                        "stale_region_voxels_after": manifest.get(
                            "stale_region_voxels_after"
                        ),
                        "grasp_bilateral_contact": manifest["grasp_bilateral_contact"],
                        "place_success": manifest["place_success"],
                        "video": {
                            "path": str(episode / "dream_dynamic_reacquisition.mp4"),
                            **manifest["artifacts"]["dream_dynamic_reacquisition.mp4"],
                        },
                        "manifest_sha256": sha256(manifest_path),
                    }
                )
                continue
        detail: dict[str, Any] = {"episode": row["episode"]}
        if failure_path.exists():
            detail.update(json.loads(failure_path.read_text(encoding="utf-8")))
        else:
            detail["error"] = f"subprocess return code {row['returncode']}"
        failures.append(detail)

    def mean(key: str) -> float | None:
        values = [float(item[key]) for item in successes]
        return sum(values) / len(values) if values else None

    return {
        "schema_version": 1,
        "protocol": "DREAM dynamic target relocation, invalidation, reacquisition, grasp, carry, place",
        "counting_rule": "only per-episode manifests with status=success are counted",
        "requested_episode_count": len(command_rows),
        "successful_episode_count": len(successes),
        "failed_episode_count": len(failures),
        "unique_successful_scenes": sorted({item["scene_id"] for item in successes}),
        "means_over_successes": {
            "base_travel_m": mean("base_travel_m"),
            "frontier_replans": mean("frontier_replans"),
            "measured_relocation_m": mean("measured_relocation_m"),
            "measured_object_lift_m": mean("measured_object_lift_m"),
        },
        "successes": successes,
        "failures": failures,
        "commands": command_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/architecthor_dynamic_suite"),
    )
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--only", type=int, nargs="*", default=None)
    parser.add_argument("--rerun-successes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    adapter = Path(__file__).with_name("maniskill_architecthor_dynamic.py")
    selected = range(len(SUITE)) if args.only is None else args.only
    commands: list[dict[str, Any]] = []
    for index in selected:
        scene_id, seed = SUITE[index]
        slug = scene_id.lower().replace("architecthor-", "").replace("-", "_")
        episode_name = f"episode_{index + 1:02d}_{slug}_seed{seed}"
        episode = args.output_root / episode_name
        manifest = episode / "manifest.json"
        command = [
            sys.executable,
            str(adapter),
            "--scene-id", scene_id,
            "--seed", str(seed),
            "--output-root", str(episode),
            "--capture-stride", str(args.capture_stride),
            "--width", str(args.width),
            "--height", str(args.height),
            "--fps", str(args.fps),
        ]
        if (
            not args.rerun_successes
            and manifest.exists()
            and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "success"
        ):
            returncode = 0
            skipped = True
        else:
            log = episode / "run.log"
            episode.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            returncode = process.returncode
            skipped = False
        row = {
            "index": index,
            "episode": episode_name,
            "scene_id": scene_id,
            "seed": seed,
            "command": command,
            "returncode": returncode,
            "skipped_existing_success": skipped,
        }
        commands.append(row)
        report = summarize(args.output_root, commands)
        (args.output_root / "aggregate.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"[{index + 1:02d}] {scene_id} seed={seed}: "
            f"{'SUCCESS' if returncode == 0 and manifest.exists() else 'FAILURE'}",
            flush=True,
        )
    final = summarize(args.output_root, commands)
    (args.output_root / "aggregate.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: final[k] for k in (
        "requested_episode_count", "successful_episode_count",
        "failed_episode_count", "unique_successful_scenes"
    )}, indent=2))
    return 0 if final["successful_episode_count"] == len(commands) else 1


if __name__ == "__main__":
    raise SystemExit(main())
