#!/usr/bin/env python3
"""Unified, manifest-oriented entry point for DREAM simulator experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dream_simulation_core import PROTOCOLS, SCENE_FAMILIES, VARIANTS


HERE = Path(__file__).resolve().parent


def _csv_values(text: str, cast=str):
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _from_manifest(path: Path) -> tuple[str, list[str], list[int], str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    spec = data["spec"]
    return (
        spec["scene_family"],
        [str(spec["scene_id"])],
        [int(spec["seed"])],
        spec["protocol"],
        spec["variant"],
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run reproducible DREAM dynamic simulation episodes; unknown options are forwarded to the backend."
    )
    parser.add_argument("--scene-family", choices=SCENE_FAMILIES, default="replicacad")
    parser.add_argument("--scene-ids", default="apt_0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="target_move")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--render", choices=("none", "video"), default="video")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/dream_simulation"))
    parser.add_argument("--replay-manifest", type=Path)
    return parser.parse_known_args()


def main() -> None:
    args, forwarded = parse_args()
    scene_ids = _csv_values(args.scene_ids)
    seeds = _csv_values(args.seeds, int)
    scene_family, protocol, variant = args.scene_family, args.protocol, args.variant
    if args.replay_manifest:
        scene_family, scene_ids, seeds, protocol, variant = _from_manifest(args.replay_manifest)

    plan = []
    for scene_id in scene_ids:
        for seed in seeds:
            episode_root = args.output_root / scene_family / scene_id / f"seed_{seed}" / protocol / variant
            if scene_family == "replicacad":
                if scene_id != "apt_0":
                    raise SystemExit(
                        "The verified physical ReplicaCAD backend currently supports apt_0 only; "
                        "use the six-scene policy compatibility script for apt_1--apt_5."
                    )
                if protocol not in {"target_move", "combined"}:
                    raise SystemExit(
                        "The physical ReplicaCAD backend currently implements target_move/combined only."
                    )
                backend = HERE / "maniskill_dynamic_dream.py"
            else:
                backend = HERE / "maniskill_architecthor_dynamic.py"
                if not backend.exists():
                    raise SystemExit(
                        "ArchitecTHOR backend is unavailable until the official AI2THOR asset is installed."
                    )
            command = [
                sys.executable,
                str(backend),
                "--output-root",
                str(episode_root),
                "--protocol",
                protocol,
                "--variant",
                variant,
                "--seed",
                str(seed),
                *forwarded,
            ]
            if scene_family == "architecthor":
                command += ["--scene-id", scene_id]
            if args.render == "none":
                command += ["--capture-stride", "100000000"]
            plan.append({"scene_id": scene_id, "seed": seed, "command": command})

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "batch_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    for episode in plan:
        print(json.dumps({"launch": episode}, sort_keys=True), flush=True)
        subprocess.run(episode["command"], check=True)


if __name__ == "__main__":
    main()
