#!/usr/bin/env python3
"""Run and audit the ten-video DREAM dynamic simulation suite.

Each episode is a fresh simulator process.  The suite spans four official
furnished interiors, six independent target/placement layouts, and ten
spatially distinct dynamic-relocation configurations.  Changing only the
random seed never creates an additional suite item.  A video is counted only
when its own manifest reports every physical success gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


SUITE = (
    {
        "family": "architecthor", "scene": "ArchitecTHOR-Test-03",
        "layout": "far-southeast-reloc-0", "layout_index": 0, "seed": 0,
        "relocation_layout_index": 0, "place_near_target": False,
    },
    {
        "family": "architecthor", "scene": "ArchitecTHOR-Test-03",
        "layout": "far-southeast-reloc-2", "layout_index": 0, "seed": 0,
        "relocation_layout_index": 2, "place_near_target": False,
    },
    {
        "family": "architecthor", "scene": "ArchitecTHOR-Test-03",
        "layout": "northwest-reloc-0", "layout_index": 2, "seed": 0,
        "relocation_layout_index": 0, "place_near_target": True,
        "place_base": (-3.817, 0.112),
    },
    {
        "family": "architecthor", "scene": "ArchitecTHOR-Test-03",
        "layout": "northwest-reloc-1", "layout_index": 2, "seed": 0,
        "relocation_layout_index": 1, "place_near_target": True,
        "place_base": (-3.817, 0.112),
    },
    {
        "family": "architecthor", "scene": "ArchitecTHOR-Test-04",
        "layout": "south-reloc-0", "layout_index": 0, "seed": 0,
        "relocation_layout_index": 0, "place_near_target": True,
    },
    {
        "family": "architecthor", "scene": "ArchitecTHOR-Test-04",
        "layout": "south-reloc-1", "layout_index": 0, "seed": 0,
        "relocation_layout_index": 1, "place_near_target": True,
    },
    {
        "family": "architecthor", "scene": "ArchitecTHOR-Test-04",
        "layout": "southwest-reloc-0", "layout_index": 2, "seed": 0,
        "relocation_layout_index": 0, "place_near_target": True,
        "place_base": (1.262, -0.658),
    },
    {
        "family": "architecthor", "scene": "ArchitecTHOR-Val-02",
        "layout": "north-reloc-0", "layout_index": 2, "seed": 0,
        "relocation_layout_index": 0, "place_near_target": True,
        "place_base": (-0.836, -0.072),
    },
    {
        "family": "replicacad_multiscene", "scene": "apt_2",
        "layout": "central-cross-room-reloc-2", "layout_index": 0, "seed": 0,
        "relocation_layout_index": 2, "layout_mode": "custom",
        "initial_target": (1.451, -3.477), "place_base": (1.10, -2.50),
        "extra_args": ("--lift-height", "0.08"),
    },
    {
        "family": "replicacad_multiscene", "scene": "apt_2",
        "layout": "central-cross-room-reloc-3", "layout_index": 0, "seed": 0,
        "relocation_layout_index": 3, "layout_mode": "custom",
        "initial_target": (1.451, -3.477), "place_base": (1.10, -2.50),
        "extra_args": ("--lift-height", "0.08"),
    },
)

REQUIRED_EVENTS = {
    "initial_360_scan_complete",
    "dynamic_motion_start",
    "dynamic_motion_complete",
    "stale_target_region_rejected",
    "bilateral_contact_check",
    "measured_lift_contact_check",
    "post_grasp_clearance_retreat",
    "grid_planned_physical_carry",
    "place_verification",
}
REACQUISITION_EVENTS = {
    "target_observed_from_current_rgbd",
    "target_observed_while_base_moving",
    "target_reacquired_in_focused_frame",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_name(index: int, item: dict[str, Any]) -> str:
    scene = item["scene"].lower().replace("architecthor-", "").replace("_", "-")
    layout = item["layout"].replace("_", "-")
    return f"episode_{index + 1:02d}_{scene}_{layout}"


def read_result(root: Path, index: int, item: dict[str, Any]) -> dict[str, Any]:
    name = episode_name(index, item)
    directory = root / name
    manifest_path = directory / "manifest.json"
    failure_path = directory / "failure.json"
    result: dict[str, Any] = {
        "episode": name,
        "family": item["family"],
        "scene_id": item["scene"],
        "layout_id": item["layout"],
        "seed": item["seed"],
        "status": "not_run",
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        events_path = directory / "events.csv"
        event_names: set[str] = set()
        event_rows: list[dict[str, str]] = []
        if events_path.exists():
            with events_path.open(newline="", encoding="utf-8") as stream:
                event_rows = list(csv.DictReader(stream))
                event_names = {row.get("event", "") for row in event_rows}
        stale_steps = [
            int(float(row["step"]))
            for row in event_rows
            if row.get("event") == "stale_target_region_rejected"
        ]
        reacquisition_after_stale = bool(stale_steps) and any(
            row.get("event") in REACQUISITION_EVENTS
            and int(float(row["step"])) >= max(stale_steps)
            for row in event_rows
        )
        required_lift = float(manifest.get("required_peak_object_lift_m", float("inf")))
        peak_lift = float(manifest.get("peak_measured_object_lift_m", 0.0))
        retained_threshold = float(
            manifest.get("retained_lift_threshold_m", float("inf"))
        )
        retained_lift = float(manifest.get("measured_object_lift_m", 0.0))
        gates = (
            manifest.get("status") == "success"
            and float(manifest.get("measured_relocation_m", 0.0)) >= 0.50
            and int(manifest.get("stale_region_voxels_before", 0)) > 0
            and int(manifest.get("stale_region_voxels_after", -1)) == 0
            and manifest.get("target_world_pose_used_by_planner") is False
            and REQUIRED_EVENTS <= event_names
            and reacquisition_after_stale
            and manifest.get("grasp_bilateral_contact") is True
            and peak_lift >= required_lift
            and retained_lift >= retained_threshold
            and manifest.get("place_success") is True
            and manifest.get("scripted_pose_updates_after_initialization") is False
        )
        if gates:
            video = directory / "dream_dynamic_reacquisition.mp4"
            result.update(
                status="success",
                control_steps=manifest["control_steps"],
                simulated_duration_s=manifest["simulated_duration_s"],
                video_duration_s=manifest["video_duration_s"],
                base_travel_m=manifest["base_travel_m"],
                frontier_replans=manifest["frontier_replans"],
                measured_relocation_m=manifest["measured_relocation_m"],
                stale_voxels_before=manifest["stale_region_voxels_before"],
                stale_voxels_after=manifest["stale_region_voxels_after"],
                peak_measured_object_lift_m=peak_lift,
                measured_object_lift_m=manifest["measured_object_lift_m"],
                grasp_bilateral_contact=manifest["grasp_bilateral_contact"],
                place_success=manifest["place_success"],
                no_pose_setters_after_initialization=not manifest[
                    "scripted_pose_updates_after_initialization"
                ],
                required_event_audit=True,
                video_path=str(video),
                video_bytes=video.stat().st_size,
                video_sha256=sha256(video),
                manifest_sha256=sha256(manifest_path),
                official_scene_id=manifest.get("official_scene_id", item["scene"]),
                episode_layout=manifest.get("episode_layout", {}),
            )
            return result
    if failure_path.exists():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        result.update(status="failure", error=failure.get("error"), phase=failure.get("phase"))
    return result


def write_aggregate(root: Path) -> dict[str, Any]:
    results = [read_result(root, index, item) for index, item in enumerate(SUITE)]
    successes = [item for item in results if item["status"] == "success"]

    def mean(key: str) -> float | None:
        values = [float(item[key]) for item in successes]
        return sum(values) / len(values) if values else None

    def total(key: str) -> float:
        return sum(float(item[key]) for item in successes)

    def value_range(key: str) -> list[float] | None:
        values = [float(item[key]) for item in successes]
        return [min(values), max(values)] if values else None

    declared_configuration_keys = [f"{item['scene']}::{item['layout']}" for item in SUITE]
    if len(set(declared_configuration_keys)) != len(SUITE):
        raise RuntimeError("suite contains a duplicate scene/layout configuration")

    endpoint_groups: dict[tuple, list[tuple[float, float]]] = {}
    for item in successes:
        layout = item["episode_layout"]
        group_key = (
            item["scene_id"],
            tuple(layout.get("initial_target_xy_m", ())),
            tuple(layout.get("place_base_xy_m", ())),
        )
        endpoint = layout.get(
            "runtime_selected_final_target_xy_m",
            layout.get("final_target_xy_m"),
        )
        if endpoint is not None:
            endpoint_groups.setdefault(group_key, []).append(
                (float(endpoint[0]), float(endpoint[1]))
            )
    repeated_endpoint_distances = [
        math.hypot(first[0] - second[0], first[1] - second[1])
        for endpoints in endpoint_groups.values()
        if len(endpoints) > 1
        for first, second in itertools.combinations(endpoints, 2)
    ]
    minimum_repeated_endpoint_separation = (
        min(repeated_endpoint_distances) if repeated_endpoint_distances else None
    )
    if (
        minimum_repeated_endpoint_separation is not None
        and minimum_repeated_endpoint_separation < 0.45 - 1e-6
    ):
        raise RuntimeError(
            "repeated-layout relocation endpoints violate the 0.45 m separation rule"
        )

    successful_scene_episode_counts = {
        scene: sum(item["scene_id"] == scene for item in successes)
        for scene in sorted({item["scene_id"] for item in successes})
    }
    aggregate = {
        "schema_version": 1,
        "protocol": (
            "initial rotation; seeded online RGB-D search; force-driven target/support "
            "relocation; focus and stale-memory invalidation; reacquisition; bilateral "
            "contact grasp; measured lift; physical carry; verified placement"
        ),
        "counting_rule": (
            "Only status=success manifests with at least 0.50 m measured relocation, a "
            "nonempty stale target region reduced to zero, no target-world-pose planning, "
            "the complete rotate/search/move/clear/reacquire/grasp/lift/carry/place event trace, "
            "bilateral grasp, peak measured lift at or above the declared threshold, "
            "retained lift at or above 75% of that threshold, verified placement, and no "
            "post-initialization pose setters are counted."
        ),
        "diversity_rule": (
            "Every episode has a unique official-scene/layout/relocation identifier; "
            "seed-only duplicates are forbidden. Repeated initial target/placement "
            "layouts use relocation endpoints selected at least 0.45 m apart."
        ),
        "intended_episode_count": len(SUITE),
        "successful_episode_count": len(successes),
        "failed_episode_count": sum(item["status"] == "failure" for item in results),
        "not_run_episode_count": sum(item["status"] == "not_run" for item in results),
        "scene_families": sorted({item["family"] for item in successes}),
        "scene_ids": sorted({item["scene_id"] for item in successes}),
        "successful_scene_episode_counts": successful_scene_episode_counts,
        "distinct_successful_scene_count": len(
            {item["scene_id"] for item in successes}
        ),
        "distinct_successful_configuration_count": len(
            {(item["scene_id"], item["layout_id"]) for item in successes}
        ),
        "distinct_successful_target_placement_layout_count": len({
            (
                item["scene_id"],
                tuple(item["episode_layout"].get("initial_target_xy_m", ())),
                tuple(item["episode_layout"].get("place_base_xy_m", ())),
            )
            for item in successes
        }),
        "distinct_successful_dynamic_endpoint_count": len({
            (
                item["scene_id"],
                tuple(
                    item["episode_layout"].get(
                        "runtime_selected_final_target_xy_m",
                        item["episode_layout"].get("final_target_xy_m", ()),
                    )
                ),
            )
            for item in successes
        }),
        "minimum_repeated_layout_endpoint_separation_m": (
            minimum_repeated_endpoint_separation
        ),
        "declared_configuration_keys": declared_configuration_keys,
        "means_over_successes": {
            "base_travel_m": mean("base_travel_m"),
            "frontier_replans": mean("frontier_replans"),
            "measured_relocation_m": mean("measured_relocation_m"),
            "measured_object_lift_m": mean("measured_object_lift_m"),
        },
        "totals_over_successes": {
            "control_steps": int(total("control_steps")),
            "simulated_duration_s": total("simulated_duration_s"),
            "video_duration_s": total("video_duration_s"),
            "base_travel_m": total("base_travel_m"),
            "frontier_replans": int(total("frontier_replans")),
            "stale_voxels_before": int(total("stale_voxels_before")),
            "video_bytes": int(total("video_bytes")),
        },
        "ranges_over_successes": {
            "base_travel_m": value_range("base_travel_m"),
            "frontier_replans": value_range("frontier_replans"),
            "measured_relocation_m": value_range("measured_relocation_m"),
            "peak_measured_object_lift_m": value_range(
                "peak_measured_object_lift_m"
            ),
            "retained_measured_object_lift_m": value_range(
                "measured_object_lift_m"
            ),
        },
        "episodes": results,
    }
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/maniskill_dynamic_video_suite_r2"),
    )
    parser.add_argument("--capture-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--only", type=int, nargs="*", default=None)
    parser.add_argument("--rerun-successes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = range(len(SUITE)) if args.only is None else args.only
    for index in selected:
        item = SUITE[index]
        name = episode_name(index, item)
        output = args.output_root / name
        manifest = output / "manifest.json"
        if (
            manifest.exists()
            and not args.rerun_successes
            and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "success"
        ):
            print(f"[{index + 1:02d}] {name}: SKIP EXISTING SUCCESS", flush=True)
            continue
        adapter_name = {
            "architecthor": "maniskill_architecthor_dynamic.py",
            "replicacad_multiscene": "maniskill_replicacad_multiscene_dynamic.py",
        }[item["family"]]
        adapter = Path(__file__).with_name(adapter_name)
        command = [
            sys.executable,
            str(adapter),
            "--seed", str(item["seed"]),
            "--output-root", str(output),
            "--capture-stride", str(args.capture_stride),
            "--width", str(args.width),
            "--height", str(args.height),
            "--fps", str(args.fps),
            # This limit was fixed before the formal suite and is shared by
            # every episode.  It suppresses overshoot in the Cartesian wrist
            # motion without changing any success threshold.
            "--ee-action-limit", "0.03",
        ]
        if item["family"] == "architecthor":
            command.extend((
                "--scene-id", item["scene"],
                "--layout-index", str(item["layout_index"]),
                "--minimum-layout-separation", "1.0",
                "--relocation-layout-index", str(item["relocation_layout_index"]),
                "--minimum-relocation-layout-separation", "0.45",
            ))
            command.append(
                "--place-near-target"
                if item["place_near_target"]
                else "--no-place-near-target"
            )
            if "place_base" in item:
                command.extend((
                    "--place-base-override-x", str(item["place_base"][0]),
                    "--place-base-override-y", str(item["place_base"][1]),
                ))
        elif item["family"] == "replicacad_multiscene":
            command.extend((
                "--build-config-index", item["scene"].split("_")[-1],
                "--layout-index", str(item["layout_index"]),
                "--layout-mode", item["layout_mode"],
                "--minimum-layout-separation", "1.0",
                "--relocation-layout-index", str(item["relocation_layout_index"]),
                "--minimum-relocation-layout-separation", "0.45",
            ))
            if item["layout_mode"] == "custom":
                command.extend((
                    "--initial-target-x", str(item["initial_target"][0]),
                    "--initial-target-y", str(item["initial_target"][1]),
                    "--place-base-x", str(item["place_base"][0]),
                    "--place-base-y", str(item["place_base"][1]),
                ))
        command.extend(item.get("extra_args", ()))
        output.mkdir(parents=True, exist_ok=True)
        with (output / "run.log").open("w", encoding="utf-8") as stream:
            returncode = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            ).returncode
        status = read_result(args.output_root, index, item)["status"]
        print(f"[{index + 1:02d}] {name}: {status.upper()} (rc={returncode})", flush=True)
        write_aggregate(args.output_root)
    aggregate = write_aggregate(args.output_root)
    print(json.dumps({key: aggregate[key] for key in (
        "successful_episode_count", "failed_episode_count", "not_run_episode_count",
        "scene_families", "scene_ids", "distinct_successful_scene_count",
        "distinct_successful_configuration_count",
        "distinct_successful_target_placement_layout_count",
        "distinct_successful_dynamic_endpoint_count",
    )}, indent=2))
    return 0 if aggregate["successful_episode_count"] == len(SUITE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
