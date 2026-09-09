#!/usr/bin/env python3
"""Controlled DREAM search check on all six ReplicaCAD apartment configs.

This is a navigation-mesh policy component check, not a physics manipulation
trial and not a learned-perception benchmark.  It complements (but does not
replace) the larger 300-layout HouseExpo scale test and the single closed-loop
ReplicaCAD search--grasp--place integration trial.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from houseexpo_cross_room import run_scenario
from maniskill_replicacad_search_video import (
    build_search_scenario,
    rasterize_navigation_mesh,
)


POLICIES = ("history_only", "nearest", "dream")
BOUNDARY = (
    "Policy-only unknown-map search on six Fetch-specific ReplicaCAD navigation "
    "meshes with controlled context and geometric visibility; no simulator "
    "controller, grasp, learned perception, or independent large-scale layout."
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    asset_root = Path(
        os.environ.get(
            "MS_ASSET_DIR",
            Path(__file__).resolve().parents[2] / ".maniskill_assets",
        )
    )
    scene_root = asset_root / "data/scene_datasets/replica_cad_dataset/configs/scenes"
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    dream_traces: list[tuple[int, object, dict[str, object], object]] = []

    for scene_index in range(6):
        scene_name = f"apt_{scene_index}"
        mesh_path = scene_root / f"{scene_name}.scene_instance.fetch.navigable_positions.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)
        mesh = trimesh.load(mesh_path, process=False)
        grid = rasterize_navigation_mesh(mesh, args.resolution)
        scenario = replace(
            build_search_scenario(
                grid,
                target_xy=np.asarray([args.target_x, args.target_y]),
                destination_boundary_y_m=args.destination_boundary_y,
            ),
            house_id=scene_name,
        )
        for policy in POLICIES:
            result, trace = run_scenario(
                scenario,
                policy,
                budget_m=args.budget_m,
                sensor_range_m=args.sensor_range,
                semantic_rate=args.semantic_rate,
                keep_trace=policy == "dream",
            )
            row = asdict(result)
            row.update(
                scene=scene_name,
                navmesh_area_m2=scenario.floor_area_m2,
                target_world_x_m=grid.cell_to_world(scenario.target)[0],
                target_world_y_m=grid.cell_to_world(scenario.target)[1],
            )
            rows.append(row)
            if policy == "dream":
                if trace is None:
                    raise RuntimeError(f"missing DREAM trace for {scene_name}")
                dream_traces.append((scene_index, scenario, trace, grid))

    write_csv(args.output_root / "episodes.csv", rows)
    summary: list[dict[str, object]] = []
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        successes = [row for row in selected if int(row["success"])]
        summary.append(
            {
                "policy": policy,
                "successes": len(successes),
                "trials": len(selected),
                "success_rate": len(successes) / len(selected),
                "mean_path_m_successes": (
                    float(np.mean([float(row["path_m"]) for row in successes]))
                    if successes
                    else None
                ),
                "mean_spl": float(np.mean([float(row["spl"]) for row in selected])),
                "mean_replans": float(
                    np.mean([float(row["replans"]) for row in selected])
                ),
            }
        )
    write_csv(args.output_root / "summary.csv", summary)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), constrained_layout=True)
    for axis, (scene_index, scenario, trace, grid) in zip(axes.ravel(), dream_traces):
        axis.imshow(~scenario.free, cmap="gray_r", vmin=0, vmax=1)
        trajectory = np.asarray(trace["trajectory"], dtype=float)
        axis.plot(trajectory[:, 1], trajectory[:, 0], color="#1489d6", linewidth=1.4)
        axis.scatter(scenario.start[1], scenario.start[0], c="#37a657", s=42, label="start")
        axis.scatter(scenario.target[1], scenario.target[0], c="#de3b32", marker="*", s=80, label="target")
        result_row = next(
            row for row in rows if row["scene"] == f"apt_{scene_index}" and row["policy"] == "dream"
        )
        axis.set_title(
            f"apt_{scene_index}: {float(result_row['path_m']):.1f} m, "
            f"{int(result_row['replans'])} replans"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    fig.suptitle(
        "DREAM unknown-map search on all ReplicaCAD apartment configurations\n"
        "Controlled navmesh/context component check — not physics or learned perception",
        fontsize=13,
    )
    figure_path = args.output_root / "replicacad_multiconfig_paths.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    metadata = {
        "boundary": BOUNDARY,
        "scenes": [f"apt_{index}" for index in range(6)],
        "trials_per_policy": 6,
        "target_request_xy_m": [args.target_x, args.target_y],
        "resolution_m": args.resolution,
        "sensor_range_m": args.sensor_range,
        "budget_m": args.budget_m,
        "semantic_rate": args.semantic_rate,
        "summary": summary,
        "note": (
            "All six configurations have similar apartment footprints; these runs "
            "check configuration robustness and do not extend the HouseExpo scale claim."
        ),
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/replicacad_multiconfig"),
    )
    parser.add_argument("--resolution", type=float, default=0.08)
    parser.add_argument("--sensor-range", type=float, default=1.20)
    parser.add_argument("--budget-m", type=float, default=80.0)
    parser.add_argument("--semantic-rate", type=float, default=0.10)
    parser.add_argument("--destination-boundary-y", type=float, default=-2.50)
    parser.add_argument("--target-x", type=float, default=0.79)
    parser.add_argument("--target-y", type=float, default=-6.30)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
