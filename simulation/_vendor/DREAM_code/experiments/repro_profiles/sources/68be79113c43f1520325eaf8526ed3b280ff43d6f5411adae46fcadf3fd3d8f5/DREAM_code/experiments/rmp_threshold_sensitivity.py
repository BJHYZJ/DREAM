#!/usr/bin/env python3
"""Threshold sensitivity using the same real RGB-D replay and RMP policy."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rmp_rgbd_replay import (
    DATASETS,
    SOURCE_COMMIT,
    add_drift,
    build_reference,
    download_dataset,
    load_frames,
    run_variant,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="bonn_removing_nonobstructing_box")
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/rmp_sensitivity"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=100)
    args = parser.parse_args()

    settings = [
        ("translation", value, value, 1e6)
        for value in (0.05, 0.10, 0.15, 0.20)
    ] + [
        ("rotation", value, 1e6, value)
        for value in (2.5, 5.0, 7.5, 10.0)
    ]
    sequence = download_dataset(args.data_root, args.dataset)
    base_frames, intrinsics = load_frames(
        args.dataset,
        sequence,
        args.seeds[0],
        stride=args.stride,
        max_frames=args.max_frames,
        max_translation=0.50,
        max_yaw_deg=15.0,
    )
    reference = build_reference(base_frames, intrinsics, 5)
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        drifted_poses = add_drift(
            [frame.pose_true for frame in base_frames],
            seed,
            0.50,
            15.0,
        )
        frames = [
            replace(frame, pose_drifted=drifted_pose)
            for frame, drifted_pose in zip(base_frames, drifted_poses)
        ]
        for sweep, value, translation, rotation in settings:
            print(f"seed={seed} {sweep} threshold={value}", flush=True)
            result, _ = run_variant(
                args.dataset,
                seed,
                "full_rmp",
                frames,
                intrinsics,
                reference,
                keyframe_interval=5,
                max_observations=500,
                max_keyframes=100,
                translation_threshold=translation,
                rotation_threshold_deg=rotation,
                realtime_window=10,
                short_term_window=50,
                global_fraction=2.0 / 3.0,
                regional_fraction=1.0 / 3.0,
            )
            row = asdict(result)
            row.update({"sweep": sweep, "threshold": value})
            rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "threshold_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    for axis, sweep, xlabel in zip(axes, ("translation", "rotation"), ("Translation threshold (m)", "Rotation threshold (deg)")):
        sweep_rows = [row for row in rows if row["sweep"] == sweep]
        values = sorted({float(row["threshold"]) for row in sweep_rows})
        fscores = [[float(row["fscore_10cm"]) for row in sweep_rows if float(row["threshold"]) == value] for value in values]
        means = np.asarray([np.mean(item) for item in fscores])
        errors = np.asarray([1.96 * np.std(item, ddof=1) / np.sqrt(len(item)) for item in fscores])
        axis.errorbar(values, means, yerr=errors, marker="o", capsize=3, color="#2a9d8f")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("F-score @ 10 cm")
        axis.set_ylim(0, 1.02)
    figure.savefig(args.output_root / "threshold_sensitivity.png", dpi=240)
    figure.savefig(args.output_root / "threshold_sensitivity.pdf")
    plt.close(figure)
    metadata = vars(args) | {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "source_commit": SOURCE_COMMIT,
        "injected_max_translation_m": 0.50,
        "injected_max_yaw_deg": 15.0,
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
