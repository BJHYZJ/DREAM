#!/usr/bin/env python3
"""Aggregate public RGB-D RMP replays and generate paper-ready plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


VARIANTS = ("no_rmp", "reintegration_only", "pruning_only", "full_rmp")
LABELS = ("No RMP", "Reintegrate", "Prune", "Full RMP")
COLORS = ("#9e9e9e", "#f4a261", "#457b9d", "#2a9d8f")


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        with path.open(encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    if not rows:
        raise FileNotFoundError(f"No result CSV files found below {root}")
    return rows


def mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, 0.0
    return mean, float(1.96 * np.std(array, ddof=1) / np.sqrt(len(array)))


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = (
        "chamfer_m",
        "fscore_10cm",
        "ghost_voxel_rate",
        "free_space_violation_rate",
        "rgb_feature_residual",
        "cache_megabytes",
        "correction_seconds",
        "active_voxels",
    )
    output: list[dict[str, object]] = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in (*datasets, "all"):
        for variant in ("oracle", *VARIANTS):
            subset = [
                row
                for row in rows
                if row["variant"] == variant
                and (dataset == "all" or row["dataset"] == dataset)
            ]
            if not subset:
                continue
            aggregate_row: dict[str, object] = {
                "dataset": dataset,
                "variant": variant,
                "runs": len(subset),
            }
            for metric in metrics:
                values = [float(row[metric]) for row in subset]
                mean, ci = mean_ci(values)
                aggregate_row[f"{metric}_mean"] = mean
                aggregate_row[f"{metric}_ci95"] = ci
            output.append(aggregate_row)
    return output


def paired_tests(rows: list[dict[str, str]]) -> dict[str, object]:
    indexed = {(row["dataset"], row["seed"], row["variant"]): row for row in rows}
    pairs = sorted({(row["dataset"], row["seed"]) for row in rows})
    tests: dict[str, object] = {}
    for metric in ("chamfer_m", "fscore_10cm", "ghost_voxel_rate", "free_space_violation_rate"):
        before = []
        after = []
        for dataset, seed in pairs:
            no_rmp = indexed.get((dataset, seed, "no_rmp"))
            full = indexed.get((dataset, seed, "full_rmp"))
            if no_rmp is not None and full is not None:
                before.append(float(no_rmp[metric]))
                after.append(float(full[metric]))
        statistic, pvalue = wilcoxon(before, after, alternative="two-sided")
        tests[metric] = {
            "pairs": len(before),
            "no_rmp_mean": float(np.mean(before)),
            "full_rmp_mean": float(np.mean(after)),
            "wilcoxon_statistic": float(statistic),
            "two_sided_p": float(pvalue),
        }
    return tests


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, str]], output: Path) -> None:
    metrics = (
        ("fscore_10cm", "F-score @ 10 cm", (0, 1.05)),
        ("ghost_voxel_rate", "Ghost-voxel rate", (0, None)),
        ("free_space_violation_rate", "Free-space violation", (0, None)),
        ("cache_megabytes", "Retained cache (MiB)", (0, None)),
    )
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for axis, (metric, ylabel, ylim) in zip(axes, metrics):
        values = [[float(row[metric]) for row in rows if row["variant"] == variant] for variant in VARIANTS]
        means = [np.mean(value) for value in values]
        errors = [1.96 * np.std(value, ddof=1) / np.sqrt(len(value)) for value in values]
        axis.bar(range(4), means, yerr=errors, capsize=3, color=COLORS)
        axis.set_xticks(range(4), LABELS, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("experiments/results/rmp"))
    args = parser.parse_args()
    rows = read_rows(args.input_root)
    summary = aggregate(rows)
    write_summary(args.input_root / "summary.csv", summary)
    tests = paired_tests(rows)
    (args.input_root / "paired_tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    plot(rows, args.input_root / "rmp_ablation")
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
