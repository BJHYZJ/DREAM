#!/usr/bin/env python3
"""Combine RMP and exploration sensitivity results for the manuscript."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "results"


def main() -> None:
    with (ROOT / "rmp_sensitivity/threshold_runs.csv").open(encoding="utf-8") as stream:
        rmp = list(csv.DictReader(stream))
    with (ROOT / "exploration_sensitivity/semantic_weight.csv").open(encoding="utf-8") as stream:
        exploration = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 2.65), constrained_layout=True)
    for axis, sweep, xlabel in zip(
        axes[:2],
        ("translation", "rotation"),
        ("Translation trigger (m)", "Rotation trigger (deg)"),
    ):
        sweep_rows = [row for row in rmp if row["sweep"] == sweep]
        thresholds = sorted({float(row["threshold"]) for row in sweep_rows})
        values = [
            [float(row["fscore_10cm"]) for row in sweep_rows if float(row["threshold"]) == threshold]
            for threshold in thresholds
        ]
        means = np.asarray([np.mean(item) for item in values])
        errors = np.asarray([1.96 * np.std(item, ddof=1) / np.sqrt(len(item)) for item in values])
        axis.errorbar(thresholds, means, yerr=errors, marker="o", capsize=3, color="#2a9d8f")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("RMP F-score @ 10 cm")
        axis.set_ylim(0.5, 1.03)
        axis.grid(alpha=0.2)
    rates = np.asarray([float(row["semantic_rate"]) for row in exploration])
    success = np.asarray([float(row["success_rate"]) for row in exploration])
    lower = success - np.asarray([float(row["ci95_low"]) for row in exploration])
    upper = np.asarray([float(row["ci95_high"]) for row in exploration]) - success
    axes[2].errorbar(rates, success, yerr=[lower, upper], marker="o", capsize=3, color="#457b9d")
    axes[2].set_xlabel("Semantic exploration weight")
    axes[2].set_ylabel("Target-search success")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(alpha=0.2)
    output = ROOT / "combined_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "controlled_sensitivity.png", dpi=240)
    figure.savefig(output / "controlled_sensitivity.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
