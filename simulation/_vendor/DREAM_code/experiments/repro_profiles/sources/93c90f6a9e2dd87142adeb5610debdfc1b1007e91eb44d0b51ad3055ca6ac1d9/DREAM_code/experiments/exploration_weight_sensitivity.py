#!/usr/bin/env python3
"""Sensitivity of DREAM's semantic/recency frontier weight."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exploration_gridworld import LAYOUTS, run_episode, wilson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/exploration_sensitivity"))
    args = parser.parse_args()
    rows = []
    for rate in args.rates:
        results = []
        for layout in LAYOUTS:
            for seed in range(args.trials):
                result, _ = run_episode(
                    layout,
                    seed,
                    "dream",
                    budget=args.budget,
                    visibility_radius=5,
                    detection_radius=4,
                    semantic_rate=rate,
                    semantic_signal=1.0,
                )
                results.append(result)
        successes = sum(result.success for result in results)
        low, high = wilson(successes, len(results))
        successful_steps = [result.path_steps for result in results if result.success]
        row = {
            "semantic_rate": rate,
            "episodes": len(results),
            "success_rate": successes / len(results),
            "ci95_low": low,
            "ci95_high": high,
            "median_steps_success": float(np.median(successful_steps)) if successful_steps else float("nan"),
        }
        rows.append(row)
        print(row, flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "semantic_weight.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rates = np.asarray([row["semantic_rate"] for row in rows])
    success = np.asarray([row["success_rate"] for row in rows])
    lower = success - np.asarray([row["ci95_low"] for row in rows])
    upper = np.asarray([row["ci95_high"] for row in rows]) - success
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    axes[0].errorbar(rates, success, yerr=[lower, upper], marker="o", capsize=3, color="#2a9d8f")
    axes[0].set_xlabel("Semantic weight")
    axes[0].set_ylabel("Target-search success")
    axes[0].set_ylim(0, 1.05)
    axes[1].plot(rates, [row["median_steps_success"] for row in rows], marker="o", color="#457b9d")
    axes[1].set_xlabel("Semantic weight")
    axes[1].set_ylabel("Median steps (successes)")
    figure.savefig(args.output_root / "semantic_weight.png", dpi=240)
    figure.savefig(args.output_root / "semantic_weight.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
