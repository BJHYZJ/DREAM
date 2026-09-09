#!/usr/bin/env python3
"""Measure the lightweight RMP decision overhead as history length grows."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_policy():
    path = REPO_ROOT / "src/dream/mapping/rmp_policy.py"
    spec = importlib.util.spec_from_file_location("dream_scaling_rmp", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def benchmark(function, repetitions: int) -> tuple[float, float]:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1000.0)
    return float(np.median(samples)), float(np.percentile(samples, 95))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/rmp_scaling"))
    args = parser.parse_args()
    policy = load_policy()
    rows = []
    for count in (100, 500, 1_000, 5_000, 10_000, 50_000):
        observations = list(range(count))
        keyframes = set(range(0, count, 5))
        affected = set(list(keyframes)[-max(1, len(keyframes) // 2) :])
        repetitions = 100 if count <= 5_000 else 25
        scope_median, scope_p95 = benchmark(
            lambda: policy.select_reintegration_scope(
                observations,
                keyframes,
                affected,
                realtime_window=10,
                short_term_window=50,
            ),
            repetitions,
        )
        prune_median, prune_p95 = benchmark(
            lambda: policy.select_observations_to_prune(
                observations,
                is_pose_graph_node=keyframes.__contains__,
                max_observations=count // 2,
                max_pose_graph_observations=count // 10,
            ),
            repetitions,
        )
        rows.append(
            {
                "observations": count,
                "repetitions": repetitions,
                "scope_median_us": scope_median,
                "scope_p95_us": scope_p95,
                "prune_median_us": prune_median,
                "prune_p95_us": prune_p95,
            }
        )
        print(rows[-1], flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "policy_scaling.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(4.1, 3.0), constrained_layout=True)
    x = [row["observations"] for row in rows]
    axis.loglog(x, [row["scope_median_us"] for row in rows], marker="o", label="Scope selection")
    axis.loglog(x, [row["prune_median_us"] for row in rows], marker="s", label="Keyframe pruning")
    axis.set_xlabel("Retained observations")
    axis.set_ylabel("Median policy time ($\\mu$s)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(args.output_root / "policy_scaling.png", dpi=240)
    figure.savefig(args.output_root / "policy_scaling.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
