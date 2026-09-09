#!/usr/bin/env python3
"""Controlled grid-world evaluation of DREAM's frontier-selection policy.

This benchmark isolates navigation-policy behavior from perception.  It uses
procedurally randomized indoor layouts and a noisy target-context field; it is
therefore reported as a controlled simulation, not as additional robot data.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binomtest, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "6d558e25f045a0a414e2f7ebdfcab5b78940c83d"


def _load_policy():
    path = REPO_ROOT / "src/dream/mapping/exploration_policy.py"
    spec = importlib.util.spec_from_file_location("dream_gridworld_policy", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.select_frontier_goal


select_frontier_goal = _load_policy()
POLICIES = ("random", "nearest", "history_only", "dream")
LAYOUTS = ("rooms", "corridors", "clutter")
NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class EpisodeResult:
    layout: str
    seed: int
    policy: str
    success: int
    path_steps: int
    replans: int
    coverage: float
    final_target_distance: float
    planning_ms: float
    semantic_rate: float
    source_commit: str = SOURCE_COMMIT


def _carve_room(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top : top + height, left : left + width] = True


def make_layout(layout: str, rng: np.random.Generator, size: int = 61) -> np.ndarray:
    """Create a connected free-space mask with randomized doors/obstacles."""

    free = np.ones((size, size), dtype=bool)
    free[[0, -1], :] = False
    free[:, [0, -1]] = False
    if layout == "rooms":
        for row in (15, 30, 45):
            free[row, 1:-1] = False
            for segment in range(4):
                door = int(rng.integers(segment * 15 + 3, min(segment * 15 + 13, size - 2)))
                free[row, door - 1 : door + 2] = True
        for col in (15, 30, 45):
            free[1:-1, col] = False
            for segment in range(4):
                door = int(rng.integers(segment * 15 + 3, min(segment * 15 + 13, size - 2)))
                free[door - 1 : door + 2, col] = True
    elif layout == "corridors":
        free[:] = False
        free[28:33, 2:-2] = True
        for col in (8, 19, 30, 41, 52):
            upward = bool(rng.integers(0, 2))
            if upward:
                free[5:31, col - 2 : col + 3] = True
                _carve_room(free, 3, col - 5, 10, 11)
            else:
                free[30:56, col - 2 : col + 3] = True
                _carve_room(free, 49, col - 5, 9, 11)
    elif layout == "clutter":
        for _ in range(24):
            height, width = rng.integers(3, 9, size=2)
            top = int(rng.integers(3, size - height - 3))
            left = int(rng.integers(3, size - width - 3))
            free[top : top + height, left : left + width] = False
        # Two guaranteed cross-aisles keep the sampled layouts connected.
        free[28:33, 1:-1] = True
        free[1:-1, 28:33] = True
    else:
        raise ValueError(layout)
    return free


def flood_distances(free: np.ndarray, start: tuple[int, int]) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    distances = np.full(free.shape, -1, dtype=np.int32)
    distances[start] = 0
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for dr, dc in NEIGHBORS:
            nxt = (row + dr, col + dc)
            if (
                0 <= nxt[0] < free.shape[0]
                and 0 <= nxt[1] < free.shape[1]
                and free[nxt]
                and distances[nxt] < 0
            ):
                distances[nxt] = distances[row, col] + 1
                parents[nxt] = (row, col)
                queue.append(nxt)
    return distances, parents


def reconstruct_path(
    parents: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    if goal != start and goal not in parents:
        return []
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    return list(reversed(path))


def sample_start_target(
    free: np.ndarray, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates = np.column_stack(np.where(free))
    start_arr = candidates[int(rng.integers(len(candidates)))]
    start = (int(start_arr[0]), int(start_arr[1]))
    distances, _ = flood_distances(free, start)
    far = np.column_stack(np.where(distances >= max(30, int(distances.max() * 0.7))))
    target_arr = far[int(rng.integers(len(far)))]
    return start, (int(target_arr[0]), int(target_arr[1]))


def reveal(
    free: np.ndarray,
    known: np.ndarray,
    last_seen: np.ndarray,
    position: tuple[int, int],
    timestamp: int,
    radius: int,
) -> None:
    rr, cc = np.ogrid[: free.shape[0], : free.shape[1]]
    visible = (rr - position[0]) ** 2 + (cc - position[1]) ** 2 <= radius * radius
    known[visible] = np.where(free[visible], 1, -1)
    last_seen[visible & free] = timestamp


def frontier_mask(known: np.ndarray, reachable: np.ndarray) -> np.ndarray:
    adjacent_to_reachable = np.zeros_like(reachable)
    for dr, dc in NEIGHBORS:
        adjacent_to_reachable |= np.roll(reachable, (dr, dc), axis=(0, 1))
    frontier = (known == 0) & adjacent_to_reachable
    frontier[[0, -1], :] = False
    frontier[:, [0, -1]] = False
    return frontier


def approach_path(
    parents: dict[tuple[int, int], tuple[int, int]],
    distances: np.ndarray,
    start: tuple[int, int],
    frontier_goal: tuple[int, int],
) -> list[tuple[int, int]]:
    approaches = []
    for dr, dc in NEIGHBORS:
        point = (frontier_goal[0] + dr, frontier_goal[1] + dc)
        if (
            0 <= point[0] < distances.shape[0]
            and 0 <= point[1] < distances.shape[1]
            and distances[point] >= 0
        ):
            approaches.append(point)
    if not approaches:
        return []
    approach = min(approaches, key=lambda point: distances[point])
    return reconstruct_path(parents, start, approach)


def target_context_field(
    free: np.ndarray,
    target: tuple[int, int],
    rng: np.random.Generator,
    signal: float,
) -> np.ndarray:
    """Noisy semantic-context proxy with distractors and localization error."""

    field = np.zeros(free.shape, dtype=np.float64)
    offset = np.rint(rng.normal(0.0, 4.0, size=2)).astype(int)
    cue = np.clip(np.asarray(target) + offset, 1, np.asarray(free.shape) - 2)
    cue_point = (int(cue[0]), int(cue[1]))
    if not free[cue_point]:
        cue_point = target
    geodesic, _ = flood_distances(free, cue_point)
    reachable = geodesic >= 0
    field[reachable] = signal * (1.0 - geodesic[reachable] / max(1, geodesic[reachable].max()))
    distractors = np.zeros_like(field)
    for _ in range(3):
        distractor = np.asarray(
            [rng.integers(1, free.shape[0] - 1), rng.integers(1, free.shape[1] - 1)]
        )
        distractors[int(distractor[0]), int(distractor[1])] += rng.uniform(0.20, 0.55)
    field += gaussian_filter(distractors, sigma=9.0)
    field += rng.normal(0.0, max(field.max(), 1e-8) * 0.03, size=field.shape)
    field[~free] = 0.0
    field -= field.min()
    if field.max() > 0:
        field /= field.max()
    return field


def choose_goal(
    policy: str,
    frontier: np.ndarray,
    position: tuple[int, int],
    last_seen: np.ndarray,
    semantic: np.ndarray,
    rng: np.random.Generator,
    semantic_rate: float,
) -> tuple[int, int] | None:
    candidates = np.column_stack(np.where(frontier))
    if not len(candidates):
        return None
    if policy == "random":
        chosen = candidates[int(rng.integers(len(candidates)))]
        return int(chosen[0]), int(chosen[1])
    distances = np.linalg.norm(candidates - np.asarray(position), axis=1)
    if policy == "nearest":
        chosen = candidates[int(np.argmin(distances))]
        return int(chosen[0]), int(chosen[1])

    normalized_history = last_seen.copy()
    positive = normalized_history > 0
    if np.any(positive):
        minimum = normalized_history[positive].min()
        span = normalized_history[positive].max() - minimum
        if span > 0:
            normalized_history = np.maximum(normalized_history - minimum, 0.0) / span
            normalized_history[positive] = np.maximum(
                normalized_history[positive], np.finfo(float).eps * 10
            )
    history = np.ma.array(normalized_history, mask=~frontier)
    age = history.max() - history
    age[history <= 1e-8] = float("inf")
    time_scores = 1.0 / (1.0 + np.exp(-10.0 * (age - 0.01)))
    decision = select_frontier_goal(
        time_scores,
        semantic if policy == "dream" else None,
        frontier,
        np.asarray(position),
        semantic_rate=semantic_rate,
        min_distance_cells=3.0,
    )
    return decision.index


def run_episode(
    layout: str,
    seed: int,
    policy: str,
    *,
    budget: int,
    visibility_radius: int,
    detection_radius: int,
    semantic_rate: float,
    semantic_signal: float,
    keep_trace: bool = False,
) -> tuple[EpisodeResult, dict[str, object] | None]:
    # Layout, target, and semantic field are identical across policies.
    scenario_rng = np.random.default_rng(seed * 104729 + LAYOUTS.index(layout) * 1009)
    free = make_layout(layout, scenario_rng)
    start, target = sample_start_target(free, scenario_rng)
    semantic = target_context_field(free, target, scenario_rng, semantic_signal)
    policy_rng = np.random.default_rng(seed * 8191 + POLICIES.index(policy) * 131)
    known = np.zeros(free.shape, dtype=np.int8)
    last_seen = np.zeros(free.shape, dtype=np.float64)
    position = start
    trajectory = [position]
    steps = 0
    replans = 0
    planning_seconds = 0.0
    success = False
    reveal(free, known, last_seen, position, 1, visibility_radius)

    while steps < budget:
        if math.dist(position, target) <= detection_radius:
            success = True
            break
        distances, parents = flood_distances(known == 1, position)
        reachable = distances >= 0
        frontiers = frontier_mask(known, reachable)
        started = time.perf_counter()
        goal = choose_goal(
            policy,
            frontiers,
            position,
            last_seen,
            semantic,
            policy_rng,
            semantic_rate,
        )
        planning_seconds += time.perf_counter() - started
        if goal is None:
            break
        path = approach_path(parents, distances, position, goal)
        if len(path) <= 1:
            # Inspect the selected outside-frontier cell without charging a
            # motion step when it directly borders the current footprint.
            reveal(free, known, last_seen, position, steps + 1, visibility_radius + 1)
            continue
        replans += 1
        for point in path[1:]:
            if steps >= budget:
                break
            position = point
            trajectory.append(position)
            steps += 1
            reveal(free, known, last_seen, position, steps + 1, visibility_radius)
            if math.dist(position, target) <= detection_radius:
                success = True
                break
        if success:
            break

    result = EpisodeResult(
        layout=layout,
        seed=seed,
        policy=policy,
        success=int(success),
        path_steps=steps,
        replans=replans,
        coverage=float(np.sum(known == 1) / np.sum(free)),
        final_target_distance=float(math.dist(position, target)),
        planning_ms=planning_seconds * 1000.0,
        semantic_rate=semantic_rate,
    )
    trace = None
    if keep_trace:
        trace = {
            "free": free,
            "start": start,
            "target": target,
            "semantic": semantic,
            "trajectory": np.asarray(trajectory),
            "known": known,
        }
    return result, trace


def wilson(successes: int, count: int) -> tuple[float, float]:
    z = 1.96
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return center - margin, center + margin


def summarize(results: list[EpisodeResult]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for layout in (*LAYOUTS, "all"):
        for policy in POLICIES:
            subset = [
                result
                for result in results
                if result.policy == policy and (layout == "all" or result.layout == layout)
            ]
            successes = sum(result.success for result in subset)
            low, high = wilson(successes, len(subset))
            successful_steps = [result.path_steps for result in subset if result.success]
            rows.append(
                {
                    "layout": layout,
                    "policy": policy,
                    "episodes": len(subset),
                    "success_rate": successes / len(subset),
                    "success_ci_low": low,
                    "success_ci_high": high,
                    "median_steps_success": float(np.median(successful_steps)) if successful_steps else float("nan"),
                    "mean_coverage": float(np.mean([result.coverage for result in subset])),
                    "mean_planning_ms": float(np.mean([result.planning_ms for result in subset])),
                }
            )
    return rows


def paired_tests(results: list[EpisodeResult], budget: int) -> dict[str, object]:
    indexed = {
        (result.layout, result.seed, result.policy): result for result in results
    }
    scenarios = sorted({(result.layout, result.seed) for result in results})
    output: dict[str, object] = {}
    for baseline in POLICIES[:-1]:
        baseline_wins = dream_wins = 0
        baseline_costs = []
        dream_costs = []
        for layout, seed in scenarios:
            baseline_result = indexed[(layout, seed, baseline)]
            dream_result = indexed[(layout, seed, "dream")]
            baseline_wins += int(baseline_result.success and not dream_result.success)
            dream_wins += int(dream_result.success and not baseline_result.success)
            baseline_costs.append(
                baseline_result.path_steps if baseline_result.success else budget + 1
            )
            dream_costs.append(
                dream_result.path_steps if dream_result.success else budget + 1
            )
        discordant = baseline_wins + dream_wins
        mcnemar_p = (
            float(binomtest(dream_wins, discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        statistic, step_p = wilcoxon(baseline_costs, dream_costs, alternative="two-sided")
        output[f"dream_vs_{baseline}"] = {
            "paired_scenarios": len(scenarios),
            "dream_only_successes": dream_wins,
            "baseline_only_successes": baseline_wins,
            "exact_mcnemar_p": mcnemar_p,
            "penalized_steps_wilcoxon_statistic": float(statistic),
            "penalized_steps_two_sided_p": float(step_p),
        }
    return output


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) if hasattr(row, "__dataclass_fields__") else row for row in rows]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def plot_summary(summary: list[dict[str, float | str | int]], output: Path) -> None:
    aggregate = {str(row["policy"]): row for row in summary if row["layout"] == "all"}
    colors = ("#9e9e9e", "#607d8b", "#f4a261", "#2a9d8f")
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), constrained_layout=True)
    rates = [float(aggregate[policy]["success_rate"]) for policy in POLICIES]
    lower = [rates[i] - float(aggregate[policy]["success_ci_low"]) for i, policy in enumerate(POLICIES)]
    upper = [float(aggregate[policy]["success_ci_high"]) - rates[i] for i, policy in enumerate(POLICIES)]
    axes[0].bar(range(4), rates, yerr=[lower, upper], capsize=3, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Target-search success")
    axes[0].set_xticks(range(4), ["Random", "Nearest", "History", "DREAM"])
    steps = [float(aggregate[policy]["median_steps_success"]) for policy in POLICIES]
    axes[1].bar(range(4), steps, color=colors)
    axes[1].set_ylabel("Median steps (successes)")
    axes[1].set_xticks(range(4), ["Random", "Nearest", "History", "DREAM"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def plot_example(traces: dict[str, dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for axis, policy in zip(axes, POLICIES):
        trace = traces[policy]
        free = np.asarray(trace["free"])
        semantic = np.asarray(trace["semantic"])
        trajectory = np.asarray(trace["trajectory"])
        axis.imshow(~free, cmap="gray_r", vmin=0, vmax=1)
        axis.imshow(semantic, cmap="magma", alpha=np.where(free, 0.35, 0.0), vmin=0, vmax=1)
        axis.plot(trajectory[:, 1], trajectory[:, 0], color="#168aad", linewidth=1.2)
        start = trace["start"]
        target = trace["target"]
        axis.scatter(start[1], start[0], marker="o", c="#52b788", s=30, edgecolor="black")
        axis.scatter(target[1], target[0], marker="*", c="#e63946", s=65, edgecolor="black")
        axis.set_title(policy.replace("_", " "))
        axis.set_xticks([])
        axis.set_yticks([])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--visibility-radius", type=int, default=5)
    parser.add_argument("--detection-radius", type=int, default=4)
    parser.add_argument("--semantic-rate", type=float, default=0.1)
    parser.add_argument("--semantic-signal", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/exploration"))
    args = parser.parse_args()

    results: list[EpisodeResult] = []
    example_traces: dict[str, dict[str, object]] = {}
    for layout in LAYOUTS:
        for seed in range(args.trials):
            for policy in POLICIES:
                result, trace = run_episode(
                    layout,
                    seed,
                    policy,
                    budget=args.budget,
                    visibility_radius=args.visibility_radius,
                    detection_radius=args.detection_radius,
                    semantic_rate=args.semantic_rate,
                    semantic_signal=args.semantic_signal,
                    keep_trace=layout == "rooms" and seed == 0,
                )
                results.append(result)
                if trace is not None:
                    example_traces[policy] = trace
        print(f"Finished {layout}: {args.trials * len(POLICIES)} episodes", flush=True)

    summary = summarize(results)
    tests = paired_tests(results, args.budget)
    write_csv(args.output_root / "episodes.csv", results)
    write_csv(args.output_root / "summary.csv", summary)
    (args.output_root / "paired_tests.json").write_text(
        json.dumps(tests, indent=2), encoding="utf-8"
    )
    plot_summary(summary, args.output_root / "policy_comparison")
    plot_example(example_traces, args.output_root / "example_trajectories")
    metadata = vars(args) | {
        "output_root": str(args.output_root),
        "layouts": LAYOUTS,
        "policies": POLICIES,
        "source_commit": SOURCE_COMMIT,
        "semantic_field": "Gaussian target-context proxy with offset, distractors, and noise",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps([row for row in summary if row["layout"] == "all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
