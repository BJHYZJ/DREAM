#!/usr/bin/env python3
"""Cross-room target-search benchmark on public HouseExpo floor plans.

The benchmark rasterizes official HouseExpo test layouts, forces the start and
target into different annotated rooms, and reveals the map online with an
occlusion-aware range sensor.  It exercises DREAM's shared production frontier
selector.  Room/context scores are exposed only in already visible cells and
serve as a controlled semantic proxy, not as a perception accuracy claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import sys
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.stats import binomtest, wilcoxon

from exploration_gridworld import (
    POLICIES,
    choose_goal,
    frontier_mask,
    wilson,
)


HOUSEEXPO_COMMIT = "45e2b2505f6ea1fe49c0203f14efb7ce20b94e7c"
HOUSEEXPO_ARCHIVE = (
    "https://raw.githubusercontent.com/TeaganLi/HouseExpo/"
    f"{HOUSEEXPO_COMMIT}/HouseExpo/json.tar.gz"
)
HOUSEEXPO_TEST_IDS = (
    "https://raw.githubusercontent.com/TeaganLi/HouseExpo/"
    f"{HOUSEEXPO_COMMIT}/experiments/map_id_test.txt"
)
DREAM_COMMIT = "6d558e25f045a0a414e2f7ebdfcab5b78940c83d"
SCALE_RANGES = {"4--6 rooms": (4, 6), "7--9 rooms": (7, 9), "10+ rooms": (10, 10_000)}
MOVES_8 = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class Room:
    index: int
    labels: tuple[str, ...]
    bounds: tuple[float, float, float, float]
    mask: np.ndarray


@dataclass(frozen=True)
class Scenario:
    scale: str
    house_id: str
    official_room_count: int
    annotated_room_boxes: int
    floor_area_m2: float
    free: np.ndarray
    room_index: np.ndarray
    start: tuple[int, int]
    target: tuple[int, int]
    start_room: int
    target_room: int
    start_room_labels: tuple[str, ...]
    target_room_labels: tuple[str, ...]
    context: np.ndarray
    shortest_cells: float
    resolution_m: float


@dataclass(frozen=True)
class CrossRoomResult:
    scale: str
    house_id: str
    official_rooms: int
    annotated_rooms: int
    floor_area_m2: float
    start_room: str
    target_room: str
    policy: str
    success: int
    path_m: float
    shortest_path_m: float
    spl: float
    rooms_visited: int
    replans: int
    budget_m: float
    source_floorplan: str = "HouseExpo official test split"
    houseexpo_commit: str = HOUSEEXPO_COMMIT
    dream_commit: str = DREAM_COMMIT


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
    partial.replace(path)


def ensure_houseexpo(root: Path) -> tuple[Path, list[str]]:
    archive = root / "json.tar.gz"
    ids_path = root / "map_id_test.txt"
    json_root = root / "json"
    if not archive.exists():
        print(f"Downloading HouseExpo layouts from {HOUSEEXPO_ARCHIVE}", flush=True)
        _download(HOUSEEXPO_ARCHIVE, archive)
    if not ids_path.exists():
        _download(HOUSEEXPO_TEST_IDS, ids_path)
    if not json_root.exists():
        root.mkdir(parents=True, exist_ok=True)
        print("Extracting HouseExpo JSON layouts", flush=True)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(root, filter="data")
    ids = [value.strip() for value in ids_path.read_text(encoding="utf-8").split()]
    return json_root, ids


def _normalize_label(label: str) -> str:
    replacements = {
        "toilet": "bathroom",
        "guest_room": "bedroom",
        "child_room": "bedroom",
        "hall": "hallway",
    }
    normalized = label.strip().lower()
    return replacements.get(normalized, normalized)


def _deduplicated_boxes(data: dict[str, object]) -> list[tuple[tuple[float, ...], tuple[str, ...]]]:
    grouped: dict[tuple[float, ...], set[str]] = {}
    categories = data.get("room_category", {})
    assert isinstance(categories, dict)
    for label, boxes in categories.items():
        for box in boxes:
            key = tuple(round(float(value), 2) for value in box)
            grouped.setdefault(key, set()).add(_normalize_label(str(label)))
    return [(box, tuple(sorted(labels))) for box, labels in grouped.items()]


def _largest_component(free: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), 4)
    if count <= 1:
        return np.zeros_like(free)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def rasterize(data: dict[str, object], resolution: float, robot_radius: float) -> tuple[np.ndarray, list[Room], np.ndarray]:
    vertices = np.asarray(data["verts"], dtype=np.float64)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    pad = 3
    width, height = np.ceil((maximum - minimum) / resolution).astype(int) + 2 * pad + 1
    if width > 420 or height > 420:
        raise ValueError("layout exceeds raster size limit")
    polygon = np.rint((vertices - minimum) / resolution).astype(np.int32)
    polygon[:, 0] += pad
    polygon[:, 1] += pad
    free = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(free, [polygon], 1)
    radius_cells = max(1, int(math.ceil(robot_radius / resolution)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_cells + 1, 2 * radius_cells + 1)
    )
    free = cv2.erode(free, kernel).astype(bool)
    free = _largest_component(free)
    if int(free.sum()) < 200:
        raise ValueError("insufficient navigable area")

    rooms: list[Room] = []
    for box, labels in _deduplicated_boxes(data):
        x1, y1, x2, y2 = box
        c1 = max(0, int(math.floor((x1 - minimum[0]) / resolution)) + pad)
        r1 = max(0, int(math.floor((y1 - minimum[1]) / resolution)) + pad)
        c2 = min(width, int(math.ceil((x2 - minimum[0]) / resolution)) + pad + 1)
        r2 = min(height, int(math.ceil((y2 - minimum[1]) / resolution)) + pad + 1)
        mask = np.zeros_like(free)
        mask[r1:r2, c1:c2] = True
        mask &= free
        if int(mask.sum()) >= 20:
            rooms.append(Room(len(rooms), labels, box, mask))
    if len(rooms) < 2:
        raise ValueError("fewer than two usable annotated rooms")

    # Assign overlaps to the smallest annotated box, which prevents open-plan
    # or duplicate labels from inflating the number of visited rooms.
    room_index = np.full(free.shape, -1, dtype=np.int16)
    order = sorted(rooms, key=lambda room: float(room.mask.sum()), reverse=True)
    for room in order:
        room_index[room.mask] = room.index
    return free, rooms, room_index


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def _central_candidates(room: Room, free_clearance: np.ndarray) -> np.ndarray:
    values = free_clearance[room.mask]
    if not len(values):
        return np.empty((0, 2), dtype=int)
    threshold = max(1.0, float(np.quantile(values, 0.60)))
    return np.column_stack(np.where(room.mask & (free_clearance >= threshold)))


def weighted_distances(
    free: np.ndarray, start: tuple[int, int]
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    """Eight-connected Dijkstra search with diagonal corner-cut prevention."""

    distances = np.full(free.shape, np.inf, dtype=np.float64)
    distances[start] = 0.0
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    while queue:
        cost, row, col = heapq.heappop(queue)
        if cost > distances[row, col] + 1e-12:
            continue
        for dr, dc, move_cost in MOVES_8:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < free.shape[0] and 0 <= nc < free.shape[1]):
                continue
            if not free[nr, nc]:
                continue
            if dr and dc and (not free[row + dr, col] or not free[row, col + dc]):
                continue
            candidate = cost + move_cost
            if candidate + 1e-12 < distances[nr, nc]:
                distances[nr, nc] = candidate
                parents[(nr, nc)] = (row, col)
                heapq.heappush(queue, (candidate, nr, nc))
    return distances, parents


def _reconstruct_path(
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


def weighted_approach_path(
    parents: dict[tuple[int, int], tuple[int, int]],
    distances: np.ndarray,
    start: tuple[int, int],
    frontier_goal: tuple[int, int],
) -> list[tuple[int, int]]:
    approaches = []
    for dr, dc, move_cost in MOVES_8:
        point = (frontier_goal[0] + dr, frontier_goal[1] + dc)
        if (
            0 <= point[0] < distances.shape[0]
            and 0 <= point[1] < distances.shape[1]
            and np.isfinite(distances[point])
        ):
            approaches.append((distances[point] + move_cost, point))
    if not approaches:
        return []
    return _reconstruct_path(parents, start, min(approaches)[1])


def _line_cells(start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Conservatively sample grid cells intersected by a continuous segment."""

    extent = max(abs(goal[0] - start[0]), abs(goal[1] - start[1]))
    samples = max(1, 2 * extent)
    cells: list[tuple[int, int]] = []
    for fraction in np.linspace(0.0, 1.0, samples + 1):
        cell = (
            int(round(start[0] + fraction * (goal[0] - start[0]))),
            int(round(start[1] + fraction * (goal[1] - start[1]))),
        )
        if not cells or cell != cells[-1]:
            cells.append(cell)
    return cells


def shortcut_path(
    path: list[tuple[int, int]], traversable: np.ndarray
) -> list[tuple[int, int]]:
    """Greedy line-of-sight compression over robot-radius-inflated free space."""

    if len(path) <= 2:
        return path
    output = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        endpoint = anchor + 1
        for candidate in range(len(path) - 1, anchor, -1):
            cells = _line_cells(path[anchor], path[candidate])
            if all(traversable[cell] for cell in cells):
                endpoint = candidate
                break
        output.append(path[endpoint])
        anchor = endpoint
    return output


def make_scenario(
    path: Path,
    scale: str,
    *,
    resolution: float,
    robot_radius: float,
    minimum_cross_room_m: float,
) -> Scenario:
    data = json.loads(path.read_text(encoding="utf-8"))
    free, rooms, room_index = rasterize(data, resolution, robot_radius)
    clearance = distance_transform_edt(free)
    rng = np.random.default_rng(_stable_seed(path.stem))
    target_order = rng.permutation(len(rooms))
    selected = None
    for target_index in target_order:
        target_candidates = _central_candidates(rooms[int(target_index)], clearance)
        if not len(target_candidates):
            continue
        target_point = target_candidates[int(rng.integers(len(target_candidates)))]
        target = (int(target_point[0]), int(target_point[1]))
        distances, _ = weighted_distances(free, target)
        start_order = rng.permutation(len(rooms))
        for start_index in start_order:
            if int(start_index) == int(target_index):
                continue
            # Explicitly exclude overlapping room boxes for the selected cells.
            start_candidates = _central_candidates(rooms[int(start_index)], clearance)
            if not len(start_candidates):
                continue
            valid = np.asarray(
                [
                    np.isfinite(distances[tuple(point)])
                    and distances[tuple(point)] * resolution >= minimum_cross_room_m
                    and not rooms[int(target_index)].mask[tuple(point)]
                    and not rooms[int(start_index)].mask[target]
                    for point in start_candidates
                ]
            )
            start_candidates = start_candidates[valid]
            if not len(start_candidates):
                continue
            # Favor a distant point while retaining deterministic variation.
            candidate_distances = distances[start_candidates[:, 0], start_candidates[:, 1]]
            far = start_candidates[candidate_distances >= np.quantile(candidate_distances, 0.70)]
            point = far[int(rng.integers(len(far)))]
            start = (int(point[0]), int(point[1]))
            selected = int(start_index), int(target_index), start, target, float(distances[start])
            break
        if selected is not None:
            break
    if selected is None:
        raise ValueError("no sufficiently separated cross-room pair")

    start_index, target_index, start, target, shortest = selected
    context = np.zeros(free.shape, dtype=np.float64)
    context[rooms[target_index].mask] = 1.0
    distractor_indices = [i for i in rng.permutation(len(rooms)) if i != target_index][:2]
    for rank, index in enumerate(distractor_indices):
        context[rooms[int(index)].mask] = np.maximum(
            context[rooms[int(index)].mask], 0.40 - rank * 0.10
        )
    context += rng.normal(0.0, 0.035, size=context.shape)
    context[~free] = 0.0
    context = np.clip(context, 0.0, 1.0)
    bbox = data["bbox"]
    floor_area = (bbox["max"][0] - bbox["min"][0]) * (bbox["max"][1] - bbox["min"][1])
    return Scenario(
        scale=scale,
        house_id=path.stem,
        official_room_count=int(data["room_num"]),
        annotated_room_boxes=len(rooms),
        floor_area_m2=float(floor_area),
        free=free,
        room_index=room_index,
        start=start,
        target=target,
        start_room=start_index,
        target_room=target_index,
        start_room_labels=rooms[start_index].labels,
        target_room_labels=rooms[target_index].labels,
        context=context,
        shortest_cells=shortest,
        resolution_m=resolution,
    )


@lru_cache(maxsize=16)
def _ray_offsets(radius: int, rays: int = 180) -> tuple[tuple[tuple[int, int], ...], ...]:
    output = []
    for angle in np.linspace(0.0, 2.0 * np.pi, rays, endpoint=False):
        ray = []
        last = None
        for distance in np.linspace(0.0, radius, radius * 3 + 1):
            point = (int(round(math.sin(angle) * distance)), int(round(math.cos(angle) * distance)))
            if point != last:
                ray.append(point)
                last = point
        output.append(tuple(ray))
    return tuple(output)


def reveal_with_occlusion(
    free: np.ndarray,
    known: np.ndarray,
    last_seen: np.ndarray,
    position: tuple[int, int],
    timestamp: int,
    radius: int,
) -> np.ndarray:
    visible = np.zeros_like(free)
    for ray in _ray_offsets(radius):
        for dr, dc in ray:
            row, col = position[0] + dr, position[1] + dc
            if not (0 <= row < free.shape[0] and 0 <= col < free.shape[1]):
                break
            visible[row, col] = True
            if not free[row, col]:
                break
    known[visible] = np.where(free[visible], 1, -1)
    last_seen[visible & free] = timestamp
    return visible


def run_scenario(
    scenario: Scenario,
    policy: str,
    *,
    budget_m: float,
    sensor_range_m: float,
    semantic_rate: float,
    keep_trace: bool = False,
) -> tuple[CrossRoomResult, dict[str, object] | None]:
    sensor_radius = max(2, int(round(sensor_range_m / scenario.resolution_m)))
    known = np.zeros(scenario.free.shape, dtype=np.int8)
    last_seen = np.zeros(scenario.free.shape, dtype=np.float64)
    position = scenario.start
    continuous_position = np.asarray(position, dtype=np.float64)
    trajectory = [continuous_position.copy()]
    visited_rooms: set[int] = set()
    if scenario.room_index[position] >= 0:
        visited_rooms.add(int(scenario.room_index[position]))
    replans = actions = 0
    traveled_m = 0.0
    max_actions = int(math.ceil(budget_m / scenario.resolution_m)) * 20
    scan_spacing_m = min(1.0, sensor_range_m / 2.0)
    success = False
    policy_rng = np.random.default_rng(_stable_seed(f"{scenario.house_id}:{policy}"))

    while traveled_m + 1e-9 < budget_m and actions < max_actions:
        actions += 1
        visible = reveal_with_occlusion(
            scenario.free, known, last_seen, position, actions, sensor_radius
        )
        if visible[scenario.target]:
            success = True
            break
        distances, parents = weighted_distances(known == 1, position)
        frontiers = frontier_mask(known, np.isfinite(distances))

        # Match the production semantic path: only observed feature evidence is
        # projected, maximum-filtered into nearby cells, then sigmoid-normalized.
        observed_context = np.where(known == 1, scenario.context, 0.0)
        projected_context = maximum_filter(observed_context, size=5)
        semantic_scores = 1.0 / (1.0 + np.exp(-10.0 * (projected_context - 0.01)))
        goal = choose_goal(
            policy,
            frontiers,
            position,
            last_seen,
            semantic_scores,
            policy_rng,
            semantic_rate,
        )
        if goal is None:
            break
        path = weighted_approach_path(parents, distances, position, goal)
        if len(path) <= 1:
            # The selected unknown cell borders the current reachable region.
            # Reveal it as an in-place sensor turn; this prevents repeatedly
            # selecting an occluded frontier without adding fictitious travel.
            known[goal] = 1 if scenario.free[goal] else -1
            if scenario.free[goal]:
                last_seen[goal] = actions
            continue
        replans += 1
        smoothed_path = shortcut_path(path, known == 1)
        for waypoint in smoothed_path[1:]:
            segment_start = continuous_position.copy()
            segment_end = np.asarray(waypoint, dtype=np.float64)
            segment_cells = float(np.linalg.norm(segment_end - segment_start))
            segment_m = segment_cells * scenario.resolution_m
            if segment_m <= 1e-12:
                continue
            remaining_m = budget_m - traveled_m
            fraction = min(1.0, remaining_m / segment_m)
            actual_end = segment_start + fraction * (segment_end - segment_start)
            actual_m = float(np.linalg.norm(actual_end - segment_start)) * scenario.resolution_m
            samples = max(1, int(math.ceil(actual_m / scan_spacing_m)))
            previous_cell = position
            for sample_index in range(1, samples + 1):
                next_continuous = segment_start + (sample_index / samples) * (
                    actual_end - segment_start
                )
                increment_m = (
                    float(np.linalg.norm(next_continuous - continuous_position))
                    * scenario.resolution_m
                )
                traveled_m += increment_m
                continuous_position = next_continuous
                position = (
                    int(round(continuous_position[0])),
                    int(round(continuous_position[1])),
                )
                trajectory.append(continuous_position.copy())
                for crossed_cell in _line_cells(previous_cell, position):
                    room = int(scenario.room_index[crossed_cell])
                    if room >= 0:
                        visited_rooms.add(room)
                previous_cell = position
                actions += 1
                visible = reveal_with_occlusion(
                    scenario.free, known, last_seen, position, actions, sensor_radius
                )
                if visible[scenario.target]:
                    success = True
                    break
            if success or fraction < 1.0:
                break
        if success:
            break

    path_m = traveled_m
    shortest_m = scenario.shortest_cells * scenario.resolution_m
    spl = int(success) * shortest_m / max(shortest_m, path_m, 1e-9)
    result = CrossRoomResult(
        scale=scenario.scale,
        house_id=scenario.house_id,
        official_rooms=scenario.official_room_count,
        annotated_rooms=scenario.annotated_room_boxes,
        floor_area_m2=scenario.floor_area_m2,
        start_room="/".join(scenario.start_room_labels),
        target_room="/".join(scenario.target_room_labels),
        policy=policy,
        success=int(success),
        path_m=path_m,
        shortest_path_m=shortest_m,
        spl=float(spl),
        rooms_visited=len(visited_rooms),
        replans=replans,
        budget_m=budget_m,
    )
    trace = None
    if keep_trace:
        trace = {
            "scenario": scenario,
            "trajectory": np.asarray(trajectory),
            "known": known,
        }
    return result, trace


def select_scenarios(
    json_root: Path,
    test_ids: list[str],
    *,
    trials_per_scale: int,
    resolution: float,
    robot_radius: float,
    minimum_cross_room_m: float,
) -> list[Scenario]:
    rng = np.random.default_rng(20260802)
    ids = list(test_ids)
    rng.shuffle(ids)
    scenarios: list[Scenario] = []
    counts = {scale: 0 for scale in SCALE_RANGES}
    for house_id in ids:
        path = json_root / f"{house_id}.json"
        if not path.exists():
            continue
        try:
            room_count = int(json.loads(path.read_text(encoding="utf-8"))["room_num"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        scale = next(
            (name for name, (low, high) in SCALE_RANGES.items() if low <= room_count <= high),
            None,
        )
        if scale is None or counts[scale] >= trials_per_scale:
            continue
        try:
            scenario = make_scenario(
                path,
                scale,
                resolution=resolution,
                robot_radius=robot_radius,
                minimum_cross_room_m=minimum_cross_room_m,
            )
        except (ValueError, cv2.error):
            continue
        scenarios.append(scenario)
        counts[scale] += 1
        if all(value >= trials_per_scale for value in counts.values()):
            break
    if any(value < trials_per_scale for value in counts.values()):
        raise RuntimeError(f"Could not select requested HouseExpo scenarios: {counts}")
    return scenarios


def _mean_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    half = 0.0 if len(array) < 2 else 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array))
    return float(np.mean(array)), half


def summarize(results: list[CrossRoomResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scale in (*SCALE_RANGES, "all"):
        for policy in POLICIES:
            subset = [
                result
                for result in results
                if result.policy == policy and (scale == "all" or result.scale == scale)
            ]
            successes = sum(result.success for result in subset)
            low, high = wilson(successes, len(subset))
            spl_mean, spl_ci = _mean_ci([result.spl for result in subset])
            rooms_mean, rooms_ci = _mean_ci([result.rooms_visited for result in subset])
            path_success = [result.path_m for result in subset if result.success]
            rows.append(
                {
                    "scale": scale,
                    "policy": policy,
                    "episodes": len(subset),
                    "success_rate": successes / len(subset),
                    "success_ci_low": low,
                    "success_ci_high": high,
                    "spl_mean": spl_mean,
                    "spl_ci95": spl_ci,
                    "median_path_m_success": float(np.median(path_success)) if path_success else float("nan"),
                    "mean_rooms_visited": rooms_mean,
                    "mean_rooms_visited_ci95": rooms_ci,
                    "mean_floor_area_m2": float(np.mean([result.floor_area_m2 for result in subset])),
                }
            )
    return rows


def paired_tests(results: list[CrossRoomResult], comparison: str = "nearest") -> dict[str, object]:
    indexed = {(result.house_id, result.policy): result for result in results}
    houses = sorted({result.house_id for result in results})
    dream_only = baseline_only = 0
    baseline_costs, dream_costs = [], []
    for house_id in houses:
        baseline = indexed[(house_id, comparison)]
        dream = indexed[(house_id, "dream")]
        dream_only += int(dream.success and not baseline.success)
        baseline_only += int(baseline.success and not dream.success)
        failure_cost = baseline.budget_m + baseline.shortest_path_m
        baseline_costs.append(baseline.path_m if baseline.success else failure_cost)
        dream_costs.append(dream.path_m if dream.success else failure_cost)
    discordant = dream_only + baseline_only
    mcnemar = float(binomtest(dream_only, discordant, 0.5).pvalue) if discordant else 1.0
    statistic, path_p = wilcoxon(baseline_costs, dream_costs, alternative="two-sided")
    return {
        "comparison": f"dream_vs_{comparison}",
        "paired_floorplans": len(houses),
        "dream_only_successes": dream_only,
        "baseline_only_successes": baseline_only,
        "exact_mcnemar_p": mcnemar,
        "penalized_path_wilcoxon_statistic": float(statistic),
        "penalized_path_two_sided_p": float(path_p),
    }


def write_csv(path: Path, rows: list[object]) -> None:
    dictionaries = [asdict(row) if hasattr(row, "__dataclass_fields__") else row for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def plot_results(summary: list[dict[str, object]], traces: dict[str, dict[str, object]], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), constrained_layout=True)
    for axis, scale in zip(axes[0], SCALE_RANGES):
        trace = traces[scale]
        scenario: Scenario = trace["scenario"]
        trajectory = np.asarray(trace["trajectory"])
        axis.imshow(~scenario.free, cmap="gray_r", vmin=0, vmax=1)
        room_overlay = np.ma.masked_where(scenario.room_index < 0, scenario.room_index)
        axis.imshow(room_overlay, cmap="Pastel1", alpha=0.30, interpolation="nearest")
        axis.plot(trajectory[:, 1], trajectory[:, 0], color="#168aad", linewidth=1.3)
        axis.scatter(scenario.start[1], scenario.start[0], c="#52b788", s=28, edgecolor="black", label="start")
        axis.scatter(scenario.target[1], scenario.target[0], c="#e63946", marker="*", s=65, edgecolor="black", label="target")
        axis.set_title(f"{scale}: {scenario.floor_area_m2:.0f} m$^2$")
        axis.set_xticks([])
        axis.set_yticks([])
    policies = ("history_only", "nearest", "dream")
    colors = ("#9e9e9e", "#607d8b", "#2a9d8f")
    labels = {"history_only": "No semantics", "nearest": "Nearest", "dream": "DREAM"}
    scale_names = tuple(SCALE_RANGES)
    tick_labels = ("4--6", "7--9", "10+")
    for axis, metric, ylabel, ylim in zip(
        axes[1],
        ("success_rate", "spl_mean", "mean_rooms_visited"),
        ("Cross-room success", "SPL", "Rooms visited"),
        ((0, 1.05), (0, 1.0), (0, None)),
    ):
        x = np.arange(len(scale_names), dtype=float)
        width = 0.25
        for policy_index, (policy, color) in enumerate(zip(policies, colors)):
            policy_rows = {
                str(row["scale"]): row for row in summary if row["policy"] == policy
            }
            values = np.asarray([float(policy_rows[scale][metric]) for scale in scale_names])
            if metric == "success_rate":
                lower = values - np.asarray(
                    [float(policy_rows[scale]["success_ci_low"]) for scale in scale_names]
                )
                upper = np.asarray(
                    [float(policy_rows[scale]["success_ci_high"]) for scale in scale_names]
                ) - values
                errors = np.vstack((lower, upper))
            else:
                error_key = "spl_ci95" if metric == "spl_mean" else "mean_rooms_visited_ci95"
                errors = np.asarray([float(policy_rows[scale][error_key]) for scale in scale_names])
            offset = (policy_index - 1.0) * width
            axis.bar(
                x + offset,
                values,
                width,
                yerr=errors,
                capsize=2,
                color=color,
                label=labels[policy],
            )
        axis.set_xticks(x, tick_labels)
        axis.set_xlabel("Official room count")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.2)
    axes[1, 0].legend(frameon=False, ncol=3, loc="lower left", fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=240)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data/houseexpo"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/houseexpo_cross_room"))
    parser.add_argument("--trials-per-scale", type=int, default=100)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--robot-radius", type=float, default=0.20)
    parser.add_argument("--sensor-range", type=float, default=2.5)
    parser.add_argument("--minimum-cross-room", type=float, default=6.0)
    parser.add_argument("--budget-m", type=float, default=80.0)
    parser.add_argument("--semantic-rate", type=float, default=0.1)
    args = parser.parse_args()

    json_root, test_ids = ensure_houseexpo(args.data_root)
    scenarios = select_scenarios(
        json_root,
        test_ids,
        trials_per_scale=args.trials_per_scale,
        resolution=args.resolution,
        robot_radius=args.robot_radius,
        minimum_cross_room_m=args.minimum_cross_room,
    )
    results: list[CrossRoomResult] = []
    traces: dict[str, dict[str, object]] = {}
    trace_scores: dict[str, tuple[float, ...]] = {}
    for index, scenario in enumerate(scenarios):
        for policy in POLICIES:
            result, trace = run_scenario(
                scenario,
                policy,
                budget_m=args.budget_m,
                sensor_range_m=args.sensor_range,
                semantic_rate=args.semantic_rate,
                keep_trace=policy == "dream",
            )
            results.append(result)
            score = (
                float(result.success),
                float(min(result.rooms_visited, 6)),
                float(min(result.replans, 10)),
                -abs(float(result.path_m) - 30.0),
            )
            if trace is not None and score > trace_scores.get(scenario.scale, (-1.0,)):
                traces[scenario.scale] = trace
                trace_scores[scenario.scale] = score
        if (index + 1) % 25 == 0:
            print(f"Finished {index + 1}/{len(scenarios)} floor plans", flush=True)

    summary = summarize(results)
    tests = {
        f"dream_vs_{comparison}": paired_tests(results, comparison)
        for comparison in ("random", "nearest", "history_only")
    }
    write_csv(args.output_root / "episodes.csv", results)
    write_csv(args.output_root / "summary.csv", summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "paired_tests.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    metadata = vars(args) | {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "scale_ranges": SCALE_RANGES,
        "policies": POLICIES,
        "houseexpo_commit": HOUSEEXPO_COMMIT,
        "dream_commit": DREAM_COMMIT,
        "houseexpo_archive": HOUSEEXPO_ARCHIVE,
        "houseexpo_test_ids": HOUSEEXPO_TEST_IDS,
        "semantic_boundary": "noisy room-context proxy; scores projected only from visible cells",
        "motion_model": "eight-connected weighted Dijkstra; diagonal corner-cut prevention",
        "path_execution": "collision-checked line-of-sight shortcut in robot-radius-inflated known free space",
        "distance_accounting": "continuous Euclidean travel; sensing sampled at <=1 m intervals",
        "trace_selection": "successful DREAM trace; prefer up to 6 rooms and 10 replans, then path nearest 30 m",
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    plot_results(summary, traces, args.output_root / "cross_room_results")
    print(json.dumps([row for row in summary if row["scale"] == "all"], indent=2), flush=True)
    print(json.dumps(tests, indent=2), flush=True)


if __name__ == "__main__":
    main()
