"""Pure frontier-scoring policy shared by DREAM and offline benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrontierDecision:
    """Selected grid cell and the score field used for the decision."""

    index: tuple[int, int] | None
    total_scores: np.ma.MaskedArray


def select_frontier_goal(
    time_scores: np.ndarray | np.ma.MaskedArray,
    semantic_scores: np.ndarray | np.ma.MaskedArray | None,
    frontier_mask: np.ndarray,
    robot_point: np.ndarray,
    *,
    semantic_rate: float = 0.1,
    min_distance_cells: float = 0.0,
    score_resolution: int = 200,
) -> FrontierDecision:
    """Select DREAM's maximum-scoring frontier with its distance tie-break.

    Quantization reproduces the deployed implementation: scores within 0.005
    are treated as ties and the farther frontier is selected to avoid short,
    low-information moves.
    """

    if semantic_rate < 0:
        raise ValueError("semantic_rate must be non-negative")
    if score_resolution <= 0:
        raise ValueError("score_resolution must be positive")

    frontier = np.asarray(frontier_mask, dtype=bool)
    time_values = np.ma.array(time_scores, copy=False)
    if time_values.shape != frontier.shape:
        raise ValueError("time_scores and frontier_mask must have the same shape")

    combined_mask = np.ma.getmaskarray(time_values) | ~frontier
    if semantic_scores is None:
        total = np.ma.array(np.asarray(time_values), mask=combined_mask)
    else:
        semantic_values = np.ma.array(semantic_scores, copy=False)
        if semantic_values.shape != frontier.shape:
            raise ValueError("semantic_scores and frontier_mask must have the same shape")
        combined_mask |= np.ma.getmaskarray(semantic_values)
        total = np.ma.array(
            np.asarray(time_values) + semantic_rate * np.asarray(semantic_values),
            mask=combined_mask,
        )

    rounded = np.ceil(np.ma.filled(total, -np.inf) * score_resolution) / score_resolution
    maximum = float(np.max(rounded))
    if not np.isfinite(maximum):
        return FrontierDecision(None, total)

    candidates = np.column_stack(np.where(rounded == maximum))
    robot = np.asarray(robot_point, dtype=float)
    distances = np.linalg.norm(candidates - robot, axis=-1)
    if min_distance_cells > 0:
        valid = distances >= min_distance_cells
        if np.any(valid):
            candidates = candidates[valid]
            distances = distances[valid]
    selected = candidates[int(np.argmax(distances))]
    return FrontierDecision((int(selected[0]), int(selected[1])), total)
