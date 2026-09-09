"""Pure policy helpers for DREAM's pose-graph memory maintenance.

The robot agent owns I/O and voxel-map mutations.  Keeping the scope selection
and observation-retention policy here makes those decisions independently
testable and allows the offline benchmarks to exercise exactly the same rules
as the deployed system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Collection, Iterable, Literal, Sequence


UpdateScope = Literal["none", "local", "regional", "global"]


@dataclass(frozen=True)
class RmpDecision:
    """Selected reintegration scope and candidate observation identifiers."""

    scope: UpdateScope
    observation_ids: tuple[int, ...]


def select_reintegration_scope(
    observation_ids: Sequence[int],
    shared_pose_graph_ids: Collection[int],
    affected_pose_graph_ids: Collection[int],
    *,
    realtime_window: int,
    short_term_window: int,
    enable_global: bool = True,
    enable_regional: bool = True,
    enable_local: bool = True,
    global_fraction: float = 2.0 / 3.0,
    regional_fraction: float = 1.0 / 3.0,
) -> RmpDecision:
    """Apply the production RMP trigger hierarchy without mutating state."""

    if realtime_window <= 0 or short_term_window <= 0:
        raise ValueError("RMP windows must be positive")
    if not 0.0 <= global_fraction <= 1.0 or not 0.0 <= regional_fraction <= 1.0:
        raise ValueError("RMP trigger fractions must be in [0, 1]")

    # The production cache is insertion ordered and therefore chronological.
    obs_ids = tuple(observation_ids)
    shared = set(shared_pose_graph_ids)
    affected = set(affected_pose_graph_ids)
    if not obs_ids or not affected:
        return RmpDecision("none", ())

    recent_ids = obs_ids[-realtime_window:]
    short_ids = obs_ids[-short_term_window:]
    has_recent_correction = bool(set(recent_ids) & affected)

    if (
        enable_global
        and len(obs_ids) > realtime_window
        and shared
        and len(affected) / len(shared) > global_fraction
    ):
        return RmpDecision("global", obs_ids)

    if (
        enable_regional
        and len(obs_ids) > short_term_window
        and len(set(short_ids) & affected) / short_term_window > regional_fraction
        and has_recent_correction
    ):
        return RmpDecision("regional", short_ids)

    if enable_local and has_recent_correction:
        return RmpDecision("local", recent_ids)

    return RmpDecision("none", ())


def select_observations_to_prune(
    observation_ids: Iterable[int],
    *,
    is_pose_graph_node: Callable[[int], bool],
    max_observations: int,
    max_pose_graph_observations: int,
) -> tuple[int, ...]:
    """Return oldest redundant observations selected by DREAM's pruning rule."""

    if max_observations <= 0 or max_pose_graph_observations < 0:
        raise ValueError("RMP capacities must be non-negative and max_observations positive")

    # The production cache is insertion ordered and therefore chronological.
    obs_ids = tuple(observation_ids)
    overflow = max(0, len(obs_ids) - max_observations)
    if overflow == 0:
        return ()

    keyframe_count = sum(bool(is_pose_graph_node(obs_id)) for obs_id in obs_ids)
    selected: list[int] = []
    for obs_id in obs_ids:
        if overflow <= 0:
            break
        keyframe = bool(is_pose_graph_node(obs_id))
        if not keyframe or keyframe_count > max_pose_graph_observations:
            selected.append(obs_id)
            overflow -= 1
            if keyframe:
                keyframe_count -= 1
    return tuple(selected)
