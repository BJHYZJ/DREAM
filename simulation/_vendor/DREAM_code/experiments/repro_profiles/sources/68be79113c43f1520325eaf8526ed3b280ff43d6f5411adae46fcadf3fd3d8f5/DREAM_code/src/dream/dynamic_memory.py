"""Sensor-driven dynamic-memory rules shared by DREAM and its simulators.

This module deliberately has no robot, mapping-package, or learned-perception
dependencies.  The deployed voxel map and controlled simulation therefore use
the same ray-consistency and target-relocation decisions without pulling the
full hardware stack into the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor


class FocusOutcome(str, Enum):
    """Result of checking a cached target in a newly captured observation."""

    KEEP = "keep"
    CLEAR_STALE = "clear_stale"
    RELOCATED = "relocated"


@dataclass(frozen=True)
class DreamExecutionConfig:
    """Published DREAM execution thresholds used by hardware and simulation."""

    initial_rotation_steps: int = 8
    navigation_step_num: int = 16
    focused_tracking_radius_m: float = 2.0
    target_reject_region_radius_m: float = 0.30
    target_relocation_radius_m: float = 0.50
    depth_clear_epsilon_m: float = 0.10
    depth_min_m: float = 0.01
    depth_max_m: float = 2.50

    def __post_init__(self) -> None:
        if self.initial_rotation_steps <= 0 or self.navigation_step_num <= 0:
            raise ValueError("rotation and navigation step counts must be positive")
        distances = (
            self.focused_tracking_radius_m,
            self.target_reject_region_radius_m,
            self.target_relocation_radius_m,
            self.depth_clear_epsilon_m,
            self.depth_min_m,
            self.depth_max_m,
        )
        if any(value <= 0 for value in distances):
            raise ValueError("DREAM distance thresholds must be positive")
        if self.depth_min_m >= self.depth_max_m:
            raise ValueError("depth_min_m must be smaller than depth_max_m")


DEFAULT_EXECUTION_CONFIG = DreamExecutionConfig()


def initial_scan_yaws(start_yaw: float = 0.0, steps: int = 8) -> tuple[float, ...]:
    """Return the eight 45-degree base headings used by ``rotate_in_place``."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    full_turn = 2.0 * torch.pi
    return tuple(float(start_yaw + full_turn * (index + 1) / steps) for index in range(steps))


def should_focus(robot_xy: Tensor, cached_target_xy: Tensor, radius_m: float = 2.0) -> bool:
    """Whether DREAM should replace a full scan with focused verification."""

    return bool(torch.linalg.vector_norm(cached_target_xy[:2] - robot_xy[:2]) <= radius_m)


def classify_focused_observation(
    cached_target_xyz: Tensor,
    detected_target_xyz: Tensor | None,
    *,
    observation_is_current: bool,
    relocation_radius_m: float = 0.50,
) -> FocusOutcome:
    """Classify a fresh target check without consulting simulator world state.

    A missing target only invalidates memory when a *new* image was captured.
    This mirrors the deployed guard against clearing valid memory after a
    dropped camera frame.
    """

    if detected_target_xyz is None:
        return FocusOutcome.CLEAR_STALE if observation_is_current else FocusOutcome.KEEP
    displacement = torch.linalg.vector_norm(detected_target_xyz[:2] - cached_target_xyz[:2])
    return FocusOutcome.RELOCATED if displacement > relocation_radius_m else FocusOutcome.KEEP


def project_world_points(points_world: Tensor, intrinsics: Tensor, camera_in_world: Tensor) -> tuple[Tensor, Tensor]:
    """Project world points and return image ``(row, column)`` plus camera depth."""

    points = torch.as_tensor(points_world)
    intrinsics = torch.as_tensor(intrinsics, dtype=points.dtype, device=points.device)
    pose = torch.as_tensor(camera_in_world, dtype=points.dtype, device=points.device)
    homogeneous = torch.cat(
        (points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)),
        dim=1,
    )
    camera_points = (torch.linalg.inv(pose) @ homogeneous.T).T[:, :3]
    pixels_h = (intrinsics @ camera_points.T).T
    columns_rows = pixels_h[:, :2] / pixels_h[:, 2:].clamp_min(torch.finfo(points.dtype).eps)
    return columns_rows[:, [1, 0]], camera_points[:, 2]


def raycast_keep_mask(
    points_world: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    camera_in_world: Tensor,
    *,
    epsilon_m: float = 0.10,
    min_distance_m: float = 0.01,
    max_distance_m: float = 2.50,
) -> Tensor:
    """Return old semantic-memory points that remain consistent with RGB-D.

    Old points that should be visible in valid range are removed before the new
    frame is integrated.  Points outside the image/range, behind an observed
    surface, or paired with invalid near-zero depth are retained.  This is the
    deployed DREAM ``clear_points`` rule expressed as a side-effect-free mask.
    """

    points = torch.as_tensor(points_world)
    depth_tensor = torch.as_tensor(depth, dtype=points.dtype, device=points.device)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_world must have shape (N, 3)")
    if depth_tensor.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    if len(points) == 0:
        return torch.ones(0, dtype=torch.bool, device=points.device)

    rows_columns, projected_depth = project_world_points(
        points, intrinsics, camera_in_world
    )
    pixels = rows_columns.to(torch.long)
    height, width = depth_tensor.shape
    outside = (
        (pixels[:, 0] < 0)
        | (pixels[:, 0] >= height)
        | (pixels[:, 1] < 0)
        | (pixels[:, 1] >= width)
    )
    safe_pixels = pixels.clone()
    safe_pixels[outside] = 0
    measured_depth = depth_tensor[safe_pixels[:, 0], safe_pixels[:, 1]]
    occluded = measured_depth < (projected_depth - epsilon_m)
    invalid_depth = measured_depth < min_distance_m
    outside_range = (projected_depth < min_distance_m) | (projected_depth > max_distance_m)
    return outside | occluded | invalid_depth | outside_range
