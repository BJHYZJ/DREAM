"""Reproducible sensor-memory and online-navigation core for DREAM simulation.

The simulator adapter is intentionally thin: it supplies RGB-D, instance IDs,
camera calibration, and a traversability raster.  Target world pose is never
an input to the planner.  It is retained separately only for scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import maximum_filter


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dream.dynamic_memory import (  # noqa: E402
    DEFAULT_EXECUTION_CONFIG,
    DreamExecutionConfig,
    raycast_keep_mask,
)
from exploration_gridworld import choose_goal, frontier_mask  # noqa: E402
from houseexpo_cross_room import (  # noqa: E402
    reveal_with_occlusion,
    shortcut_path,
    weighted_approach_path,
    weighted_distances,
)


PROTOCOLS = ("static", "target_move", "obstacle_move", "pose_correction", "combined")
VARIANTS = ("full", "no_clearing", "no_focus", "no_rmp", "dynamic_off")
SCENE_FAMILIES = ("architecthor", "replicacad")


@dataclass(frozen=True)
class SimulationSpec:
    scene_family: str
    scene_id: str
    seed: int
    protocol: str
    variant: str
    sensor_range_m: float = 2.5
    resolution_m: float = 0.08
    semantic_rate: float = 0.10

    def __post_init__(self) -> None:
        if self.scene_family not in SCENE_FAMILIES:
            raise ValueError(f"unknown scene family: {self.scene_family}")
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol: {self.protocol}")
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant: {self.variant}")
        if self.sensor_range_m <= 0 or self.resolution_m <= 0:
            raise ValueError("sensor range and map resolution must be positive")


@dataclass(frozen=True)
class MemoryUpdate:
    frame_id: int
    input_points: int
    cleared_points: int
    retained_points: int
    fused_points: int
    voxel_points: int


class SensorVoxelMemory:
    """Small CPU voxel memory driven only by rendered RGB-D and calibration."""

    def __init__(self, voxel_size_m: float = 0.05, max_points: int = 80_000):
        if voxel_size_m <= 0 or max_points <= 0:
            raise ValueError("voxel size and capacity must be positive")
        self.voxel_size_m = float(voxel_size_m)
        self.max_points = int(max_points)
        self.points = np.empty((0, 3), dtype=np.float32)
        self.rgb = np.empty((0, 3), dtype=np.uint8)
        self.instance_ids = np.empty((0,), dtype=np.int32)
        self.frame_ids = np.empty((0,), dtype=np.int32)

    def _voxelize_keep_latest(self) -> None:
        if not len(self.points):
            return
        keys = np.floor(self.points / self.voxel_size_m).astype(np.int32)
        # Reverse before unique so a newly observed sample wins over stale data
        # in the same voxel, then restore deterministic lexicographic order.
        _, reverse_indices = np.unique(keys[::-1], axis=0, return_index=True)
        keep = len(keys) - 1 - reverse_indices
        keep.sort()
        if len(keep) > self.max_points:
            keep = keep[-self.max_points :]
        self.points = self.points[keep]
        self.rgb = self.rgb[keep]
        self.instance_ids = self.instance_ids[keep]
        self.frame_ids = self.frame_ids[keep]

    def integrate(
        self,
        *,
        frame_id: int,
        points_world: np.ndarray,
        colors_rgb: np.ndarray,
        instance_ids: np.ndarray,
        depth_m: np.ndarray,
        intrinsics_cv: np.ndarray,
        camera_in_world_cv: np.ndarray,
        clearing: bool = True,
        execution: DreamExecutionConfig = DEFAULT_EXECUTION_CONFIG,
    ) -> MemoryUpdate:
        points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors_rgb).reshape(-1, 3).astype(np.uint8)
        instances = np.asarray(instance_ids).reshape(-1).astype(np.int32)
        if not (len(points) == len(colors) == len(instances)):
            raise ValueError("point, color, and instance arrays must have equal length")
        finite = np.all(np.isfinite(points), axis=1)
        points, colors, instances = points[finite], colors[finite], instances[finite]

        before = len(self.points)
        if clearing and before:
            keep = raycast_keep_mask(
                torch.from_numpy(self.points),
                torch.as_tensor(depth_m, dtype=torch.float32),
                torch.as_tensor(intrinsics_cv, dtype=torch.float32),
                torch.as_tensor(camera_in_world_cv, dtype=torch.float32),
                epsilon_m=execution.depth_clear_epsilon_m,
                min_distance_m=execution.depth_min_m,
                max_distance_m=execution.depth_max_m,
            ).cpu().numpy()
            self.points, self.rgb = self.points[keep], self.rgb[keep]
            self.instance_ids, self.frame_ids = self.instance_ids[keep], self.frame_ids[keep]
        retained = len(self.points)
        if len(points):
            self.points = np.concatenate((self.points, points))
            self.rgb = np.concatenate((self.rgb, colors))
            self.instance_ids = np.concatenate((self.instance_ids, instances))
            self.frame_ids = np.concatenate(
                (self.frame_ids, np.full(len(points), frame_id, dtype=np.int32))
            )
        self._voxelize_keep_latest()
        return MemoryUpdate(
            frame_id=int(frame_id),
            input_points=len(points),
            cleared_points=before - retained,
            retained_points=retained,
            fused_points=len(points),
            voxel_points=len(self.points),
        )

    def clear_region(self, center_xyz: np.ndarray, radius_m: float = 0.30) -> int:
        """Reject a stale semantic candidate without deleting unrelated geometry."""

        if not len(self.points):
            return 0
        center = np.asarray(center_xyz, dtype=np.float32)
        remove = np.linalg.norm(self.points[:, :2] - center[:2], axis=1) <= radius_m
        count = int(remove.sum())
        keep = ~remove
        self.points, self.rgb = self.points[keep], self.rgb[keep]
        self.instance_ids, self.frame_ids = self.instance_ids[keep], self.frame_ids[keep]
        return count

    def count_region(self, center_xyz: np.ndarray, radius_m: float = 0.30) -> int:
        """Count currently retained voxels in a horizontal audit region."""

        if not len(self.points):
            return 0
        center = np.asarray(center_xyz, dtype=np.float32)
        return int(
            np.count_nonzero(
                np.linalg.norm(self.points[:, :2] - center[:2], axis=1) <= radius_m
            )
        )

    def instance_centroid(self, instance_id: int, current_frame_only: bool = True) -> np.ndarray | None:
        mask = self.instance_ids == int(instance_id)
        if current_frame_only and len(self.frame_ids):
            mask &= self.frame_ids == self.frame_ids.max()
        if not np.any(mask):
            return None
        return np.median(self.points[mask], axis=0)


class OnlineFrontierPlanner:
    """DREAM frontier selection with 16-cell receding-horizon execution."""

    def __init__(
        self,
        free: np.ndarray,
        context: np.ndarray,
        *,
        resolution_m: float,
        sensor_range_m: float,
        semantic_rate: float = 0.10,
        seed: int = 0,
        execution: DreamExecutionConfig = DEFAULT_EXECUTION_CONFIG,
    ):
        self.free = np.asarray(free, dtype=bool)
        self.context = np.asarray(context, dtype=float)
        if self.free.shape != self.context.shape:
            raise ValueError("free and context maps must have equal shape")
        self.known = np.zeros(self.free.shape, dtype=np.int8)
        self.last_seen = np.zeros(self.free.shape, dtype=np.float64)
        self.resolution_m = float(resolution_m)
        self.sensor_radius_cells = max(2, int(round(sensor_range_m / resolution_m)))
        self.semantic_rate = float(semantic_rate)
        self.rng = np.random.default_rng(seed)
        self.execution = execution
        self.timestamp = 0
        self.replans = 0
        self.cached_frontier: tuple[int, int] | None = None

    def observe(self, position: tuple[int, int]) -> np.ndarray:
        self.timestamp += 1
        return reveal_with_occlusion(
            self.free,
            self.known,
            self.last_seen,
            position,
            self.timestamp,
            self.sensor_radius_cells,
        )

    def next_chunk(self, position: tuple[int, int]) -> list[tuple[int, int]]:
        distances, parents = weighted_distances(self.known == 1, position)
        frontiers = frontier_mask(self.known, np.isfinite(distances))
        observed_context = np.where(self.known == 1, self.context, 0.0)
        projected_context = maximum_filter(observed_context, size=5)
        semantic = 1.0 / (1.0 + np.exp(-10.0 * (projected_context - 0.01)))
        # DREAM caches the selected navigation goal while executing a
        # receding-horizon prefix.  Re-selecting a frontier after every short
        # prefix causes deterministic ping-pong between score ties and is not
        # faithful to RobotAgent._cached_navigation_goal.
        goal = self.cached_frontier
        if goal is None or not frontiers[goal]:
            goal = choose_goal(
                "dream",
                frontiers,
                position,
                self.last_seen,
                semantic,
                self.rng,
                self.semantic_rate,
            )
            self.cached_frontier = goal
        if goal is None:
            return []
        path = weighted_approach_path(parents, distances, position, goal)
        if len(path) <= 1:
            # A frontier next to the robot can be revealed by a sensor sweep.
            self.known[goal] = 1 if self.free[goal] else -1
            if self.free[goal]:
                self.last_seen[goal] = self.timestamp
            self.cached_frontier = None
            return []
        dense = path[: self.execution.navigation_step_num + 1]
        self.replans += 1
        return shortcut_path(dense, self.known == 1)


def sensor_frame_to_world(
    *,
    position_gl_mm: np.ndarray,
    rgb: np.ndarray,
    segmentation: np.ndarray,
    cam2world_gl: np.ndarray,
    stride: int = 4,
    min_depth_m: float = 0.01,
    max_depth_m: float = 2.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a ManiSkill camera frame into sampled world points and depth."""

    position = np.asarray(position_gl_mm, dtype=np.float32)
    if position.ndim == 4:
        position = position[0]
    image_rgb = np.asarray(rgb)
    if image_rgb.ndim == 4:
        image_rgb = image_rgb[0]
    image_rgb = image_rgb[..., :3]
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb * (255 if image_rgb.max() <= 1 else 1), 0, 255).astype(np.uint8)
    ids = np.asarray(segmentation)
    if ids.ndim == 4:
        ids = ids[0]
    ids = ids[..., 0]
    depth_m = -position[..., 2] / 1000.0
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    sample = np.zeros_like(valid)
    sample[::stride, ::stride] = True
    select = valid & sample
    local = position[..., :3] / 1000.0
    homogeneous = np.concatenate((local, np.ones((*local.shape[:2], 1), dtype=np.float32)), axis=-1)
    transform = np.asarray(cam2world_gl, dtype=np.float32)
    if transform.ndim == 3:
        transform = transform[0]
    world = homogeneous.reshape(-1, 4) @ transform.T
    world = world[:, :3].reshape(*local.shape[:2], 3)
    return world[select], image_rgb[select], ids[select], depth_m


def homogeneous_extrinsic(extrinsic_cv: np.ndarray) -> np.ndarray:
    """Normalize ManiSkill's batched 3x4/4x4 OpenCV extrinsic to one 4x4 matrix."""

    extrinsic = np.asarray(extrinsic_cv, dtype=np.float32)
    if extrinsic.ndim == 3:
        if len(extrinsic) != 1:
            raise ValueError("the single-environment adapter expects one camera extrinsic")
        extrinsic = extrinsic[0]
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack((extrinsic, np.asarray([0, 0, 0, 1], dtype=np.float32)))
    if extrinsic.shape != (4, 4):
        raise ValueError(f"camera extrinsic must be 3x4 or 4x4, got {extrinsic.shape}")
    return extrinsic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, spec: SimulationSpec, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": 1, "spec": asdict(spec), **payload}
    path.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
