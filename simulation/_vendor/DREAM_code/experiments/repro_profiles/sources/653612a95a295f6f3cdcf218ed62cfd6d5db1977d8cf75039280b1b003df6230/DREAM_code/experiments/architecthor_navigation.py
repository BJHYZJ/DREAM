#!/usr/bin/env python3
"""Derive conservative Fetch navigation grids from official ArchitecTHOR assets.

The ManiSkill AI2THOR archive does not ship Fetch navigable-position files for
the ten ArchitecTHOR scenes.  This module therefore rasterizes walkable stage
triangles and subtracts projected wall and furniture geometry.  It never
modifies the simulator scene and records the derivation parameters so the same
grid can be reproduced independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh
import transforms3d

from houseexpo_cross_room import Scenario, weighted_distances
from maniskill_replicacad_search_video import NavGrid


SCENE_IDS = (
    "ArchitecTHOR-Test-03",
    "ArchitecTHOR-Val-02",
    "ArchitecTHOR-Val-01",
    "ArchitecTHOR-Test-02",
    "ArchitecTHOR-Val-04",
    "ArchitecTHOR-Val-03",
    "ArchitecTHOR-Val-00",
    "ArchitecTHOR-Test-01",
    "ArchitecTHOR-Test-04",
    "ArchitecTHOR-Test-00",
)

FETCH_START_XY = (
    (-3.0, 0.0),
    (-2.0, -2.0),
    (0.0, 0.0),
    (-3.5, 0.0),
    (0.0, -2.0),
    (-1.0, 1.5),
    (1.0, -0.5),
    (3.25, 1.0),
    (1.0, 2.0),
    (1.0, 1.0),
)


@dataclass(frozen=True)
class DerivedGridReport:
    scene_id: str
    resolution_m: float
    robot_clearance_m: float
    floor_triangle_count: int
    blocking_stage_triangle_count: int
    blocking_object_count: int
    component_area_m2: float
    component_span_x_m: float
    component_span_y_m: float
    requested_spawn_xy_m: tuple[float, float]
    raster_spawn_xy_m: tuple[float, float]


@dataclass(frozen=True)
class EpisodeLayout:
    initial_target_xy: np.ndarray
    final_target_xy: np.ndarray
    place_base_xy: np.ndarray
    search_scenario: Scenario
    relocation_distance_m: float
    shortest_path_to_target_m: float


def _mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"mesh contains no geometry: {path}")
        # ``Scene.geometry.values()`` are node-local meshes; concatenating them
        # directly drops GLB scene-graph transforms and can create wildly
        # incorrect furniture footprints.  ``to_geometry`` applies every node
        # transform before concatenation.
        return loaded.to_geometry()
    return loaded


def _world_stage_mesh(path: Path) -> trimesh.Trimesh:
    mesh = _mesh(path)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
    return mesh


def _object_world_vertices(asset: Path, instance: dict) -> np.ndarray:
    vertices = np.asarray(_mesh(asset).vertices, dtype=np.float64)
    q_up = transforms3d.quaternions.axangle2quat(np.asarray([1, 0, 0]), math.pi / 2)
    q_instance = np.asarray(instance["rotation"], dtype=np.float64)
    q = transforms3d.quaternions.qmult(q_up, q_instance)
    rotation = transforms3d.quaternions.quat2mat(q)
    translation = np.asarray(
        [
            instance["translation"][0],
            -instance["translation"][2],
            instance["translation"][1],
        ],
        dtype=np.float64,
    )
    return vertices @ rotation.T + translation


def derive_navigation_grid(
    asset_root: Path,
    scene_id: str,
    build_config_idx: int,
    *,
    resolution_m: float = 0.08,
    robot_clearance_m: float = 0.30,
    padding_cells: int = 5,
) -> tuple[NavGrid, DerivedGridReport]:
    """Build a floor/wall/furniture raster from the exact loaded GLB files."""

    config_path = (
        asset_root
        / "ai2thor-hab/configs/scenes/ArchitecTHOR"
        / f"{scene_id}.scene_instance.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage_path = asset_root / "ai2thor-hab/assets" / (
        config["stage_instance"]["template_name"] + ".glb"
    )
    stage = _world_stage_mesh(stage_path)
    triangles = np.asarray(stage.triangles, dtype=np.float64)
    centroids = triangles.mean(axis=1)
    normals = np.asarray(stage.face_normals, dtype=np.float64)

    floor_selector = (
        (normals[:, 2] > 0.75)
        & (centroids[:, 2] > -0.10)
        & (centroids[:, 2] < 0.12)
        & (triangles[:, :, 2].max(axis=1) < 0.14)
    )
    floors = triangles[floor_selector]
    if not len(floors):
        raise RuntimeError(f"no ground-floor triangles found for {scene_id}")
    floor_xy = floors[:, :, :2].reshape(-1, 2)
    minimum = floor_xy.min(axis=0) - padding_cells * resolution_m
    maximum = floor_xy.max(axis=0) + padding_cells * resolution_m
    width_height = np.ceil((maximum - minimum) / resolution_m).astype(int) + 1
    floor = np.zeros((int(width_height[1]), int(width_height[0])), dtype=np.uint8)
    obstacles = np.zeros_like(floor)

    def pixels(xy: np.ndarray) -> np.ndarray:
        result = np.empty_like(xy, dtype=np.float64)
        result[:, 0] = (xy[:, 0] - minimum[0]) / resolution_m
        result[:, 1] = (maximum[1] - xy[:, 1]) / resolution_m
        return np.rint(result).astype(np.int32)

    for triangle in floors:
        cv2.fillConvexPoly(floor, pixels(triangle[:, :2]), 1)

    wall_selector = (
        (normals[:, 2] < 0.75)
        & (triangles[:, :, 2].max(axis=1) > 0.08)
        & (triangles[:, :, 2].min(axis=1) < 1.25)
    )
    blocking_stage = triangles[wall_selector]
    for triangle in blocking_stage:
        cv2.fillConvexPoly(obstacles, pixels(triangle[:, :2]), 1)

    blocking_objects = 0
    object_asset_root = asset_root / "ai2thorhab-uncompressed/assets"
    for instance in config["object_instances"]:
        asset = object_asset_root / (instance["template_name"] + ".glb")
        vertices = _object_world_vertices(asset, instance)
        # Floor coverings and ceiling-only fixtures do not constrain Fetch.
        if float(vertices[:, 2].max()) <= 0.10 or float(vertices[:, 2].min()) >= 1.25:
            continue
        hull = cv2.convexHull(pixels(vertices[:, :2]))
        if len(hull) >= 3:
            cv2.fillConvexPoly(obstacles, hull, 1)
            blocking_objects += 1

    radius_cells = int(math.ceil(robot_clearance_m / resolution_m))
    kernel_size = radius_cells * 2 + 1
    expanded = cv2.dilate(
        obstacles,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    free = (floor > 0) & (expanded == 0)
    provisional = NavGrid(free, minimum, maximum, resolution_m, 0)
    requested_spawn = np.asarray(FETCH_START_XY[build_config_idx], dtype=np.float64)
    all_free = np.column_stack(np.where(free))
    if not len(all_free):
        raise RuntimeError(f"derived navigation grid is empty for {scene_id}")
    requested_cell = np.asarray(provisional.world_to_cell(requested_spawn))
    spawn_cell = all_free[
        int(np.argmin(np.linalg.norm(all_free - requested_cell[None, :], axis=1)))
    ]
    count, labels, _, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), 4)
    label = int(labels[tuple(spawn_cell)]) if count > 1 else 0
    if label <= 0:
        raise RuntimeError(f"cannot locate spawn component for {scene_id}")
    component = labels == label
    grid = NavGrid(component, minimum, maximum, resolution_m, 0)
    cells = np.column_stack(np.where(component))
    span = (cells.max(axis=0) - cells.min(axis=0)) * resolution_m
    raster_spawn = grid.cell_to_world(tuple(spawn_cell))
    report = DerivedGridReport(
        scene_id=scene_id,
        resolution_m=resolution_m,
        robot_clearance_m=robot_clearance_m,
        floor_triangle_count=int(len(floors)),
        blocking_stage_triangle_count=int(len(blocking_stage)),
        blocking_object_count=blocking_objects,
        component_area_m2=float(len(cells) * resolution_m**2),
        component_span_x_m=float(span[1]),
        component_span_y_m=float(span[0]),
        requested_spawn_xy_m=tuple(requested_spawn.tolist()),
        raster_spawn_xy_m=tuple(raster_spawn.tolist()),
    )
    return grid, report


def _free_at(grid: NavGrid, xy: np.ndarray) -> bool:
    row, col = grid.world_to_cell(xy)
    return 0 <= row < grid.free.shape[0] and 0 <= col < grid.free.shape[1] and bool(grid.free[row, col])


def _free_segment(grid: NavGrid, start: np.ndarray, end: np.ndarray) -> bool:
    distance = float(np.linalg.norm(end - start))
    for alpha in np.linspace(0.0, 1.0, max(2, int(math.ceil(distance / (grid.resolution_m / 2))))):
        if not _free_at(grid, (1 - alpha) * start + alpha * end):
            return False
    return True


def choose_episode_layout(
    grid: NavGrid,
    build_config_idx: int,
    *,
    relocation_distance_m: float = 0.85,
    layout_index: int = 0,
    minimum_layout_separation_m: float = 1.0,
    place_near_target: bool = False,
) -> EpisodeLayout:
    """Choose a deterministic, spatially distinct dynamic episode layout."""

    if layout_index < 0:
        raise ValueError("layout_index must be non-negative")

    cells = np.column_stack(np.where(grid.free))
    requested_start = np.asarray(FETCH_START_XY[build_config_idx], dtype=np.float64)
    start = min(cells, key=lambda cell: np.linalg.norm(cell - np.asarray(grid.world_to_cell(requested_start))))
    distances, _ = weighted_distances(grid.free, tuple(start))
    finite_cells = cells[np.isfinite(distances[grid.free])]
    order = np.argsort(distances[finite_cells[:, 0], finite_cells[:, 1]])[::-1]
    physical_offset = np.asarray([0.049, -0.023])
    pick_yaw = math.atan2(0.48, -0.96)
    rotation = np.asarray(
        [[math.cos(pick_yaw), -math.sin(pick_yaw)], [math.sin(pick_yaw), math.cos(pick_yaw)]]
    )
    dock_from_physical_target = -(rotation @ np.asarray([0.550, 0.038]))
    dock_direction = dock_from_physical_target / np.linalg.norm(dock_from_physical_target)
    perpendicular = np.asarray([-dock_direction[1], dock_direction[0]])
    start_world = grid.cell_to_world(tuple(start))
    accepted_targets: list[np.ndarray] = []

    # Subsample far cells at roughly 16 cm to avoid repeating equivalent tests.
    for initial_cell in finite_cells[order][::2]:
        initial_physical = grid.cell_to_world(tuple(initial_cell))
        if float(distances[tuple(initial_cell)]) < 3.0 / grid.resolution_m:
            break
        # Reject targets connected to the spawn by a direct traversable
        # segment.  This forces a turn around a wall/furniture boundary and is
        # the geometric definition used here for the cross-region task.
        if _free_segment(grid, start_world, initial_physical):
            continue
        if any(
            np.linalg.norm(initial_physical - prior) < minimum_layout_separation_m
            for prior in accepted_targets
        ):
            continue
        selected_for_initial: EpisodeLayout | None = None
        nominal_arrival = initial_physical - start_world
        nominal_arrival /= max(float(np.linalg.norm(nominal_arrival)), 1e-9)
        for theta in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            relocation_direction = np.asarray([math.cos(theta), math.sin(theta)])
            # Search normally reaches a far target from the spawn side.  Move
            # the furniture further along that direction, so its physical
            # trajectory does not run through the observing robot.
            if float(relocation_direction @ nominal_arrival) < 0.35:
                continue
            final_physical = initial_physical + relocation_distance_m * relocation_direction
            dock = final_physical + dock_from_physical_target
            def dock_arc(side: float) -> np.ndarray:
                angles = np.linspace(side * math.pi / 2, 0.0, 7)
                radii = np.linspace(0.70, np.linalg.norm(dock_from_physical_target), 7)
                return np.asarray(
                    [
                        final_physical
                        + radius
                        * (math.cos(angle) * dock_direction + math.sin(angle) * perpendicular)
                        for angle, radius in zip(angles, radii)
                    ]
                )

            arc_a, arc_b = dock_arc(1.0), dock_arc(-1.0)
            arc_a_free = all(_free_at(grid, point) for point in arc_a) and all(
                _free_segment(grid, first, second)
                for first, second in zip(arc_a[:-1], arc_a[1:])
            )
            arc_b_free = all(_free_at(grid, point) for point in arc_b) and all(
                _free_segment(grid, first, second)
                for first, second in zip(arc_b[:-1], arc_b[1:])
            )
            if not (
                _free_at(grid, final_physical)
                and _free_at(grid, dock)
                and _free_segment(grid, initial_physical, final_physical)
                and (arc_a_free or arc_b_free)
            ):
                continue
            initial_logical = initial_physical - physical_offset
            final_logical = final_physical - physical_offset

            # Receptacle center is 0.696 m behind a pi-facing placement base.
            if place_near_target:
                placement_cost = np.asarray(
                    [
                        np.linalg.norm(grid.cell_to_world(tuple(cell)) - final_physical)
                        for cell in finite_cells
                    ]
                )
            else:
                placement_cost = distances[
                    finite_cells[:, 0], finite_cells[:, 1]
                ]
            for place_cell in finite_cells[np.argsort(placement_cost)]:
                place = grid.cell_to_world(tuple(place_cell))
                tray = place + np.asarray([-0.696, 0.009])
                if np.linalg.norm(place - final_physical) > 1.4 and _free_at(grid, tray):
                    rows, cols = np.indices(grid.free.shape)
                    world_x = grid.minimum_xy[0] + cols * grid.resolution_m
                    world_y = grid.maximum_xy[1] - rows * grid.resolution_m
                    context = (
                        (world_x - initial_physical[0]) ** 2
                        + (world_y - initial_physical[1]) ** 2
                        < 1.8**2
                    ).astype(np.float64)
                    scenario = Scenario(
                        scale="ArchitecTHOR",
                        house_id=SCENE_IDS[build_config_idx],
                        official_room_count=0,
                        annotated_room_boxes=0,
                        floor_area_m2=float(np.count_nonzero(grid.free) * grid.resolution_m**2),
                        free=grid.free,
                        room_index=np.full(grid.free.shape, -1, dtype=np.int16),
                        context=context,
                        start=tuple(int(v) for v in start),
                        target=tuple(int(v) for v in initial_cell),
                        start_room=-1,
                        target_room=-1,
                        start_room_labels=("spawn region",),
                        target_room_labels=("destination context proxy",),
                        shortest_cells=float(distances[tuple(initial_cell)]),
                        resolution_m=grid.resolution_m,
                    )
                    selected_for_initial = EpisodeLayout(
                        initial_target_xy=initial_logical,
                        final_target_xy=final_logical,
                        place_base_xy=place,
                        search_scenario=scenario,
                        relocation_distance_m=relocation_distance_m,
                        shortest_path_to_target_m=float(distances[tuple(initial_cell)] * grid.resolution_m),
                    )
                    break
            if selected_for_initial is not None:
                break
        if selected_for_initial is not None:
            if len(accepted_targets) == layout_index:
                return selected_for_initial
            accepted_targets.append(initial_physical)
    raise RuntimeError("could not choose a collision-clear dynamic episode layout")
