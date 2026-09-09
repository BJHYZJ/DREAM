#!/usr/bin/env python3
"""Visualize DREAM object-goal search in an actual ReplicaCAD apartment.

The script instantiates ManiSkill's ``ReplicaCAD_SceneManipulation-v1`` with
the Fetch embodiment, rasterizes the Fetch-specific navigable mesh, and runs
the released DREAM frontier selector from an initially unknown map.  A red
target is placed in the far side of apartment 0.  The semantic input is a
controlled destination-area context proxy exposed only in locally observed
cells, exactly as documented for the HouseExpo component test.

The simulator frames are a qualitative kinematic replay of the computed route:
the Fetch planar joints are set along collision-free navigable positions.  The
video therefore demonstrates integration, route geometry, and scene scale; it
is not a closed-loop dynamics, controller, or learned-perception benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

# Set headless defaults before importing SAPIEN/ManiSkill.  Callers can
# override every value explicitly in their shell.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MS_ASSET_DIR", str(WORKSPACE_ROOT / ".maniskill_assets"))
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien

import mani_skill  # noqa: F401  # registers ManiSkill environments with Gymnasium
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors

from houseexpo_cross_room import Scenario, run_scenario, weighted_distances
from maniskill_cross_room_video import _sha256, _uniform_frames


ENVIRONMENT_ID = "ReplicaCAD_SceneManipulation-v1"
SCENE_NAME = "apt_0"
DATASET_URL = "https://huggingface.co/datasets/haosulab/ReplicaCAD"
QUALITATIVE_BOUNDARY = (
    "DREAM plans from local observations on the Fetch-specific ReplicaCAD navmesh; "
    "ManiSkill/SAPIEN then kinematically replays the route. This is not a "
    "closed-loop dynamics, controller, or learned-perception benchmark."
)


@dataclass(frozen=True)
class NavGrid:
    """Metric conversion for a rasterized 2-D trimesh navigation surface."""

    free: np.ndarray
    minimum_xy: np.ndarray
    maximum_xy: np.ndarray
    resolution_m: float
    padding_cells: int

    def world_to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        row = int(
            round(
                (float(self.maximum_xy[1]) - float(xy[1])) / self.resolution_m
                + self.padding_cells
            )
        )
        col = int(
            round(
                (float(xy[0]) - float(self.minimum_xy[0])) / self.resolution_m
                + self.padding_cells
            )
        )
        return row, col

    def cell_to_world(self, point: np.ndarray | tuple[float, float]) -> np.ndarray:
        row, col = float(point[0]), float(point[1])
        return np.asarray(
            [
                (col - self.padding_cells) * self.resolution_m
                + float(self.minimum_xy[0]),
                float(self.maximum_xy[1])
                - (row - self.padding_cells) * self.resolution_m,
            ],
            dtype=np.float64,
        )


def rasterize_navigation_mesh(mesh, resolution_m: float, padding_cells: int = 4) -> NavGrid:
    """Rasterize every triangle of ManiSkill's Fetch-specific navmesh."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or not len(faces):
        raise ValueError("Expected a non-empty 2-D triangular navigation mesh")
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    width, height = (
        np.ceil((maximum - minimum) / resolution_m).astype(int)
        + 2 * padding_cells
        + 1
    )
    free = np.zeros((height, width), dtype=np.uint8)
    image_vertices = np.empty_like(vertices)
    image_vertices[:, 0] = (
        (vertices[:, 0] - minimum[0]) / resolution_m + padding_cells
    )
    image_vertices[:, 1] = (
        (maximum[1] - vertices[:, 1]) / resolution_m + padding_cells
    )
    for face in faces:
        cv2.fillConvexPoly(
            free,
            np.rint(image_vertices[face]).astype(np.int32),
            1,
        )

    # Anti-aliased triangle boundaries can leave isolated pixels.  Keep the
    # connected component containing the nominal Fetch spawn position.
    count, labels, _, _ = cv2.connectedComponentsWithStats(free, 4)
    provisional = NavGrid(
        free=free.astype(bool),
        minimum_xy=minimum,
        maximum_xy=maximum,
        resolution_m=resolution_m,
        padding_cells=padding_cells,
    )
    spawn = provisional.world_to_cell(np.asarray([-1.0, 0.0]))
    spawn_label = int(labels[spawn]) if count > 1 else 0
    if spawn_label <= 0:
        raise RuntimeError("ReplicaCAD Fetch spawn is outside the rasterized navmesh")
    return NavGrid(
        free=labels == spawn_label,
        minimum_xy=minimum,
        maximum_xy=maximum,
        resolution_m=resolution_m,
        padding_cells=padding_cells,
    )


def _nearest_free_cell(grid: NavGrid, desired_xy: np.ndarray) -> tuple[int, int]:
    cells = np.column_stack(np.where(grid.free))
    desired_cell = np.asarray(grid.world_to_cell(desired_xy))
    chosen = cells[int(np.argmin(np.linalg.norm(cells - desired_cell, axis=1)))]
    return int(chosen[0]), int(chosen[1])


def build_search_scenario(
    grid: NavGrid,
    *,
    target_xy: np.ndarray,
    destination_boundary_y_m: float,
) -> Scenario:
    """Create the deterministic cross-apartment object-search task.

    The context proxy is binary over the southern destination area.  It does
    not encode distance or a shortest path, and it becomes usable by the
    selector only after the corresponding cells have been observed.
    """

    start = _nearest_free_cell(grid, np.asarray([-1.0, 0.0]))
    target = _nearest_free_cell(grid, target_xy)
    distances, _ = weighted_distances(grid.free, start)
    if not np.isfinite(distances[target]):
        raise RuntimeError("Selected ReplicaCAD target is unreachable from the Fetch spawn")

    rows, cols = np.indices(grid.free.shape)
    world_y = grid.maximum_xy[1] - (
        rows - grid.padding_cells
    ) * grid.resolution_m
    context = np.zeros(grid.free.shape, dtype=np.float64)
    context[grid.free & (world_y <= destination_boundary_y_m)] = 1.0

    return Scenario(
        scale="ReplicaCAD apartment",
        house_id=SCENE_NAME,
        official_room_count=0,
        annotated_room_boxes=0,
        floor_area_m2=float(grid.free.sum()) * grid.resolution_m**2,
        free=grid.free,
        room_index=np.full(grid.free.shape, -1, dtype=np.int16),
        start=start,
        target=target,
        start_room=-1,
        target_room=-1,
        start_room_labels=("Fetch spawn area",),
        target_room_labels=("southern destination area",),
        context=context,
        shortest_cells=float(distances[target]),
        resolution_m=grid.resolution_m,
    )


def _trajectory_inset(
    scenario: Scenario,
    trajectory: np.ndarray,
    current_index: int,
    width: int,
) -> np.ndarray:
    canvas = np.full((*scenario.free.shape, 3), 235, dtype=np.uint8)
    canvas[~scenario.free] = (52, 58, 65)
    context = scenario.context > 0
    canvas[context] = (218, 230, 248)
    complete = np.rint(trajectory[: current_index + 1]).astype(np.int32)
    future = np.rint(trajectory[current_index:]).astype(np.int32)
    if len(future) > 1:
        cv2.polylines(canvas, [future[:, ::-1]], False, (155, 160, 166), 2, cv2.LINE_AA)
    if len(complete) > 1:
        cv2.polylines(canvas, [complete[:, ::-1]], False, (225, 139, 24), 3, cv2.LINE_AA)
    cv2.circle(canvas, (scenario.start[1], scenario.start[0]), 5, (70, 174, 95), -1, cv2.LINE_AA)
    cv2.drawMarker(
        canvas,
        (scenario.target[1], scenario.target[0]),
        (50, 55, 225),
        cv2.MARKER_STAR,
        14,
        2,
    )
    current = complete[-1]
    cv2.circle(canvas, (int(current[1]), int(current[0])), 4, (8, 8, 8), -1, cv2.LINE_AA)
    # The apartment is tall in raster coordinates; rotate the inset so it
    # remains readable without obscuring most of a landscape video frame.
    canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
    height = max(1, int(round(canvas.shape[0] * width / canvas.shape[1])))
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)


def _annotate(
    rgb: np.ndarray,
    scenario: Scenario,
    trajectory: np.ndarray,
    frame_index: int,
    route_m: float,
    replans: int,
) -> np.ndarray:
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 84), (17, 24, 30), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(
        frame,
        "DREAM object-goal search | actual ReplicaCAD apartment + Fetch",
        (22, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{ENVIRONMENT_ID} | route {route_m:.2f} m | {replans} replans",
        (22, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (202, 220, 229),
        1,
        cv2.LINE_AA,
    )
    inset = _trajectory_inset(
        scenario,
        trajectory,
        frame_index,
        width=min(255, frame.shape[1] // 3),
    )
    margin = 16
    x0 = frame.shape[1] - inset.shape[1] - margin
    y0 = frame.shape[0] - inset.shape[0] - margin
    if y0 > 90:
        frame[y0 : y0 + inset.shape[0], x0 : x0 + inset.shape[1]] = inset
        cv2.rectangle(
            frame,
            (x0 - 2, y0 - 2),
            (x0 + inset.shape[1] + 1, y0 + inset.shape[0] + 1),
            (248, 248, 248),
            2,
        )
    cv2.putText(
        frame,
        "QUALITATIVE KINEMATIC REPLAY | controlled context proxy",
        (18, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _add_target(base, target_xy: np.ndarray) -> None:
    actors.build_box(
        base.scene,
        half_sizes=[0.18, 0.18, 0.32],
        color=[0.78, 0.12, 0.12, 1.0],
        name="dream_target_pedestal",
        body_type="static",
        add_collision=False,
        initial_pose=sapien.Pose([float(target_xy[0]), float(target_xy[1]), 0.32]),
    )
    actors.build_sphere(
        base.scene,
        radius=0.18,
        color=[1.0, 0.04, 0.03, 1.0],
        name="dream_target_object",
        body_type="static",
        add_collision=False,
        initial_pose=sapien.Pose([float(target_xy[0]), float(target_xy[1]), 0.82]),
    )


def _write_route_csv(path: Path, trajectory: np.ndarray, grid: NavGrid) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "grid_row", "grid_col", "world_x_m", "world_y_m"))
        for index, point in enumerate(trajectory):
            world = grid.cell_to_world(point)
            writer.writerow((index, float(point[0]), float(point[1]), world[0], world[1]))


def render(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    env = gym.make(
        ENVIRONMENT_ID,
        robot_uids="fetch",
        build_config_idxs=[args.build_config_index],
        num_envs=1,
        obs_mode="none",
        reward_mode="none",
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        render_backend="cpu",
        human_render_camera_configs=dict(
            width=args.width,
            height=args.height,
            fov=1.05,
            near=0.05,
            far=100,
            shader_pack="default",
        ),
    )
    env.reset(seed=args.seed)
    base = env.unwrapped
    mesh = base.scene_builder.navigable_positions[0]
    if mesh is None:
        env.close()
        raise RuntimeError("Fetch-specific ReplicaCAD navigable mesh was not loaded")
    grid = rasterize_navigation_mesh(mesh, args.resolution)
    scenario = build_search_scenario(
        grid,
        target_xy=np.asarray([args.target_x, args.target_y]),
        destination_boundary_y_m=args.destination_boundary_y,
    )
    result, trace = run_scenario(
        scenario,
        "dream",
        budget_m=args.budget_m,
        sensor_range_m=args.sensor_range,
        semantic_rate=args.semantic_rate,
        keep_trace=True,
    )
    if not result.success or trace is None:
        env.close()
        raise RuntimeError("DREAM did not find the ReplicaCAD target within the travel budget")

    raw_trajectory = np.asarray(trace["trajectory"], dtype=np.float64)
    trajectory = _uniform_frames(raw_trajectory, args.frames)
    world_trajectory = np.asarray([grid.cell_to_world(point) for point in trajectory])
    target_world = grid.cell_to_world(scenario.target)
    start_world = grid.cell_to_world(scenario.start)
    _add_target(base, target_world)

    qpos = base.agent.robot.get_qpos().clone()
    rest = np.asarray(base.agent.keyframes["rest"].qpos, dtype=np.float32)
    qpos[0, : len(rest)] = qpos.new_tensor(rest)
    root_xy = np.asarray(base.agent.robot.pose.p[0, :2].cpu(), dtype=np.float64)
    camera = base._human_render_cameras["render_camera"].camera
    frames: list[np.ndarray] = []
    previous_yaw = 0.0
    for index, xy in enumerate(world_trajectory):
        direction = (
            world_trajectory[index + 1] - xy
            if index + 1 < len(world_trajectory)
            else xy - world_trajectory[index - 1]
        )
        if np.linalg.norm(direction) > 1e-8:
            desired_yaw = math.atan2(float(direction[1]), float(direction[0]))
            previous_yaw += math.atan2(
                math.sin(desired_yaw - previous_yaw),
                math.cos(desired_yaw - previous_yaw),
            )
        qpos[0, 0] = float(xy[0] - root_xy[0])
        qpos[0, 1] = float(xy[1] - root_xy[1])
        qpos[0, 2] = float(previous_yaw)
        base.agent.robot.set_qpos(qpos)

        heading = np.asarray([math.cos(previous_yaw), math.sin(previous_yaw)])
        side = np.asarray([-heading[1], heading[0]])
        # A steep, close oblique view keeps both the robot and furnished room
        # visible while looking over ReplicaCAD's full-height interior walls.
        eye_xy = xy - 2.2 * heading - 1.3 * side
        camera_pose = sapien_utils.look_at(
            [float(eye_xy[0]), float(eye_xy[1]), 4.6],
            [
                float(xy[0] + 0.45 * heading[0]),
                float(xy[1] + 0.45 * heading[1]),
                0.55,
            ],
        )
        camera.set_local_pose(camera_pose.sp)
        # Render only the moving external camera. SceneManipulation also
        # registers a torso-mounted diagnostic camera; env.render() would tile
        # both horizontally and make the response-letter panels unreadable.
        raw = base.render_rgb_array("render_camera")
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        rgb = np.asarray(raw)[0]
        if rgb.dtype != np.uint8:
            scale = 255.0 if float(rgb.max()) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
        frames.append(
            _annotate(
                rgb,
                scenario,
                trajectory,
                index,
                result.path_m,
                result.replans,
            )
        )
        if (index + 1) % 24 == 0:
            print(f"Rendered {index + 1}/{len(world_trajectory)} frames", flush=True)
    env.close()

    video_path = args.output_root / "dream_replicacad_object_search.mp4"
    with imageio.get_writer(
        video_path,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_log_level="warning",
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    montage_indices = np.linspace(0, len(frames) - 1, 4).astype(int)
    panels = []
    for panel_index, frame_index in enumerate(montage_indices):
        frame = frames[frame_index].copy()
        cv2.putText(
            frame,
            f"({chr(ord('a') + panel_index)}) {100 * frame_index / max(len(frames) - 1, 1):.0f}% route",
            (18, 111),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(frame)
    montage = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    montage_path = args.output_root / "dream_replicacad_montage.png"
    imageio.imwrite(montage_path, montage)
    if args.paper_montage is not None:
        args.paper_montage.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.paper_montage, montage)

    route_path = args.output_root / "trajectory.csv"
    _write_route_csv(route_path, raw_trajectory, grid)
    metadata_path = args.output_root / "metadata.json"
    navmesh_source = (
        Path(os.environ["MS_ASSET_DIR"])
        / "data/scene_datasets/replica_cad_dataset/configs/scenes"
        / "apt_0.scene_instance.fetch.navigable_positions.obj"
    )
    metadata = {
        "evidence_type": "qualitative simulator object-goal search visualization",
        "boundary": QUALITATIVE_BOUNDARY,
        "task_definition": "start at the Fetch spawn and find the red object in the southern destination area",
        "environment_id": ENVIRONMENT_ID,
        "scene": SCENE_NAME,
        "build_config_index": args.build_config_index,
        "simulator": "ManiSkill 3.0.1 / SAPIEN 3.0.3",
        "robot": "Fetch embodiment distributed with ManiSkill",
        "simulation_backend": "PhysX CPU",
        "render_backend": "CPU Vulkan (lavapipe)",
        "navigation_surface": "Fetch-specific ReplicaCAD navigable mesh",
        "replicacad_source": DATASET_URL,
        "context_proxy": (
            f"binary destination-area context for world y <= {args.destination_boundary_y:.2f} m; "
            "available to the selector only after local observation"
        ),
        "context_is_learned_perception": False,
        "map_initially_known": False,
        "local_sensor_range_m": args.sensor_range,
        "semantic_rate": args.semantic_rate,
        "travel_budget_m": args.budget_m,
        "success": bool(result.success),
        "shortest_path_m": result.shortest_path_m,
        "executed_planner_path_m": result.path_m,
        "spl": result.spl,
        "replans": result.replans,
        "raw_trajectory_points": len(raw_trajectory),
        "start_world_xy_m": start_world.tolist(),
        "target_world_xy_m": target_world.tolist(),
        "navmesh_resolution_m": args.resolution,
        "navmesh_free_area_m2": scenario.floor_area_m2,
        "navmesh_source_sha256": _sha256(navmesh_source),
        "frames": len(frames),
        "fps": args.fps,
        "frame_width_px": int(frames[0].shape[1]),
        "frame_height_px": int(frames[0].shape[0]),
        "camera": "moving external render_camera",
        "video_sha256": _sha256(video_path),
        "montage_sha256": _sha256(montage_path),
        "trajectory_sha256": _sha256(route_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/results/maniskill_replicacad"),
    )
    parser.add_argument(
        "--paper-montage",
        type=Path,
        default=None,
        help="optional second PNG destination for the response-letter figure",
    )
    parser.add_argument("--build-config-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=0.08)
    parser.add_argument("--sensor-range", type=float, default=1.2)
    parser.add_argument("--semantic-rate", type=float, default=0.1)
    parser.add_argument("--budget-m", type=float, default=80.0)
    parser.add_argument("--target-x", type=float, default=0.79)
    parser.add_argument("--target-y", type=float, default=-6.30)
    parser.add_argument("--destination-boundary-y", type=float, default=-2.50)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
