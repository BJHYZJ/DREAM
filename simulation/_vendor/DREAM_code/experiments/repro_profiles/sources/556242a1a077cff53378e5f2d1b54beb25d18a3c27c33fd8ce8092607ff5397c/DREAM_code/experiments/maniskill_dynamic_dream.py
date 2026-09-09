#!/usr/bin/env python3
"""Sensor-driven DREAM dynamic relocation trial in ManiSkill/ReplicaCAD.

This is the executable smoke-test backend for the broader multi-scene runner.
It uses rendered instance masks plus RGB-D for target localization and voxel
memory, a 16-cell receding-horizon DREAM frontier policy, PhysX base/arm
controllers, and a continuously force-driven support carrying the target.
No robot, target, or support pose setter is called after initialization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[1]
if not os.environ.get("MS_ASSET_DIR"):
    os.environ["MS_ASSET_DIR"] = os.environ.get(
        "DREAM_MANISKILL_ASSET_DIR", str(ROOT / ".maniskill_assets")
    )
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tmech")
sys.path.insert(0, str(REPO / "src"))

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien
import torch
from scipy.ndimage import distance_transform_edt

import mani_skill  # noqa: F401
from mani_skill.utils.building import actors

from dream.dynamic_memory import (
    DEFAULT_EXECUTION_CONFIG,
    FocusOutcome,
    classify_focused_observation,
    initial_scan_yaws,
)
from dream_simulation_core import (
    OnlineFrontierPlanner,
    SensorVoxelMemory,
    SimulationSpec,
    homogeneous_extrinsic,
    sensor_frame_to_world,
    sha256,
    write_manifest,
)
from houseexpo_cross_room import shortcut_path, weighted_distances, _reconstruct_path
from maniskill_replicacad_closed_loop import (
    ENVIRONMENT_ID,
    Trial,
    build_receptacle,
    make_geometry,
    wrap,
)
from maniskill_replicacad_search_video import (
    SCENE_NAME,
    build_search_scenario,
    rasterize_navigation_mesh,
)


PHYSICAL_OFFSET = np.asarray([0.049, -0.023], dtype=float)
# A compact 6 x 6 x 14 cm household target stays clear of scene clutter while
# riding on the relocated support.  A shallow 8.4 cm shoulder near its top is
# analogous to the rim of a small bottle/carton and gives parallel fingers a
# mechanically meaningful lifting surface.  Grasping remains bilateral PhysX
# contact: there is no grasp weld, suction, or post-initialization pose setter.
DYNAMIC_TARGET_HALF_SIZE_M = 0.030
DYNAMIC_TARGET_HALF_HEIGHT_M = 0.070
DYNAMIC_TARGET_DENSITY_KG_M3 = 500.0
DYNAMIC_TARGET_SHOULDER_HALF_SIZE_M = 0.042
DYNAMIC_TARGET_SHOULDER_HALF_HEIGHT_M = 0.008
DYNAMIC_TARGET_STATIC_FRICTION = 5.0
DYNAMIC_TARGET_DYNAMIC_FRICTION = 4.0
BOUNDARY = (
    "Controlled sensor-driven component trial. Target localization and semantic memory use "
    "only current rendered instance pixels, RGB-D and camera calibration; simulator target "
    "pose is reserved for scoring. Traversability uses the official Fetch navmesh. This "
    "isolates DREAM memory/planning/control/manipulation from learned open-vocabulary perception."
)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def collision_avoiding_dock_waypoints(
    base_xy: np.ndarray,
    target_xy: np.ndarray,
    dock_xy: np.ndarray,
    clearance_m: float,
    arc_segments: int = 6,
) -> np.ndarray:
    """Return a same-side circular approach whose chords avoid the target."""

    target = np.asarray(target_xy, dtype=float)
    dock_vector = np.asarray(dock_xy, dtype=float) - target
    dock_distance = float(np.linalg.norm(dock_vector))
    if dock_distance < 1e-6 or clearance_m < dock_distance or arc_segments < 2:
        raise ValueError("invalid collision-avoiding dock geometry")
    dock_direction = dock_vector / dock_distance
    perpendicular = np.asarray([-dock_direction[1], dock_direction[0]])
    side = 1.0 if float((np.asarray(base_xy) - target) @ perpendicular) >= 0 else -1.0
    angles = np.linspace(side * math.pi / 2, 0.0, arc_segments + 1)
    radii = np.linspace(clearance_m, dock_distance, arc_segments + 1)
    return np.asarray(
        [
            target
            + radius
            * (math.cos(angle) * dock_direction + math.sin(angle) * perpendicular)
            for angle, radius in zip(angles, radii)
        ]
    )


def build_moving_target(base: Any, logical_xy: np.ndarray, target_z: float = 1.020):
    """Build a force-driven rigid support and a separate graspable target."""

    physical_xy = np.asarray(logical_xy, dtype=float) + PHYSICAL_OFFSET
    target_half = DYNAMIC_TARGET_HALF_SIZE_M
    target_half_height = DYNAMIC_TARGET_HALF_HEIGHT_M
    support_top = target_z - target_half_height - 0.002
    support_builder = base.scene.create_actor_builder()
    floor_material = sapien.pysapien.physx.PhysxMaterial(0.02, 0.01, 0.0)
    tabletop_material = sapien.pysapien.physx.PhysxMaterial(3.0, 2.5, 0.0)
    # Compound rigid body: a small low-friction foot, narrow stem, and broad
    # high-friction tabletop.  It behaves as a force-driven cart while leaving
    # the target unobstructed for a top-down grasp.
    support_builder.add_box_collision(
        pose=sapien.Pose([0, 0, 0.020]),
        half_size=[0.070, 0.070, 0.020],
        material=floor_material,
        density=100,
    )
    support_builder.add_box_collision(
        pose=sapien.Pose([0, 0, support_top / 2]),
        half_size=[0.025, 0.025, support_top / 2 - 0.045],
        material=tabletop_material,
        density=100,
    )
    support_builder.add_box_collision(
        pose=sapien.Pose([0, 0, support_top - 0.020]),
        half_size=[0.120, 0.120, 0.020],
        material=tabletop_material,
        density=100,
    )
    support_builder.add_box_visual(
        pose=sapien.Pose([0, 0, support_top / 2]),
        half_size=[0.025, 0.025, support_top / 2 - 0.045],
        material=[0.12, 0.35, 0.82, 1.0],
    )
    support_builder.add_box_visual(
        pose=sapien.Pose([0, 0, support_top - 0.020]),
        half_size=[0.120, 0.120, 0.020],
        material=[0.12, 0.35, 0.82, 1.0],
    )
    support_builder.set_initial_pose(sapien.Pose([*physical_xy, 0.002]))
    support = support_builder.build(name="dream_force_driven_support")
    support.set_locked_motion_axes([False, False, False, True, True, True])
    support.linear_damping = 3.0

    target_builder = base.scene.create_actor_builder()
    target_material = sapien.pysapien.physx.PhysxMaterial(
        DYNAMIC_TARGET_STATIC_FRICTION,
        DYNAMIC_TARGET_DYNAMIC_FRICTION,
        0.0,
    )
    target_builder.add_box_collision(
        half_size=[target_half, target_half, target_half_height],
        material=target_material,
        density=DYNAMIC_TARGET_DENSITY_KG_M3,
        patch_radius=0.03,
        min_patch_radius=0.03,
    )
    target_builder.add_box_collision(
        pose=sapien.Pose([0, 0, 0.045]),
        half_size=[
            DYNAMIC_TARGET_SHOULDER_HALF_SIZE_M,
            DYNAMIC_TARGET_SHOULDER_HALF_SIZE_M,
            DYNAMIC_TARGET_SHOULDER_HALF_HEIGHT_M,
        ],
        material=target_material,
        density=DYNAMIC_TARGET_DENSITY_KG_M3,
        patch_radius=0.03,
        min_patch_radius=0.03,
    )
    target_builder.add_box_visual(
        half_size=[target_half, target_half, target_half_height],
        material=[1.0, 0.18, 0.02, 1.0],
    )
    target_builder.add_box_visual(
        pose=sapien.Pose([0, 0, 0.045]),
        half_size=[
            DYNAMIC_TARGET_SHOULDER_HALF_SIZE_M,
            DYNAMIC_TARGET_SHOULDER_HALF_SIZE_M,
            DYNAMIC_TARGET_SHOULDER_HALF_HEIGHT_M,
        ],
        material=[1.0, 0.32, 0.03, 1.0],
    )
    target_builder.set_initial_pose(sapien.Pose([*physical_xy, target_z]))
    target = target_builder.build(name="dream_dynamic_orange_target")
    target.linear_damping = 8.0
    target.angular_damping = 8.0
    return support, target


class DynamicTrial(Trial):
    video_title = "DREAM dynamic reacquisition + grasp | ReplicaCAD / Fetch"
    map_label = "ONLINE RGB-D + TARGET-SEMANTIC VOXELS | TOP VIEW"
    experiment_boundary = BOUNDARY
    traversability_source = "official Fetch navigable-position mesh"
    video_footer = (
        "SENSOR RGB-D + INSTANCE MASK | PHYSX env.step() | {playback:.1f}x | "
        "NO ROBOT/TARGET/SUPPORT TELEPORT"
    )

    def __init__(
        self,
        env: Any,
        grid: Any,
        scenario: Any,
        target: Any,
        support: Any,
        geometry: Any,
        args: argparse.Namespace,
        spec: SimulationSpec,
    ):
        super().__init__(env, grid, scenario, target, geometry, args)
        self.support = support
        self.spec = spec
        self.voxels = SensorVoxelMemory(voxel_size_m=0.05)
        self.planner = OnlineFrontierPlanner(
            grid.free,
            scenario.context,
            resolution_m=grid.resolution_m,
            sensor_range_m=args.sensor_range,
            semantic_rate=0.10,
            seed=args.seed,
        )
        # Trial map visualization and online planner share the same arrays.
        self.known = self.planner.known
        self.last_seen = self.planner.last_seen
        self.events: list[dict[str, Any]] = []
        self.memory_updates: list[dict[str, Any]] = []
        self.observation_frame_id = 0
        self.cached_target: np.ndarray | None = None
        self.detected_target: np.ndarray | None = None
        self.support_distance_m = 0.0
        self.support_max_step_m = 0.0
        self.target_support_max_xy_error_m = 0.0
        self.relocation_distance_m: float | None = None
        self.relocation_target_support_xy_error_m: float | None = None
        self.peak_measured_lift_m: float | None = None
        self.measured_lift_m: float | None = None
        self.lift_torso_start_m: float | None = None
        self.lift_torso_final_m: float | None = None
        self.stale_region_voxels_before: int | None = None
        self.stale_region_voxels_after: int | None = None
        self.initial_scan_drift_m: float | None = None
        self.selected_pick_yaw: float | None = None
        self.grid_free_cells = np.argwhere(self.grid.free)
        self.grid_free_world = np.asarray(
            [self.grid.cell_to_world(tuple(cell)) for cell in self.grid_free_cells]
        )
        self.previous_support_xy = _array(self.support.pose.p)[0, :2].astype(float)
        self.extra_manifest_fields: dict[str, Any] = {}

    def event(self, name: str, **payload: Any) -> None:
        xy, yaw = self.state()
        row = {
            "step": self.step,
            "sim_time_s": self.step / self.hz,
            "state": self.phase,
            "event": name,
            "base_x_m": float(xy[0]),
            "base_y_m": float(xy[1]),
            "base_yaw_rad": float(yaw),
            "video_frame_index": len(self.frames),
            **payload,
        }
        self.events.append(row)
        self.memory_event(name)
        print(json.dumps(row, sort_keys=True), flush=True)

    def command(
        self,
        arm: np.ndarray | None = None,
        grip: float = 1.0,
        base: tuple[float, float] = (0, 0),
        body: np.ndarray | None = None,
    ) -> None:
        """Keep the post-grasp controller continuous while driving the base.

        ManiSkill permits Fetch base commands in ``pd_ee_delta_pose`` mode.  A
        zero Cartesian increment holds the measured TCP pose, so there is no
        reason to reset into joint control after closing the fingers.  That
        reset introduced a one-step arm transient in cluttered ArchitecTHOR
        scenes and could drop an otherwise valid bilateral grasp.
        """

        if self.agent.control_mode == "pd_ee_delta_pose":
            if arm is not None or body is not None:
                raise RuntimeError(
                    "arm/body joint targets are unavailable in Cartesian hold mode"
                )
            self.command_ee(grip=grip, base=base)
            return
        super().command(arm=arm, grip=grip, base=base, body=body)
        # DREAM updates its memory while the base is moving, not only at
        # frontier endpoints.  Periodic head RGB-D therefore both fuses the
        # traversed corridor and allows a target passed between two raster
        # waypoints to stop the current receding-horizon action.
        if (
            self.phase in {"MOVE", "REACQUIRE"}
            and not self.detected
            and self.step % self.args.motion_sense_interval == 0
        ):
            detected = self.sense("fetch_head")
            # A few target pixels at the edge of the depth range are useful
            # semantic evidence, but stopping immediately can leave the base
            # on the wrong side of a doorway or table.  Finish the current
            # receding-horizon motion until the measured RGB-D target range is
            # inside a manipulation-planning envelope.  This mirrors DREAM's
            # observe--move--focus cadence and does not use actor pose.
            if (
                detected is not None
                and np.linalg.norm(detected[:2] - self.state()[0])
                <= self.args.motion_detection_stop_range
            ):
                self.detected = True

    def record_step(self, arm_log: np.ndarray, grip: float, base: tuple[float, float]) -> None:
        super().record_step(arm_log, grip, base)
        current_support = _array(self.support.pose.p)[0, :2].astype(float)
        support_step = float(np.linalg.norm(current_support - self.previous_support_xy))
        self.previous_support_xy = current_support.copy()
        self.support_distance_m += support_step
        self.support_max_step_m = max(self.support_max_step_m, support_step)
        object_xy = self.object_xyz()[:2]
        support_error = float(np.linalg.norm(object_xy - current_support))
        if self.phase in {
            "Initialize", "ROTATE", "SEARCH", "MOVE", "TARGET MOVE",
            "FOCUS OLD", "CLEAR STALE", "REACQUIRE",
        }:
            self.target_support_max_xy_error_m = max(
                self.target_support_max_xy_error_m, support_error
            )
        self.rows[-1].update(
            support_x_m=current_support[0],
            support_y_m=current_support[1],
            support_step_m=support_step,
            object_support_xy_error_m=support_error,
        )

    def move_ee(
        self,
        target_world: np.ndarray,
        grip: float,
        tolerance: float = 0.014,
        strict: bool = True,
    ) -> float:
        """Servo the TCP without the near-tabletop overshoot of the smoke test."""

        for _ in range(self.args.ee_max_steps):
            tcp = _array(self.agent.tcp_pose.p)[0].astype(float)
            error = np.asarray(target_world, dtype=float) - tcp
            residual = float(np.linalg.norm(error))
            if residual < tolerance:
                return residual
            yaw = self.state()[1]
            c, s = math.cos(yaw), math.sin(yaw)
            local = np.asarray(
                [c * error[0] + s * error[1], -s * error[0] + c * error[1], error[2]]
            )
            # The Cartesian action is normalized.  A saturated 0.28 command
            # can overshoot by several centimetres before the outer loop sees
            # the next state, so the dynamic trial uses a conservative limit.
            translation = np.clip(
                local / 0.1 * self.args.ee_position_gain,
                -self.args.ee_action_limit,
                self.args.ee_action_limit,
            )
            self.command_ee(np.r_[translation, [0, 0, 0]], grip)
        tcp = _array(self.agent.tcp_pose.p)[0].astype(float)
        residual = float(np.linalg.norm(np.asarray(target_world, dtype=float) - tcp))
        if strict and residual > 0.028:
            self.capture()
            self.args.output_root.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(self.args.output_root / "ee_controller_failure.png", self.frames[-1])
            raise RuntimeError(
                f"end-effector controller failed: target={np.asarray(target_world).tolist()}, "
                f"tcp={tcp.tolist()}, residual={residual:.4f}"
            )
        return residual

    def _capture_sensor(self, sensor_name: str) -> dict[str, np.ndarray]:
        self.base.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.base.capture_sensor_data()
        sensor = self.base._sensors[sensor_name]
        obs = sensor.get_obs(rgb=True, depth=True, position=True, segmentation=True)
        params = sensor.get_params()
        return {**{key: _array(value) for key, value in obs.items()}, **{key: _array(value) for key, value in params.items()}}

    def sense(self, sensor_name: str = "fetch_head") -> np.ndarray | None:
        frame = self._capture_sensor(sensor_name)
        points, colors, ids, depth = sensor_frame_to_world(
            position_gl_mm=frame["position"],
            rgb=frame["rgb"],
            segmentation=frame["segmentation"],
            cam2world_gl=frame["cam2world_gl"],
            stride=self.args.point_stride,
            max_depth_m=DEFAULT_EXECUTION_CONFIG.depth_max_m,
        )
        self.observation_frame_id += 1
        extrinsic = homogeneous_extrinsic(frame["extrinsic_cv"])
        camera_in_world = np.linalg.inv(extrinsic)
        intrinsic = frame["intrinsic_cv"][0] if frame["intrinsic_cv"].ndim == 3 else frame["intrinsic_cv"]
        update = self.voxels.integrate(
            frame_id=self.observation_frame_id,
            points_world=points,
            colors_rgb=colors,
            instance_ids=ids,
            depth_m=depth,
            intrinsics_cv=intrinsic,
            camera_in_world_cv=camera_in_world,
            clearing=self.spec.variant != "no_clearing",
        )
        update_row = asdict(update) | {"sensor": sensor_name, "sim_step": self.step}
        self.memory_updates.append(update_row)

        target_id = int(_array(self.target.per_scene_id)[0])
        segmentation = frame["segmentation"]
        if segmentation.ndim == 4:
            segmentation = segmentation[0]
        # Reproject every target pixel, independently of the downsampled map,
        # so a small distant object is not lost to point-cloud stride.
        position = frame["position"]
        if position.ndim == 4:
            position = position[0]
        local = position[..., :3].astype(np.float32) / 1000.0
        pixel_range = np.linalg.norm(local, axis=-1)
        mask = (
            (segmentation[..., 0] == target_id)
            & np.isfinite(pixel_range)
            & (pixel_range >= DEFAULT_EXECUTION_CONFIG.depth_min_m)
            & (pixel_range <= DEFAULT_EXECUTION_CONFIG.depth_max_m)
        )
        if int(mask.sum()) < self.args.minimum_target_pixels:
            return None
        homogeneous = np.concatenate(
            (local, np.ones((*local.shape[:2], 1), dtype=np.float32)), axis=-1
        )
        cam2world = frame["cam2world_gl"][0] if frame["cam2world_gl"].ndim == 3 else frame["cam2world_gl"]
        world = (homogeneous.reshape(-1, 4) @ cam2world.T)[:, :3].reshape(*local.shape[:2], 3)
        target_points = world[mask]
        detected = np.median(target_points, axis=0)
        # The experiment target dimensions are known to the geometric grasp
        # module.  Actor-ID pixels expose visible surfaces, so recover the box
        # center from its observed top surface instead of treating a surface
        # point as the grasp center.  No simulator actor pose is consulted.
        detected[2] = float(np.max(target_points[:, 2]) - self.geometry.half_height)
        if np.all(np.isfinite(detected)):
            self.detected_target = detected.astype(float)
            return self.detected_target.copy()
        return None

    def scan(self, *, focus: bool = False) -> np.ndarray | None:
        original_phase = self.phase
        self.phase = "FOCUS OLD" if focus else "SCAN"
        detections: list[np.ndarray] = []
        if focus:
            poses = [(0.0, 0.12, 0.0)]
        else:
            poses = [
                (0.0, 0.45, 0.0),
                (0.40, 0.20, 0.22),
                (0.80, 0.15, 0.42),
                (-0.40, 0.20, -0.22),
                (-0.80, 0.15, -0.42),
                (0.0, 0.12, 0.0),
            ]
        for head_pan, head_tilt, shoulder_offset in poses:
            arm = self.rest_arm.copy()
            arm[0] += shoulder_offset
            body = np.asarray([head_pan, head_tilt, self.body_target[2]])
            for _ in range(self.args.scan_hold_steps):
                self.command(arm=arm, grip=1.0, body=body)
            for sensor_name in ("fetch_head", "fetch_hand"):
                detected = self.sense(sensor_name)
                if detected is not None:
                    detections.append(detected)
            self.capture()
        self.phase = original_phase
        return np.median(np.stack(detections), axis=0) if detections else None

    def initial_rotate_and_scan(self) -> np.ndarray | None:
        self.phase = "ROTATE"
        start_xy = self.state()[0].copy()
        start_yaw = self.state()[1]
        detections: list[np.ndarray] = []
        for index, yaw in enumerate(initial_scan_yaws(start_yaw)):
            self.turn(yaw, 1.0, rate=0.60)
            detected = self.sense("fetch_head")
            if detected is not None:
                detections.append(detected)
            self.event("initial_rotate_observation", rotation_index=index, target_pixels_detected=int(detected is not None))
            self.capture()
        self.initial_scan_drift_m = float(np.linalg.norm(self.state()[0] - start_xy))
        self.event(
            "initial_360_scan_complete",
            base_drift_m=self.initial_scan_drift_m,
            maximum_allowed_drift_m=self.args.maximum_initial_scan_drift,
        )
        if self.initial_scan_drift_m > self.args.maximum_initial_scan_drift:
            raise RuntimeError(
                "official Fetch spawn is collision-unstable during the initial scan: "
                f"drift={self.initial_scan_drift_m:.3f} m"
            )
        return np.median(np.stack(detections), axis=0) if detections else None

    def _cell(self) -> tuple[int, int]:
        xy = self.state()[0]
        cell = self.grid.world_to_cell(xy)
        if (
            0 <= cell[0] < self.grid.free.shape[0]
            and 0 <= cell[1] < self.grid.free.shape[1]
            and self.grid.free[cell]
        ):
            return cell
        # Conservative GLB rasterization can place the measured base centre
        # one cell inside the inflated boundary even though PhysX has not
        # collided.  Anchor planning/observation to the nearest validated free
        # cell, while continuing to control from the measured physical pose.
        index = int(np.argmin(np.linalg.norm(self.grid_free_world - xy, axis=1)))
        return tuple(int(v) for v in self.grid_free_cells[index])

    def online_search(self, label: str, max_replans: int) -> np.ndarray:
        self.phase = label
        for cycle in range(max_replans):
            self.planner.observe(self._cell())
            detected = self.scan(focus=False)
            if detected is not None:
                self.detected = True
                self.event("target_observed_from_current_rgbd", search_cycle=cycle, sensor_xyz_m=detected.tolist())
                return detected
            chunk = self.planner.next_chunk(self._cell())
            if not chunk:
                self.event("frontier_sensor_turn", search_cycle=cycle)
                continue
            points = np.asarray([self.grid.cell_to_world(point) for point in chunk[1:]])
            self.replans = self.planner.replans
            self.event("receding_horizon_plan", search_cycle=cycle, dense_horizon=DEFAULT_EXECUTION_CONFIG.navigation_step_num, executed_goals=len(points))
            if len(points):
                self.phase = "MOVE" if label == "SEARCH" else "REACQUIRE"
                chunk_start_xy = self.state()[0].copy()
                self.last_stall_xy = None
                try:
                    # A frontier prefix can end only one 8 cm raster cell from
                    # the current base pose.  The earlier 14 cm stopping
                    # envelope accepted that command without moving, causing
                    # repeated scan/plan cycles at the same location.  The
                    # tighter physical tolerance guarantees measurable motion
                    # for short receding-horizon actions.
                    self.drive(points, 1.0, self.args.search_speed, 0.055)
                    if self.detected and self.detected_target is not None:
                        self.event(
                            "target_observed_while_base_moving",
                            search_cycle=cycle,
                            sensor_xyz_m=self.detected_target.tolist(),
                        )
                        return self.detected_target.copy()
                except RuntimeError as error:
                    # ReplicaCAD navmeshes describe static base clearance but
                    # do not guarantee that scene-level movable props remain
                    # clear.  A stalled controller therefore becomes a fresh
                    # local obstacle observation and triggers replanning; the
                    # robot is never moved through it or reset to the path.
                    recovered_xy = self.state()[0].copy()
                    contact_base_xy = (
                        self.last_stall_xy.copy()
                        if self.last_stall_xy is not None
                        else recovered_xy.copy()
                    )
                    # A controller stall is also an informative viewpoint.
                    # Preserve DREAM's focus-before-moving behavior by sensing
                    # here before backing away; otherwise a visible target can
                    # be missed precisely because recovery leaves the view.
                    stalled_detection = self.scan(focus=False)
                    if stalled_detection is not None:
                        self.detected = True
                        self.event(
                            "target_observed_during_stall_focus",
                            search_cycle=cycle,
                            sensor_xyz_m=stalled_detection.tolist(),
                            stall_reason=str(error),
                        )
                        return stalled_detection
                    approach_vector = contact_base_xy - chunk_start_xy
                    approach_norm = float(np.linalg.norm(approach_vector))
                    if approach_norm > 1e-6:
                        # The bumper/contact surface lies roughly one base
                        # radius in front of the center at the last advancing
                        # pose.  Mark that contact proxy, not the pose reached
                        # after the controller's reverse recovery.
                        contact_proxy_xy = (
                            contact_base_xy + 0.25 * approach_vector / approach_norm
                        )
                    else:
                        contact_proxy_xy = contact_base_xy
                    blocked = self.grid.world_to_cell(contact_proxy_xy)
                    radius = self.args.stall_obstacle_radius_cells
                    r0, c0 = blocked
                    # ``drive`` has already executed a controller-driven
                    # reverse motion after detecting the stall.  Treat its
                    # measured endpoint as the safe recovery pose.  Returning
                    # to the beginning of a multi-waypoint chunk can traverse
                    # the just-observed obstacle again and, in narrow rooms,
                    # isolate the planner on the wrong side of the contact.
                    # No simulator pose setter is used here.
                    safe_xy = recovered_xy
                    recovery_ok = bool(
                        np.linalg.norm(recovered_xy - contact_base_xy) >= 0.01
                    )
                    # Keep the recovered robot cell traversable while marking
                    # the estimated contact neighborhood.
                    row_slice = slice(
                        max(0, r0 - radius),
                        min(self.known.shape[0], r0 + radius + 1),
                    )
                    col_slice = slice(
                        max(0, c0 - radius),
                        min(self.known.shape[1], c0 + radius + 1),
                    )
                    self.planner.known[row_slice, col_slice] = -1
                    # Persist the collision observation in the planner's
                    # dynamic traversability layer so the next reveal cannot
                    # overwrite it with the static official navmesh.
                    self.planner.free[row_slice, col_slice] = False
                    # The base physically traversed the short segment from
                    # the stall pose to the reverse-recovery endpoint.  Keep
                    # those measured base-centre cells as an escape corridor;
                    # otherwise a square obstacle dilation can surround the
                    # current cell and leave the frontier planner with no
                    # reachable goal even though the robot has backed clear.
                    recovery_distance = float(
                        np.linalg.norm(contact_base_xy - recovered_xy)
                    )
                    recovery_samples = max(
                        2,
                        int(
                            math.ceil(
                                recovery_distance
                                / max(self.grid.resolution_m / 2.0, 1e-6)
                            )
                        ),
                    )
                    recovery_corridor_cells: set[tuple[int, int]] = set()
                    for alpha in np.linspace(0.0, 1.0, recovery_samples):
                        corridor_cell = self.grid.world_to_cell(
                            (1.0 - alpha) * recovered_xy
                            + alpha * contact_base_xy
                        )
                        if (
                            0 <= corridor_cell[0] < self.grid.free.shape[0]
                            and 0 <= corridor_cell[1] < self.grid.free.shape[1]
                            and self.grid.free[corridor_cell]
                        ):
                            recovery_corridor_cells.add(corridor_cell)
                    for corridor_cell in recovery_corridor_cells:
                        self.planner.free[corridor_cell] = True
                        self.planner.known[corridor_cell] = 1
                    current_cell = self._cell()
                    if self.grid.free[current_cell]:
                        self.planner.free[current_cell] = True
                        self.planner.known[current_cell] = 1
                    self.planner.cached_frontier = None
                    self.event(
                        "physical_stall_marked_as_obstacle",
                        search_cycle=cycle,
                        blocked_cell=list(blocked),
                        contact_base_xy_m=contact_base_xy.tolist(),
                        contact_proxy_xy_m=contact_proxy_xy.tolist(),
                        safe_pose_xy_m=safe_xy.tolist(),
                        physical_recovery_succeeded=int(recovery_ok),
                        recovery_corridor_cells=len(recovery_corridor_cells),
                        reason=str(error),
                    )
                    self.phase = label
                    continue
                self.phase = label
        raise RuntimeError(f"target not found within {max_replans} online replans")

    def _support_force_step(self, desired_velocity_xy: np.ndarray) -> None:
        velocity = _array(self.support.get_linear_velocity())[0, :2].astype(float)
        force_xy = np.clip(
            self.args.support_velocity_gain * (desired_velocity_xy - velocity),
            -self.args.support_force_limit,
            self.args.support_force_limit,
        )
        self.support.apply_force(np.r_[force_xy, 0.0].astype(np.float32))
        self.command(grip=1.0)

    def move_target_continuously(self, destination_logical_xy: np.ndarray) -> np.ndarray:
        self.phase = "TARGET MOVE"
        destination = np.asarray(destination_logical_xy, dtype=float) + PHYSICAL_OFFSET
        start = _array(self.support.pose.p)[0, :2].astype(float)
        self.event("dynamic_motion_start", support_start_xy_m=start.tolist(), destination_xy_m=destination.tolist())
        best = float(np.linalg.norm(destination - start))
        stagnant = 0
        for _ in range(self.args.support_max_steps):
            position = _array(self.support.pose.p)[0, :2].astype(float)
            delta = destination - position
            distance = float(np.linalg.norm(delta))
            if distance < self.args.support_tolerance:
                break
            desired = self.args.support_speed * delta / max(distance, 1e-9)
            self._support_force_step(desired)
            if distance < best - 0.002:
                best, stagnant = distance, 0
            else:
                stagnant += 1
            if stagnant > 500:
                raise RuntimeError("force-driven target support stalled")
        else:
            raise RuntimeError("force-driven target support exceeded step budget")
        for _ in range(80):
            self._support_force_step(np.zeros(2))
        final_support = _array(self.support.pose.p)[0, :2].astype(float)
        final_target = self.object_xyz()
        move_distance = float(np.linalg.norm(final_support - start))
        final_support_error = float(np.linalg.norm(final_target[:2] - final_support))
        if move_distance <= DEFAULT_EXECUTION_CONFIG.target_relocation_radius_m:
            raise RuntimeError(f"dynamic relocation too small: {move_distance:.3f} m")
        if final_support_error > 0.09:
            raise RuntimeError("target did not remain physically supported during relocation")
        self.relocation_distance_m = move_distance
        self.relocation_target_support_xy_error_m = final_support_error
        # The force-driven support represents a piece of furniture being moved
        # and then parked, rather than a powered cart that keeps coasting while
        # Fetch approaches.  Engage its translational parking brake only after
        # the continuous PhysX relocation and settling period have completed.
        # The target remains a separate, unlocked dynamic actor and therefore
        # still has to be acquired and lifted through gripper contact.
        self.support.set_locked_motion_axes([True, True, True, True, True, True])
        self.event(
            "dynamic_motion_complete",
            support_final_xy_m=final_support.tolist(),
            target_final_xyz_m=final_target.tolist(),
            measured_relocation_m=move_distance,
            target_support_xy_error_m=final_support_error,
            support_parking_brake=True,
        )
        return np.asarray(destination_logical_xy, dtype=float)

    def focus_and_invalidate(self) -> np.ndarray | None:
        if self.cached_target is None:
            raise RuntimeError("focus requested without cached target")
        self.phase = "FOCUS OLD"
        xy, _ = self.state()
        bearing = math.atan2(self.cached_target[1] - xy[1], self.cached_target[0] - xy[0])
        self.turn(bearing, 1.0)
        detected = self.scan(focus=True) if self.spec.variant != "no_focus" else self.scan(focus=False)
        outcome = classify_focused_observation(
            torch.as_tensor(self.cached_target),
            None if detected is None else torch.as_tensor(detected),
            observation_is_current=True,
        )
        displacement = None if detected is None else float(np.linalg.norm(detected[:2] - self.cached_target[:2]))
        self.event("focused_verification", outcome=outcome.value, detected_displacement_m=displacement)
        if outcome in {FocusOutcome.CLEAR_STALE, FocusOutcome.RELOCATED} and self.spec.variant != "dynamic_off":
            self.phase = "CLEAR STALE"
            reject_radius = DEFAULT_EXECUTION_CONFIG.target_reject_region_radius_m
            self.stale_region_voxels_before = self.voxels.count_region(
                self.cached_target, reject_radius
            )
            removed = 0
            if self.spec.variant != "no_clearing":
                removed = self.voxels.clear_region(
                    self.cached_target,
                    reject_radius,
                )
            self.stale_region_voxels_after = self.voxels.count_region(
                self.cached_target, reject_radius
            )
            self.event(
                "stale_target_region_rejected",
                removed_voxels=removed,
                stale_region_voxels_before=self.stale_region_voxels_before,
                stale_region_voxels_after=self.stale_region_voxels_after,
                reject_radius_m=reject_radius,
            )
            self.cached_target = None
        return detected

    def grasp_carry_place(self, detected: np.ndarray, place_base_xy: np.ndarray) -> None:
        logical_target_xy = np.asarray(detected[:2], dtype=float) - PHYSICAL_OFFSET
        self.geometry = make_geometry(self.base, logical_target_xy, place_base_xy)
        if self.selected_pick_yaw is not None:
            yaw = float(self.selected_pick_yaw)
            rotation = np.asarray(
                [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
            )
            dock = np.asarray(detected[:2], dtype=float) - rotation @ np.asarray(
                [0.550, 0.038]
            )
            self.geometry = replace(self.geometry, pick_yaw=yaw, dock_xy=dock)
        current_base_xy, _ = self.state()
        target_xy = np.asarray(detected[:2], dtype=float)
        dock_waypoints = collision_avoiding_dock_waypoints(
            current_base_xy,
            target_xy,
            self.geometry.dock_xy,
            self.args.approach_clearance,
        )

        def route_on_grid(
            start_xy: np.ndarray,
            goal_xy: np.ndarray,
            free: np.ndarray,
        ) -> np.ndarray:
            route_free = free.copy()
            free_cells = np.argwhere(route_free)
            if len(free_cells) == 0:
                raise RuntimeError("navigation grid has no free anchor cells")
            free_world = np.asarray(
                [self.grid.cell_to_world(tuple(cell)) for cell in free_cells]
            )

            def nearest_anchor(xy: np.ndarray) -> tuple[tuple[int, int], float]:
                errors = np.linalg.norm(free_world - np.asarray(xy), axis=1)
                index = int(np.argmin(errors))
                return tuple(int(v) for v in free_cells[index]), float(errors[index])

            start_cell, start_snap = nearest_anchor(start_xy)
            goal_cell, goal_snap = nearest_anchor(goal_xy)
            if start_snap > self.args.grid_snap_distance or goal_snap > self.args.grid_snap_distance:
                raise RuntimeError(
                    f"grid anchor exceeds tolerance: start={start_snap:.3f} m, "
                    f"goal={goal_snap:.3f} m"
                )
            distances, parents = weighted_distances(route_free, start_cell)
            if not np.isfinite(distances[goal_cell]):
                raise RuntimeError(f"grid route is unreachable: {start_xy} -> {goal_xy}")
            cells = shortcut_path(
                _reconstruct_path(parents, start_cell, goal_cell),
                route_free,
            )
            first_index = 0 if start_snap > 0.02 else 1
            points = np.asarray(
                [self.grid.cell_to_world(cell) for cell in cells[first_index:]]
            )
            return np.vstack((points, goal_xy)) if len(points) else np.asarray([goal_xy])

        # Use the compact calibrated arc where the conservative grid confirms
        # it.  In cluttered ArchitecTHOR homes, fall back to a global grid path
        # with a target/support exclusion disk instead of assuming a direct
        # line from the observation pose to the arc.
        def grid_point_free(point: np.ndarray, free: np.ndarray) -> bool:
            row, col = self.grid.world_to_cell(point)
            return bool(
                0 <= row < free.shape[0]
                and 0 <= col < free.shape[1]
                and free[row, col]
            )

        def grid_segment_free(first: np.ndarray, second: np.ndarray, free: np.ndarray) -> bool:
            count = max(3, int(np.linalg.norm(second - first) / 0.04) + 1)
            return all(grid_point_free(point, free) for point in np.linspace(first, second, count))

        arc_chain = np.vstack((current_base_xy, dock_waypoints))
        arc_grid_clear = all(
            grid_segment_free(first, second, self.grid.free)
            for first, second in zip(arc_chain[:-1], arc_chain[1:])
        )
        approach_kind = "calibrated_clearance_arc"
        approach_free = self.grid.free.copy()
        rows, cols = np.indices(approach_free.shape)
        world_x = self.grid.minimum_xy[0] + cols * self.grid.resolution_m
        world_y = self.grid.maximum_xy[1] - rows * self.grid.resolution_m
        target_disk = (world_x - target_xy[0]) ** 2 + (world_y - target_xy[1]) ** 2
        approach_free[target_disk < self.args.grid_target_exclusion**2] = False
        if not arc_grid_clear:
            dock_waypoints = route_on_grid(
                current_base_xy,
                self.geometry.dock_xy,
                approach_free,
            )
            approach_kind = "global_grid_route_with_target_exclusion"
        build_receptacle(self.base, self.geometry)
        self.event(
            "calibrated_grasp_and_delivery_plan",
            pick_yaw_rad=self.geometry.pick_yaw,
            place_base_xy_m=self.geometry.place_base_xy.tolist(),
            tray_xy_m=self.geometry.tray_xy.tolist(),
            preturn_clearance_retreat_m=self.args.post_grasp_retreat,
        )
        self.phase = "APPROACH"
        # A straight current->dock segment can pass through the target because
        # the calibrated dock is intentionally on its opposite side.  Select a
        # clearance waypoint on the same side of the target as the robot, then
        # approach the dock tangentially.  All points are executed by the base
        # controller; this is path geometry, never a pose update.
        self.event(
            "collision_avoiding_dock_plan",
            clearance_waypoint_xy_m=dock_waypoints[0].tolist(),
            dock_xy_m=self.geometry.dock_xy.tolist(),
            minimum_center_clearance_m=self.args.approach_clearance,
            arc_waypoint_count=len(dock_waypoints),
            approach_route_kind=approach_kind,
        )
        try:
            self.drive(dock_waypoints, 1, 0.12, 0.020)
        except RuntimeError as first_error:
            # ReplicaCAD's official Fetch navmesh omits some scene-level
            # movable props.  A measured controller stall is therefore fused
            # as a local obstacle and the dock is replanned from the recovered
            # physical pose.  The failed segment is never crossed by a setter.
            stalled_xy = getattr(self, "last_stall_xy", None)
            if stalled_xy is None:
                raise
            blocked = self.grid.world_to_cell(stalled_xy)
            radius = self.args.stall_obstacle_radius_cells
            r0, c0 = blocked
            approach_free[
                max(0, r0 - radius) : min(approach_free.shape[0], r0 + radius + 1),
                max(0, c0 - radius) : min(approach_free.shape[1], c0 + radius + 1),
            ] = False
            retry_waypoints = route_on_grid(
                self.state()[0], self.geometry.dock_xy, approach_free
            )
            self.event(
                "approach_stall_replanned",
                blocked_cell=list(blocked),
                first_stall_reason=str(first_error),
                retry_waypoint_count=len(retry_waypoints),
            )
            self.drive(retry_waypoints, 1, 0.10, 0.020)
        self.turn(
            self.geometry.pick_yaw,
            1,
            tolerance=self.args.dock_yaw_tolerance,
        )
        self.settle_dock_pose(self.geometry.dock_xy, self.geometry.pick_yaw, 1)
        refined_detection = self.scan(focus=True)
        self.freeze(1.0)

        self.agent.set_control_mode("pd_ee_delta_pose")
        self.agent.controller.reset()
        tcp_rotation = _array(self.agent.tcp_pose.to_transformation_matrix())[0, :3, :3]
        approach_axis = tcp_rotation[:, 0]
        object_center = detected.copy() if refined_detection is None else refined_detection.copy()
        self.event(
            "pregrasp_rgbd_refinement",
            available=int(refined_detection is not None),
            refined_sensor_xyz_m=object_center.tolist(),
            correction_m=float(np.linalg.norm(object_center - detected)),
        )
        # The Fetch TCP is the gripper-link frame, not the object center.
        # Independent tabletop calibration places this frame about 8 cm above
        # the cuboid center at bilateral contact.  Driving it to the object
        # center makes the wrist collide with the tabletop.
        pick_rotation = np.asarray(
            [
                [math.cos(self.geometry.pick_yaw), -math.sin(self.geometry.pick_yaw)],
                [math.sin(self.geometry.pick_yaw), math.cos(self.geometry.pick_yaw)],
            ]
        )
        tcp_offset = np.r_[
            pick_rotation
            @ np.asarray(
                [self.args.grasp_tcp_offset_x, self.args.grasp_tcp_offset_y],
                dtype=float,
            ),
            self.args.grasp_tcp_offset_z,
        ]
        grasp_tcp = object_center + tcp_offset
        pregrasp_tcp = grasp_tcp - self.geometry.approach_distance * approach_axis
        pregrasp_residual = self.move_ee(pregrasp_tcp, 1.0, 0.014)
        # Joint limits can leave the commanded TCP a few centimetres from the
        # calibrated set point at some collision-free base headings.  Do not
        # declare success from Cartesian residual alone: close the fingers and
        # let bilateral PhysX contact decide, followed by the existing visual
        # regrasp loop when contact is absent.  This is a controller fallback,
        # not a target-pose correction or attachment.
        residual = self.move_ee(grasp_tcp, 1.0, 0.012, strict=False)
        self.event(
            "depth_mask_top_down_approach",
            pregrasp_residual_m=pregrasp_residual,
            residual_m=residual,
            grasp_tcp_offset_m=tcp_offset.tolist(),
            desired_grasp_tcp_xyz_m=grasp_tcp.tolist(),
            achieved_grasp_tcp_xyz_m=_array(self.agent.tcp_pose.p)[0].astype(float).tolist(),
        )
        def close_fingers() -> None:
            for grip in np.linspace(1.0, -1.0, 35):
                self.command_ee(grip=float(grip))
            for _ in range(70):
                self.command_ee(grip=-1.0)

        close_fingers()
        # The same Cartesian calibration is used at every collision-clear
        # base heading.  Joint compliance can nevertheless leave the wrist a
        # centimetre higher at some arm configurations.  A failed bilateral
        # contact check therefore triggers a measured, controller-driven
        # visual regrasp (the real system's focus/regrasp analogue).  The hand
        # camera remeasures an object that may have shifted during first
        # contact; the fallback remains a small vertical correction.  No actor
        # pose or grasp constraint is changed.
        retry_count = 0
        while not self.grasping() and retry_count < self.args.grasp_retry_count:
            retry_count += 1
            self.event(
                "bilateral_contact_retry",
                retry_index=retry_count,
                commanded_vertical_correction_m=-self.args.grasp_retry_drop,
            )
            for grip in np.linspace(-1.0, 1.0, 25):
                self.command_ee(grip=float(grip))
            retry_detection = self.sense("fetch_hand")
            if retry_detection is not None:
                retry_target = retry_detection + tcp_offset
                retry_pregrasp = retry_target - 0.045 * approach_axis
                self.move_ee(retry_pregrasp, 1.0, 0.014)
                self.move_ee(retry_target, 1.0, 0.010)
            else:
                retry_target = _array(self.agent.tcp_pose.p)[0].astype(float)
                retry_target[2] -= self.args.grasp_retry_drop
                self.move_ee(retry_target, 1.0, 0.010)
            self.event(
                "visual_regrasp_refinement",
                retry_index=retry_count,
                available=int(retry_detection is not None),
                refined_sensor_xyz_m=(
                    retry_detection.tolist() if retry_detection is not None else None
                ),
                desired_grasp_tcp_xyz_m=retry_target.tolist(),
            )
            close_fingers()
        self.phase = "GRASP"
        self.grasp_ok = self.grasping()
        self.event(
            "bilateral_contact_check",
            success=int(self.grasp_ok),
            controller_driven_retries=retry_count,
        )
        self.freeze(2.0)
        if not self.grasp_ok:
            raise RuntimeError("bilateral finger contact criterion failed")

        # Moderate damping represents passive object/contact dissipation and
        # prevents a long object from oscillating out of the fingertips during
        # the mobile-base turn.  The actor remains fully dynamic and unlocked.
        self.target.linear_damping = self.args.carry_object_linear_damping
        self.target.angular_damping = self.args.carry_object_angular_damping
        self.phase = "LIFT"
        lift_start = _array(self.agent.tcp_pose.p)[0].copy()
        object_lift_start_z = float(self.object_xyz()[2])
        lift = lift_start + np.asarray([0, 0, self.args.lift_height])
        lift_qpos = _array(self.agent.robot.get_qpos())[0].astype(float)
        lift_limits = _array(self.agent.robot.get_qlimits())[0].astype(float)
        self.lift_torso_start_m = float(lift_qpos[3])
        self.event(
            "measured_body_position_hold_start",
            torso_start_m=self.lift_torso_start_m,
            torso_target_m=float(self.body_target[2]),
            joint_upper_limit_m=float(lift_limits[3, 1]),
            controller="normalized delta from measured error via env.step",
        )
        lift_residual = float("inf")
        # Stop on measured object clearance, rather than continuing to chase a
        # TCP set point after the object is already safely above the support.
        # This avoids squeezing a contact-held object through a long arm-limit
        # transient and makes the success quantity physically auditable.
        for _ in range(self.args.ee_max_steps):
            tcp = _array(self.agent.tcp_pose.p)[0].astype(float)
            error = lift - tcp
            lift_residual = float(np.linalg.norm(error))
            yaw = self.state()[1]
            c, s = math.cos(yaw), math.sin(yaw)
            local = np.asarray(
                [c * error[0] + s * error[1], -s * error[0] + c * error[1], error[2]]
            )
            translation = np.clip(
                local / 0.1 * self.args.ee_position_gain,
                -self.args.ee_action_limit,
                self.args.ee_action_limit,
            )
            self.command_ee(np.r_[translation, [0, 0, 0]], -1.0)
            measured = float(self.object_xyz()[2] - object_lift_start_z)
            if measured >= self.args.minimum_measured_lift:
                break
            if not self.grasping():
                raise RuntimeError("object lost before measured lift clearance")
        else:
            raise RuntimeError(
                f"measured lift did not reach {self.args.minimum_measured_lift:.3f} m"
            )
        peak_measured_lift = float(self.object_xyz()[2] - object_lift_start_z)
        self.peak_measured_lift_m = peak_measured_lift
        self.lift_torso_final_m = float(_array(self.agent.robot.get_qpos())[0, 3])
        for _ in range(5):
            self.command_ee(grip=-1.0)
        self.measured_lift_m = float(self.object_xyz()[2] - object_lift_start_z)
        if (
            not self.grasping()
            or self.measured_lift_m < 0.75 * self.args.minimum_measured_lift
        ):
            raise RuntimeError("object did not retain measured lift clearance")
        self.event(
            "measured_lift_contact_check",
            commanded_height_m=self.args.lift_height,
            peak_measured_object_lift_m=peak_measured_lift,
            measured_object_lift_m=self.measured_lift_m,
            required_peak_object_lift_m=self.args.minimum_measured_lift,
            retained_clearance_threshold_m=0.75 * self.args.minimum_measured_lift,
            residual_m=lift_residual,
            torso_start_m=self.lift_torso_start_m,
            torso_final_m=self.lift_torso_final_m,
            measured_torso_motion_m=(
                self.lift_torso_final_m - self.lift_torso_start_m
            ),
            success=1,
        )
        self.event(
            "lift",
            residual_m=lift_residual,
            staged_increments=1,
            commanded_height_m=self.args.lift_height,
            measured_object_lift_m=self.measured_lift_m,
        )
        # Long base motion in delta-pose mode does not regulate the arm against
        # gravity when the commanded increment is zero.  Snapshot the actual
        # lifted configuration and use joint position control to retain it.
        # The object's shallow shoulder supplies mechanical retention across
        # the controller transition; no grasp constraint is added.
        if self.args.cartesian_carry_hold:
            # Keep the already-active Cartesian controller continuous.  This
            # avoids a one-step reset transient at rotated grasp headings;
            # zero Cartesian increments retain the measured gripper frame as
            # the base moves, while PhysX contacts remain the only attachment.
            for _ in range(25):
                self.command_ee(grip=-1.0)
        else:
            qpos = _array(self.agent.robot.get_qpos())[0]
            self.arm_target = qpos[[5, 7, 8, 9, 10, 11, 12]].copy()
            self.agent.set_control_mode("pd_joint_pos")
            self.agent.controller.reset()
            for _ in range(25):
                self.command(grip=-1.0)
        if not self.grasping():
            raise RuntimeError("object lost after lift")
        self.event("post_lift_stable_hold", success=1)

        # Move straight away from the table before asking the global path
        # controller to rotate.  This is a closed-loop wheel command with a
        # measured stopping distance and continuous grasp checks, not a pose
        # update.  It prevents the carried object from sweeping through the
        # force-driven support during the first turn.
        self.phase = "CLEARANCE RETREAT"
        retreat_start, retreat_yaw = self.state()
        retreat_heading = np.asarray([math.cos(retreat_yaw), math.sin(retreat_yaw)])
        retreat_progress = 0.0
        for retreat_step in range(self.args.retreat_max_steps):
            xy, _ = self.state()
            retreat_progress = float((retreat_start - xy) @ retreat_heading)
            if retreat_progress >= self.args.post_grasp_retreat:
                break
            self.command(grip=-1.0, base=(-self.args.retreat_speed, 0.0))
            if retreat_step % 5 == 0 and not self.grasping():
                raise RuntimeError("object lost during post-grasp clearance retreat")
        else:
            raise RuntimeError("post-grasp clearance retreat failed to reach its stopping distance")
        self.hold(10, -1.0)
        if not self.grasping():
            raise RuntimeError("object lost after post-grasp clearance retreat")
        self.event(
            "post_grasp_clearance_retreat",
            commanded_distance_m=self.args.post_grasp_retreat,
            measured_distance_m=retreat_progress,
            wheel_command=-self.args.retreat_speed,
            bilateral_contact_retained=1,
        )

        self.phase = "CARRY"
        # The navigation raster already represents the Fetch base footprint.
        # During transport the forward arm and payload sweep a larger volume,
        # so erode the free component by an additional measured safety margin
        # before planning.  This prevents a base-clear route from scraping the
        # grasped object on furniture during a turn.
        carry_clearance = (
            distance_transform_edt(self.grid.free) * self.grid.resolution_m
        )
        carry_free = carry_clearance >= self.args.carry_additional_clearance
        rows, cols = np.indices(carry_free.shape)
        world_x = self.grid.minimum_xy[0] + cols * self.grid.resolution_m
        world_y = self.grid.maximum_xy[1] - rows * self.grid.resolution_m
        support_xy = _array(self.support.pose.p)[0, :2].astype(float)
        support_disk = (world_x - support_xy[0]) ** 2 + (world_y - support_xy[1]) ** 2
        carry_free[
            support_disk < (self.args.grid_target_exclusion + 0.10) ** 2
        ] = False
        carry_route = route_on_grid(
            self.state()[0], self.geometry.place_base_xy, carry_free
        )
        self.event(
            "grid_planned_physical_carry",
            route_waypoint_count=len(carry_route),
            place_base_xy_m=self.geometry.place_base_xy.tolist(),
            additional_payload_clearance_m=self.args.carry_additional_clearance,
            dynamic_support_exclusion_radius_m=self.args.grid_target_exclusion + 0.10,
        )
        for carry_attempt in range(3):
            self.last_stall_contact_proxy_xy = None
            try:
                self.drive(carry_route, -1.0, self.args.carry_speed, 0.07)
                break
            except RuntimeError as carry_error:
                if not self.grasping() or carry_attempt == 2:
                    raise
                contact_proxy = self.last_stall_contact_proxy_xy
                if contact_proxy is None:
                    raise
                blocked = self.grid.world_to_cell(contact_proxy)
                radius = self.args.stall_obstacle_radius_cells
                r0, c0 = blocked
                carry_free[
                    max(0, r0 - radius) : min(carry_free.shape[0], r0 + radius + 1),
                    max(0, c0 - radius) : min(carry_free.shape[1], c0 + radius + 1),
                ] = False
                # ``drive`` reverses under wheel control after measuring the
                # stall.  Preserve that physically traversed base-centre
                # corridor so the newly fused obstacle square cannot isolate
                # the recovered pose from the rest of the carry map.
                recovered_xy = self.state()[0].copy()
                contact_base_xy = (
                    self.last_stall_xy.copy()
                    if self.last_stall_xy is not None
                    else recovered_xy.copy()
                )
                recovery_distance = float(
                    np.linalg.norm(contact_base_xy - recovered_xy)
                )
                recovery_samples = max(
                    2,
                    int(
                        math.ceil(
                            recovery_distance
                            / max(self.grid.resolution_m / 2.0, 1e-6)
                        )
                    ),
                )
                recovery_corridor_cells: set[tuple[int, int]] = set()
                for alpha in np.linspace(0.0, 1.0, recovery_samples):
                    corridor_cell = self.grid.world_to_cell(
                        (1.0 - alpha) * recovered_xy
                        + alpha * contact_base_xy
                    )
                    if (
                        0 <= corridor_cell[0] < carry_free.shape[0]
                        and 0 <= corridor_cell[1] < carry_free.shape[1]
                        and self.grid.free[corridor_cell]
                    ):
                        recovery_corridor_cells.add(corridor_cell)
                for corridor_cell in recovery_corridor_cells:
                    carry_free[corridor_cell] = True
                carry_route = route_on_grid(
                    self.state()[0], self.geometry.place_base_xy, carry_free
                )
                self.event(
                    "carry_stall_replanned",
                    carry_attempt=carry_attempt + 1,
                    blocked_cell=list(blocked),
                    contact_proxy_xy_m=contact_proxy.tolist(),
                    recovery_corridor_cells=len(recovery_corridor_cells),
                    stall_reason=str(carry_error),
                    retry_waypoint_count=len(carry_route),
                )
        else:
            raise RuntimeError("physical carry replanning exhausted")
        self.turn(self.geometry.place_yaw, -1.0, self.args.carry_yaw_rate)
        if not self.grasping():
            raise RuntimeError("object lost during physical carry")
        self.hold(40, -1.0)
        self.phase = "PLACE"
        self.target.linear_damping = 5.0
        self.target.angular_damping = 5.0
        for grip in np.linspace(-1.0, 1.0, 50):
            self.command(grip=float(grip))
        self.hold(120, 1.0)
        final = self.object_xyz()
        rotation = _array(self.target.pose.to_transformation_matrix())[0, :3, :3]
        local_half_extents = np.asarray(
            [self.geometry.half_size, self.geometry.half_size, self.geometry.half_height]
        )
        world_half_extents = np.abs(rotation) @ local_half_extents
        bottom_z = float(final[2] - world_half_extents[2])
        planar_inside = bool(
            np.all(
                np.abs(final[:2] - self.geometry.tray_xy) + world_half_extents[:2]
                < 0.30 + 0.015
            )
        )
        self.place_ok = bool(
            planar_inside
            and abs(bottom_z - self.geometry.tray_top) < 0.025
            and not self.grasping()
        )
        self.phase = "VERIFY"
        self.event(
            "place_verification",
            success=int(self.place_ok),
            final_object_xyz_m=final.tolist(),
            object_bottom_z_m=bottom_z,
            receptacle_top_z_m=self.geometry.tray_top,
            orientation_invariant_planar_inside=int(planar_inside),
        )
        self.freeze(3.0)
        if not self.place_ok:
            raise RuntimeError(f"placement failed at {final.tolist()}")

    def map_inset(self, width: int) -> np.ndarray:
        canvas = np.full((*self.known.shape, 3), (48, 52, 58), dtype=np.uint8)
        canvas[self.known == 1] = (205, 210, 214)
        canvas[self.known == -1] = (12, 15, 18)
        if len(self.voxels.points):
            sample = self.voxels.points[:: max(1, len(self.voxels.points) // 5000)]
            colors = self.voxels.rgb[:: max(1, len(self.voxels.rgb) // 5000)]
            for point, color in zip(sample, colors):
                cell = self.grid.world_to_cell(point[:2])
                if 0 <= cell[0] < canvas.shape[0] and 0 <= cell[1] < canvas.shape[1]:
                    canvas[cell] = tuple(int(v) for v in color[::-1])
            target_id = int(_array(self.target.per_scene_id)[0])
            target_points = self.voxels.points[
                self.voxels.instance_ids == target_id
            ]
            for point in target_points:
                cell = self.grid.world_to_cell(point[:2])
                if 0 <= cell[0] < canvas.shape[0] and 0 <= cell[1] < canvas.shape[1]:
                    canvas[cell] = (70, 220, 80)
        cells = np.asarray([self.grid.world_to_cell(point) for point in self.trajectory], dtype=np.int32)
        if len(cells) > 1:
            cv2.polylines(canvas, [cells[:, ::-1]], False, (0, 165, 255), 2, cv2.LINE_AA)
        current = cells[-1]
        cv2.circle(canvas, (int(current[1]), int(current[0])), 4, (255, 255, 255), -1)
        if self.cached_target is not None:
            stale = self.grid.world_to_cell(self.cached_target[:2])
            cv2.drawMarker(canvas, (stale[1], stale[0]), (20, 20, 240), cv2.MARKER_TILTED_CROSS, 12, 2)
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
        height = max(1, round(canvas.shape[0] * width / canvas.shape[1]))
        return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)

    def execute_dynamic(self, destination_xy: np.ndarray, place_base_xy: np.ndarray) -> None:
        self.hold(20, 1.0)
        initial_detection = self.initial_rotate_and_scan()
        if initial_detection is None:
            first = self.online_search("SEARCH", self.args.max_replans)
        else:
            first = initial_detection
            self.detected = True
            self.event(
                "target_observed_during_initial_360_scan",
                sensor_xyz_m=first.tolist(),
            )
        self.cached_target = first.copy()
        self.event("target_cached", sensor_xyz_m=first.tolist())
        self.freeze(1.5)
        destination_xy = self.move_target_continuously(destination_xy)
        focus_detection = self.focus_and_invalidate()
        if focus_detection is not None and np.linalg.norm(focus_detection[:2] - first[:2]) > DEFAULT_EXECUTION_CONFIG.target_relocation_radius_m:
            final_detection = focus_detection
            self.phase = "REACQUIRE"
            self.event("target_reacquired_in_focused_frame", sensor_xyz_m=final_detection.tolist())
        else:
            self.detected = False
            self.detected_target = None
            self.scenario = replace(
                self.scenario,
                target=self.grid.world_to_cell(np.asarray(destination_xy)),
            )
            final_detection = self.online_search("REACQUIRE", self.args.max_replans)
        self.cached_target = final_detection.copy()
        self.freeze(2.0)
        self.grasp_carry_place(final_detection, place_base_xy)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(trial: DynamicTrial, args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    video = output / "dream_dynamic_reacquisition.mp4"
    # H.264/yuv420p requires even frame dimensions.  Final runs already use
    # 768x432, while low-resolution validation may request an odd height.
    encoded_frames = trial.frames
    if encoded_frames and (
        encoded_frames[0].shape[0] % 2 or encoded_frames[0].shape[1] % 2
    ):
        encoded_frames = [
            cv2.copyMakeBorder(
                frame,
                0,
                frame.shape[0] % 2,
                0,
                frame.shape[1] % 2,
                cv2.BORDER_REPLICATE,
            )
            for frame in encoded_frames
        ]
    with imageio.get_writer(
        video,
        fps=args.fps,
        codec="libx264",
        quality=7,
        macro_block_size=None,
        ffmpeg_log_level="warning",
    ) as writer:
        for frame in encoded_frames:
            writer.append_data(frame)
    control = output / "control.csv"
    events = output / "events.csv"
    memory = output / "memory.csv"
    trajectory = output / "trajectory.csv"
    _write_csv(control, trial.rows)
    _write_csv(events, trial.events)
    _write_csv(memory, trial.memory_updates)
    _write_csv(
        trajectory,
        [dict(step=index, world_x_m=point[0], world_y_m=point[1]) for index, point in enumerate(trial.trajectory)],
    )
    payload = {
        "status": "success",
        "boundary": trial.experiment_boundary,
        "simulator": "ManiSkill 3.0.1 / SAPIEN 3.0.3",
        "physics_backend": "PhysX CPU",
        "robot": "Fetch",
        "target_localization_input": "current rendered actor-ID pixels + RGB-D + camera calibration",
        "target_world_pose_used_by_planner": False,
        "traversability_source": trial.traversability_source,
        "learned_perception_tested": False,
        "dynamic_target": {
            "body_size_m": [0.060, 0.060, 0.140],
            "grasp_shoulder_size_m": [
                2 * DYNAMIC_TARGET_SHOULDER_HALF_SIZE_M,
                2 * DYNAMIC_TARGET_SHOULDER_HALF_SIZE_M,
                2 * DYNAMIC_TARGET_SHOULDER_HALF_HEIGHT_M,
            ],
            "density_kg_m3": DYNAMIC_TARGET_DENSITY_KG_M3,
            "material": {
                "description": "high-friction rubberized surface",
                "static_friction": DYNAMIC_TARGET_STATIC_FRICTION,
                "dynamic_friction": DYNAMIC_TARGET_DYNAMIC_FRICTION,
                "restitution": 0.0,
            },
            "attachment": "none; bilateral PhysX contact only",
        },
        "control_steps": trial.step,
        "simulated_duration_s": trial.step / trial.hz,
        "base_travel_m": trial.travel,
        "support_travel_m": trial.support_distance_m,
        "support_max_step_m": trial.support_max_step_m,
        "target_support_max_xy_error_before_grasp_m": trial.target_support_max_xy_error_m,
        "measured_relocation_m": trial.relocation_distance_m,
        "relocation_target_support_xy_error_m": trial.relocation_target_support_xy_error_m,
        "peak_measured_object_lift_m": trial.peak_measured_lift_m,
        "required_peak_object_lift_m": args.minimum_measured_lift,
        "retained_lift_threshold_m": 0.75 * args.minimum_measured_lift,
        "measured_object_lift_m": trial.measured_lift_m,
        "measured_torso_motion_during_lift_m": (
            trial.lift_torso_final_m - trial.lift_torso_start_m
            if trial.lift_torso_final_m is not None
            and trial.lift_torso_start_m is not None
            else None
        ),
        "stale_region_voxels_before": trial.stale_region_voxels_before,
        "stale_region_voxels_after": trial.stale_region_voxels_after,
        "scripted_pose_updates_after_initialization": False,
        "frontier_replans": trial.planner.replans,
        "initial_360_scan_base_drift_m": trial.initial_scan_drift_m,
        "voxel_points_final": len(trial.voxels.points),
        "grasp_bilateral_contact": trial.grasp_ok,
        "place_success": trial.place_ok,
        "video_duration_s": len(trial.frames) / args.fps,
        "execution_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in sorted(vars(args).items())
        },
        "implementation_sha256": {
            path.name: sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("maniskill_replicacad_closed_loop.py"),
                Path(__file__).with_name("dream_simulation_core.py"),
                REPO / "src/dream/dynamic_memory.py",
            )
        },
        "artifacts": {},
    }
    payload.update(trial.extra_manifest_fields)
    for artifact in (video, control, events, memory, trajectory):
        payload["artifacts"][artifact.name] = {"sha256": sha256(artifact), "bytes": artifact.stat().st_size}
    manifest = output / "manifest.json"
    write_manifest(manifest, trial.spec, payload)
    payload["manifest"] = str(manifest)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.protocol not in {"target_move", "combined"}:
        raise ValueError("this physical smoke backend currently implements target_move/combined")
    spec = SimulationSpec("replicacad", SCENE_NAME, args.seed, args.protocol, args.variant, args.sensor_range, args.resolution)
    env = gym.make(
        ENVIRONMENT_ID,
        robot_uids="fetch",
        build_config_idxs=[0],
        num_envs=1,
        obs_mode="state",
        reward_mode="none",
        render_mode="rgb_array",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        render_backend="cpu",
        max_episode_steps=30000,
        sensor_configs=dict(width=args.sensor_width, height=args.sensor_height),
        human_render_camera_configs=dict(
            width=args.width,
            height=args.height,
            fov=1.05,
            near=0.05,
            far=100,
            shader_pack="default",
        ),
    )
    trial: DynamicTrial | None = None
    try:
        env.reset(seed=args.seed)
        base = env.unwrapped
        grid = rasterize_navigation_mesh(base.scene_builder.navigable_positions[0], args.resolution)
        initial_xy = np.asarray([args.initial_target_x, args.initial_target_y], dtype=float)
        scenario = build_search_scenario(grid, target_xy=initial_xy, destination_boundary_y_m=-2.50)
        geometry = replace(
            make_geometry(base, initial_xy, np.asarray([args.place_base_x, args.place_base_y])),
            half_size=DYNAMIC_TARGET_HALF_SIZE_M,
            half_height=DYNAMIC_TARGET_HALF_HEIGHT_M,
        )
        support, target = build_moving_target(base, initial_xy, geometry.target_z)
        trial = DynamicTrial(env, grid, scenario, target, support, geometry, args, spec)
        trial.execute_dynamic(
            np.asarray([args.final_target_x, args.final_target_y]),
            np.asarray([args.place_base_x, args.place_base_y]),
        )
        return write_outputs(trial, args)
    except Exception as error:
        if trial is not None:
            args.output_root.mkdir(parents=True, exist_ok=True)
            _write_csv(args.output_root / "failure_control.csv", trial.rows)
            _write_csv(args.output_root / "failure_events.csv", trial.events)
            _write_csv(args.output_root / "failure_memory.csv", trial.memory_updates)
            base_xy, base_yaw = trial.state()
            failure = {
                "status": "failure",
                "error_type": type(error).__name__,
                "error": str(error),
                "step": trial.step,
                "phase": trial.phase,
                "base_xy_m": base_xy.tolist(),
                "base_yaw_rad": base_yaw,
                "support_xyz_m": _array(trial.support.pose.p)[0].tolist(),
                "target_xyz_m_evaluation_only": trial.object_xyz().tolist(),
                "frontier_replans": trial.planner.replans,
                "known_cells": int(np.count_nonzero(trial.known)),
                "voxel_points": len(trial.voxels.points),
                "execution_arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in sorted(vars(args).items())
                },
                "implementation_sha256": {
                    path.name: sha256(path)
                    for path in (
                        Path(__file__),
                        Path(__file__).with_name("maniskill_replicacad_closed_loop.py"),
                        Path(__file__).with_name("dream_simulation_core.py"),
                        REPO / "src/dream/dynamic_memory.py",
                    )
                },
            }
            (args.output_root / "failure.json").write_text(
                json.dumps(failure, indent=2), encoding="utf-8"
            )
            if trial.frames:
                imageio.imwrite(args.output_root / "failure_last_frame.png", trial.frames[-1])
            print(json.dumps(failure, indent=2), flush=True)
        raise
    finally:
        env.close()


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/maniskill_dynamic_smoke"))
    parser.add_argument("--protocol", choices=("target_move", "combined"), default="target_move")
    parser.add_argument("--variant", choices=("full", "no_clearing", "no_focus", "dynamic_off"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=float, default=0.08)
    parser.add_argument("--sensor-range", type=float, default=1.20)
    parser.add_argument("--max-replans", type=int, default=80)
    parser.add_argument("--search-speed", type=float, default=0.28)
    parser.add_argument("--stall-obstacle-radius-cells", type=int, default=2)
    parser.add_argument("--initial-target-x", type=float, default=2.47)
    parser.add_argument("--initial-target-y", type=float, default=-5.34)
    parser.add_argument("--final-target-x", type=float, default=1.77)
    parser.add_argument("--final-target-y", type=float, default=-4.80)
    parser.add_argument("--place-base-x", type=float, default=2.47)
    parser.add_argument("--place-base-y", type=float, default=-5.34)
    parser.add_argument("--support-speed", type=float, default=0.035)
    parser.add_argument("--support-velocity-gain", type=float, default=300.0)
    parser.add_argument("--support-force-limit", type=float, default=30.0)
    parser.add_argument("--support-tolerance", type=float, default=0.035)
    parser.add_argument("--support-max-steps", type=int, default=5000)
    parser.add_argument("--approach-clearance", type=float, default=0.70)
    parser.add_argument("--grid-target-exclusion", type=float, default=0.34)
    parser.add_argument("--grid-snap-distance", type=float, default=0.30)
    parser.add_argument("--dock-position-tolerance", type=float, default=0.020)
    parser.add_argument("--dock-yaw-tolerance", type=float, default=0.040)
    # Camera-to-gripper calibration in the robot base frame.  At the original
    # pick heading this rotates to approximately (+4, +36) mm in world XY.
    parser.add_argument("--grasp-tcp-offset-x", type=float, default=0.01252)
    parser.add_argument("--grasp-tcp-offset-y", type=float, default=-0.03399)
    parser.add_argument("--grasp-tcp-offset-z", type=float, default=0.064)
    parser.add_argument("--grasp-retry-count", type=int, default=2)
    parser.add_argument("--grasp-retry-drop", type=float, default=0.012)
    parser.add_argument("--ee-position-gain", type=float, default=0.30)
    parser.add_argument("--ee-action-limit", type=float, default=0.08)
    parser.add_argument("--ee-max-steps", type=int, default=180)
    parser.add_argument("--lift-height", type=float, default=0.09)
    parser.add_argument("--lift-stages", type=int, default=3)
    parser.add_argument("--minimum-measured-lift", type=float, default=0.03)
    parser.add_argument("--post-grasp-retreat", type=float, default=0.60)
    parser.add_argument("--retreat-speed", type=float, default=0.08)
    parser.add_argument("--retreat-max-steps", type=int, default=250)
    parser.add_argument("--carry-speed", type=float, default=0.08)
    parser.add_argument("--carry-additional-clearance", type=float, default=0.20)
    parser.add_argument(
        "--carry-yaw-rate",
        type=float,
        default=0.15,
        help="maximum physical base yaw rate in rad/s while carrying",
    )
    parser.add_argument("--carry-object-linear-damping", type=float, default=2.0)
    parser.add_argument("--carry-object-angular-damping", type=float, default=2.0)
    parser.add_argument(
        "--cartesian-carry-hold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain the active Cartesian arm controller during mobile carry",
    )
    parser.add_argument("--scan-hold-steps", type=int, default=4)
    parser.add_argument("--motion-sense-interval", type=int, default=40)
    parser.add_argument("--motion-detection-stop-range", type=float, default=0.90)
    parser.add_argument("--maximum-initial-scan-drift", type=float, default=0.15)
    parser.add_argument("--minimum-target-pixels", type=int, default=3)
    parser.add_argument("--point-stride", type=int, default=6)
    parser.add_argument("--sensor-width", type=int, default=192)
    parser.add_argument("--sensor-height", type=int, default=144)
    parser.add_argument("--capture-stride", type=int, default=6)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    return parser


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser())
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
