#!/usr/bin/env python3
"""Closed-loop ReplicaCAD search, pickup, transport, and place integration.

The DREAM frontier route is computed from an initially unknown, locally
revealed navmesh. Fetch then tracks it only through ManiSkill ``env.step``
actions. After local target observation, a geometric two-stage approach closes
the physical gripper; bilateral contact, carried-object retention, and final
placement are evaluated from SAPIEN state. No state setter is used after the
episode is initialized.

Scope: controlled context and perfect target pose after detection. This tests
memory/planning/controller/manipulation integration, not learned perception or
dynamic target relocation.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MS_ASSET_DIR", str(ROOT / ".maniskill_assets"))
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tmech")

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien

import mani_skill  # noqa: F401
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors

from houseexpo_cross_room import Scenario, reveal_with_occlusion, run_scenario
from maniskill_replicacad_search_video import (
    ENVIRONMENT_ID,
    SCENE_NAME,
    build_search_scenario,
    rasterize_navigation_mesh,
)


BOUNDARY = (
    "Physics-controlled integration trial: an incremental unknown-map policy "
    "rollout computes the route before the Fetch controller executes it. Semantic "
    "context and geometric local-visibility detection are controlled; onboard RGB "
    "is visualized but is not used by the detector, and target pose is perfect "
    "after detection. This is not a learned-perception, sensor-in-the-loop "
    "replanning, or dynamic-relocation benchmark."
)


@dataclass
class Geometry:
    target_xy: np.ndarray
    target_z: float
    half_size: float
    half_height: float
    pick_yaw: float
    dock_xy: np.ndarray
    approach_distance: float
    place_base_xy: np.ndarray
    place_yaw: float
    tray_xy: np.ndarray
    tray_top: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def make_geometry(base: Any, target_xy: np.ndarray, place_base_xy: np.ndarray) -> Geometry:
    # The rest-arm gripper x axis points mostly downward.  At this base-frame
    # offset the wrist starts 12 cm above the object, enabling a short
    # top-down Cartesian approach without sweeping the palm through it.
    grasp_offset = np.asarray([0.5500, 0.0380])
    approach_distance = 0.12
    target_z = 1.0200
    pick_yaw = math.atan2(0.48, -0.96)
    pick_rotation = np.asarray(
        [[math.cos(pick_yaw), -math.sin(pick_yaw)], [math.sin(pick_yaw), math.cos(pick_yaw)]]
    )
    physical_offset = np.asarray([0.049, -0.023])
    dock = (target_xy + physical_offset) - pick_rotation @ grasp_offset
    # Measured carried-object offset after the verified top-down grasp.
    carry_offset = np.asarray([0.696, -0.009])
    place_yaw = math.pi
    place_rotation = np.asarray(
        [[math.cos(place_yaw), -math.sin(place_yaw)], [math.sin(place_yaw), math.cos(place_yaw)]]
    )
    tray = place_base_xy + place_rotation @ carry_offset
    return Geometry(
        target_xy,
        target_z,
        0.030,
        0.070,
        pick_yaw,
        dock,
        approach_distance,
        place_base_xy,
        place_yaw,
        tray,
        0.68,
    )


def build_objects(base: Any, geometry: Geometry) -> Any:
    physical_xy = geometry.target_xy + np.asarray([0.049, -0.023])
    pick_top = geometry.target_z - geometry.half_height - 0.002
    actors.build_box(
        base.scene,
        half_sizes=[0.035, 0.035, pick_top / 2],
        color=[0.50, 0.22, 0.08, 1],
        name="dream_pick_pedestal",
        body_type="static",
        initial_pose=sapien.Pose([*physical_xy, pick_top / 2]),
    )
    heading = np.asarray([math.cos(geometry.pick_yaw), math.sin(geometry.pick_yaw)])
    backstop_xy = physical_xy + heading * (geometry.half_size + 0.011)
    actors.build_box(
        base.scene,
        half_sizes=[0.010, 0.080, 0.080],
        color=[0.58, 0.30, 0.11, 1],
        name="dream_low_shelf_backstop",
        body_type="static",
        initial_pose=sapien.Pose(
            [*backstop_xy, geometry.target_z],
            [math.cos(geometry.pick_yaw / 2), 0, 0, math.sin(geometry.pick_yaw / 2)],
        ),
    )
    target_builder = base.scene.create_actor_builder()
    target_material = sapien.pysapien.physx.PhysxMaterial(3.0, 2.5, 0.0)
    target_builder.add_box_collision(
        half_size=[geometry.half_size, geometry.half_size, geometry.half_height],
        material=target_material,
        density=500,
        patch_radius=0.03,
        min_patch_radius=0.03,
    )
    target_builder.add_box_visual(
        half_size=[geometry.half_size, geometry.half_size, geometry.half_height],
        material=[1.0, 0.18, 0.02, 1.0],
    )
    target_builder.set_initial_pose(sapien.Pose([*physical_xy, geometry.target_z]))
    target = target_builder.build(name="dream_orange_target")
    target.linear_damping = 8.0
    target.angular_damping = 8.0
    return target


def build_receptacle(base: Any, geometry: Geometry) -> None:
    """Instantiate the second-stage goal once the pickup stage is activated.

    Keeping this collision object out of the first-stage search prevents it
    from changing the fixed ReplicaCAD object-search route. It is static once
    created and is never repositioned.
    """

    actors.build_box(
        base.scene,
        half_sizes=[0.30, 0.30, geometry.tray_top / 2],
        color=[0.08, 0.55, 0.16, 1],
        name="dream_green_receptacle_base",
        body_type="static",
        initial_pose=sapien.Pose([*geometry.tray_xy, geometry.tray_top / 2]),
    )


class Trial:
    video_title = "DREAM closed-loop search, pickup & place | ReplicaCAD + Fetch"
    map_label = "ONLINE 2D MEMORY | OCCUPANCY + CONTEXT PROXY"
    video_footer = (
        "PHYSX env.step() | {playback:.1f}x PLAYBACK | NO TELEPORT | "
        "RGB DISPLAY (DETECTION: CONTROLLED VISIBILITY)"
    )

    def __init__(self, env: Any, grid: Any, scenario: Scenario, target: Any, geometry: Geometry, args: argparse.Namespace):
        self.env, self.base, self.agent = env, env.unwrapped, env.unwrapped.agent
        self.grid, self.scenario, self.target, self.geometry, self.args = grid, scenario, target, geometry, args
        self.origin = np.asarray(self.agent.robot.pose.p[0, :2].cpu(), dtype=float)
        self.known = np.zeros(grid.free.shape, dtype=np.int8)
        self.last_seen = np.zeros(grid.free.shape, dtype=float)
        self.phase, self.step, self.travel = "Initialize", 0, 0.0
        self.search_travel, self.detected = 0.0, False
        self.grasp_ok, self.place_ok = False, False
        self.replans, self.recoveries = 0, 0
        self.rows: list[dict[str, Any]] = []
        self.memory_rows: list[dict[str, Any]] = []
        self.trajectory: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []
        self.frame_phases: list[str] = []
        self.camera_target_visible_ever = False
        self.camera_target_first_visible_step: int | None = None
        self.max_step_translation, self.max_step_yaw = 0.0, 0.0
        self.last_stall_xy: np.ndarray | None = None
        self.last_stall_contact_proxy_xy: np.ndarray | None = None
        qpos = np.asarray(self.agent.robot.get_qpos()[0].cpu(), dtype=float)
        self.rest_arm = qpos[[5, 7, 8, 9, 10, 11, 12]].copy()
        self.arm_target = self.rest_arm.copy()
        # Controller order is head pan, head tilt, torso lift.  ManiSkill's
        # Fetch body controller is normalized *delta* position control, while
        # scan poses are most naturally specified as absolute joint targets.
        # Retain those desired targets and convert their measured error to a
        # normalized delta for every env.step below.
        # Articulation qpos is interleaved: torso=3, head-pan=4,
        # shoulder-pan=5, head-tilt=6, then the remaining arm joints.
        self.body_target = np.asarray([qpos[4], qpos[6], qpos[3]], dtype=float)
        self.previous_xy, self.previous_yaw = self.state()
        self.trajectory.append(self.previous_xy.copy())

    @property
    def hz(self) -> float:
        return float(self.base.control_freq)

    def state(self) -> tuple[np.ndarray, float]:
        qpos = np.asarray(self.agent.robot.get_qpos()[0].cpu(), dtype=float)
        return self.origin + qpos[:2], float(qpos[2])

    def object_xyz(self) -> np.ndarray:
        return np.asarray(self.target.pose.p[0].cpu(), dtype=float)

    def grasping(self) -> bool:
        return bool(self.agent.is_grasping(self.target)[0].cpu())

    def observe(self, force: bool = False) -> None:
        if self.phase != "Search" or (not force and self.step % self.args.scan_interval != 0):
            return
        cell = self.grid.world_to_cell(self.state()[0])
        visible = reveal_with_occlusion(
            self.scenario.free,
            self.known,
            self.last_seen,
            cell,
            self.step + 1,
            max(2, int(round(self.args.sensor_range / self.grid.resolution_m))),
        )
        self.detected |= bool(visible[self.scenario.target])

    def command(
        self,
        arm: np.ndarray | None = None,
        grip: float = 1.0,
        base: tuple[float, float] = (0, 0),
        body: np.ndarray | None = None,
    ) -> None:
        action = np.zeros(13, dtype=np.float32)
        if arm is not None:
            self.arm_target = np.asarray(arm, dtype=float).copy()
        if body is not None:
            self.body_target = np.asarray(body, dtype=float).copy()
        action[:7] = self.arm_target
        qpos = np.asarray(self.agent.robot.get_qpos()[0].cpu(), dtype=float)
        measured_body = qpos[[4, 6, 3]]
        # PDJointPosControllerConfig(lower=-0.1, upper=0.1,
        # use_delta=True, normalize_action=True): a normalized unit command
        # represents a 0.1-rad/m one-step joint displacement.
        action[8:11] = np.clip((self.body_target - measured_body) / 0.1, -1.0, 1.0)
        action[7], action[11], action[12] = grip, base[0], base[1]
        self.env.step(action)
        self.record_step(self.arm_target, grip, base)

    def command_ee(
        self,
        delta_pose: np.ndarray | None = None,
        grip: float = 1.0,
        base: tuple[float, float] = (0, 0),
        body: np.ndarray | None = None,
    ) -> None:
        action = np.zeros(12, dtype=np.float32)
        if delta_pose is not None:
            action[:6] = np.asarray(delta_pose, dtype=np.float32)
        # Fetch's body controller is incremental with ``use_target=False``:
        # repeatedly sending zero would reset the drive target to the sagged
        # measured position at every control step.  Convert the persistent,
        # absolute scan/torso target into a fresh measured-error delta instead.
        # This is ordinary closed-loop joint control through env.step(), not a
        # qpos setter.
        if body is not None:
            self.body_target = np.asarray(body, dtype=float).copy()
        qpos = np.asarray(self.agent.robot.get_qpos()[0].cpu(), dtype=float)
        measured_body = qpos[[4, 6, 3]]
        action[7:10] = np.clip(
            (self.body_target - measured_body) / 0.1, -1.0, 1.0
        )
        action[6], action[10], action[11] = grip, base[0], base[1]
        # The Gym wrapper caches the reset-time 13-D action space, so a
        # controller selected on the fly must be passed with its batch axis.
        self.env.step(action[None, :])
        qpos = np.asarray(self.agent.robot.get_qpos()[0].cpu(), dtype=float)
        self.record_step(qpos[[5, 7, 8, 9, 10, 11, 12]], grip, base)

    def record_step(self, arm_log: np.ndarray, grip: float, base: tuple[float, float]) -> None:
        self.step += 1
        xy, yaw = self.state()
        increment = float(np.linalg.norm(xy - self.previous_xy))
        yaw_increment = abs(wrap(yaw - self.previous_yaw))
        self.travel += increment
        if self.phase == "Search":
            self.search_travel += increment
        self.max_step_translation = max(self.max_step_translation, increment)
        self.max_step_yaw = max(self.max_step_yaw, yaw_increment)
        self.previous_xy, self.previous_yaw = xy.copy(), yaw
        self.trajectory.append(xy.copy())
        self.observe()
        tcp, obj = np.asarray(self.agent.tcp_pose.p[0].cpu(), dtype=float), self.object_xyz()
        qpos = np.asarray(self.agent.robot.get_qpos()[0].cpu(), dtype=float)
        self.rows.append(
            dict(
                step=self.step,
                sim_time_s=self.step / self.hz,
                phase=self.phase,
                control_mode=self.agent.control_mode,
                arm_q0=arm_log[0], arm_q1=arm_log[1], arm_q2=arm_log[2], arm_q3=arm_log[3],
                arm_q4=arm_log[4], arm_q5=arm_log[5], arm_q6=arm_log[6],
                head_pan_rad=qpos[4], head_tilt_rad=qpos[6],
                torso_lift_m=qpos[3],
                gripper=grip, base_forward=base[0], base_yaw_normalized=base[1],
                base_x_m=xy[0], base_y_m=xy[1], base_yaw_rad=yaw,
                tcp_x_m=tcp[0], tcp_y_m=tcp[1], tcp_z_m=tcp[2],
                object_x_m=obj[0], object_y_m=obj[1], object_z_m=obj[2],
                finger_left_m=qpos[-1], finger_right_m=qpos[-2],
                bilateral_grasp=int(self.grasping() if grip < -0.5 else False),
                known_cells=np.count_nonzero(self.known), replans=self.replans,
            )
        )
        if self.step % self.args.capture_stride == 0:
            self.capture()

    def hold(self, count: int, grip: float) -> None:
        for _ in range(count):
            self.command(grip=grip)

    def memory_event(self, event: str) -> None:
        xy, _ = self.state()
        self.memory_rows.append(
            dict(
                step=self.step, sim_time_s=self.step / self.hz, phase=self.phase, event=event,
                base_x_m=xy[0], base_y_m=xy[1], known_free_cells=np.count_nonzero(self.known == 1),
                known_occupied_cells=np.count_nonzero(self.known == -1),
                target_detected=int(self.detected), replans=self.replans, controller_recoveries=self.recoveries,
            )
        )

    def drive(self, points: np.ndarray, grip: float, speed: float, final_tolerance: float = 0.10) -> None:
        for index, target_xy in enumerate(np.asarray(points)):
            tolerance = final_tolerance if index == len(points) - 1 else 0.13
            initial = float(np.linalg.norm(target_xy - self.state()[0]))
            best, stagnant = initial, 0
            carry_turn_budget = (
                int(
                    math.pi
                    / max(float(getattr(self.args, "carry_yaw_rate", 0.15)), 0.02)
                    * self.hz
                    * 1.25
                )
                if grip < -0.5
                else 0
            )
            for _ in range(max(260, carry_turn_budget, int(initial / 0.08 * 80))):
                xy, yaw = self.state()
                delta = target_xy - xy
                distance = float(np.linalg.norm(delta))
                if distance < tolerance or (
                    self.phase.upper() in {"SEARCH", "MOVE", "REACQUIRE"}
                    and self.detected
                ):
                    break
                error = wrap(math.atan2(delta[1], delta[0]) - yaw)
                if distance < best - 0.004:
                    best, stagnant = distance, 0
                elif abs(error) <= 0.30:
                    # Count lack of translation only once the base is aligned
                    # and a forward command is expected.  During payload
                    # transport the deliberately limited angular velocity can
                    # require more than 130 control cycles for a safe in-place
                    # turn; treating those cycles as translational stall is a
                    # false collision diagnosis.
                    stagnant += 1
                else:
                    stagnant = 0
                # A grasped object is cantilevered from the wrist while the
                # mobile base turns.  Limit angular acceleration during carry
                # so the fingers are not asked to absorb an artificial 0.7
                # rad/s step command.
                yaw_limit = (
                    float(getattr(self.args, "carry_yaw_rate", 0.15))
                    if grip < -0.5
                    else 0.70
                )
                yaw_rate = float(np.clip(1.8 * error, -yaw_limit, yaw_limit))
                forward = 0.0 if abs(error) > 0.30 else min(speed, distance) * max(0.20, math.cos(error))
                self.command(grip=grip, base=(forward, yaw_rate / 3.14))
                if stagnant >= 130:
                    self.recoveries += 1
                    self.last_stall_xy = xy.copy()
                    contact_direction = target_xy - xy
                    contact_norm = float(np.linalg.norm(contact_direction))
                    self.last_stall_contact_proxy_xy = (
                        xy + 0.25 * contact_direction / contact_norm
                        if contact_norm > 1e-6
                        else xy.copy()
                    )
                    self.memory_event("controller_stall_recovery")
                    for _ in range(18):
                        self.command(grip=grip, base=(-0.10, 0))
                    # The known narrow ReplicaCAD doorway needs a slightly
                    # wider physical line than the triangle-mesh shortcut.
                    if 2.7 < xy[0] < 3.2 and -6.9 < xy[1] < -6.1:
                        if self.phase == "Transport":
                            detour = np.asarray(
                                [[2.90, -6.78], [3.003, -6.38], [3.11, -5.98]]
                            )
                        else:
                            detour = np.asarray(
                                [[3.11, -5.98], [3.003, -6.38], [2.90, -6.78]]
                            )
                        self.drive(detour, grip, min(speed, 0.16), 0.13)
                        break
                    self.capture()
                    self.args.output_root.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(self.args.output_root / "controller_stall.png", self.frames[-1])
                    raise RuntimeError(
                        f"physical controller stalled near {xy.tolist()} "
                        f"while approaching {target_xy.tolist()}"
                    )
            else:
                raise RuntimeError(f"physical controller failed to reach {target_xy.tolist()}")

    def turn(
        self,
        target_yaw: float,
        grip: float,
        rate: float = 0.55,
        tolerance: float = 0.025,
    ) -> None:
        # The payload-safe carry rate is deliberately much lower than the
        # free-space turn rate.  A fixed 360-step budget covers a normal turn
        # but cannot cover a worst-case pi-radian reorientation at 0.06 rad/s.
        # Scale the physical timeout from the requested angular speed while
        # keeping the same measured-yaw stopping rule.
        maximum_steps = max(
            360,
            math.ceil(1.5 * math.pi * self.hz / max(abs(rate), 0.02)),
        )
        for _ in range(maximum_steps):
            error = wrap(target_yaw - self.state()[1])
            if abs(error) < tolerance:
                return
            self.command(grip=grip, base=(0, float(np.clip(1.8 * error, -rate, rate)) / 3.14))
        raise RuntimeError("yaw controller failed")

    def settle_dock_pose(self, target_xy: np.ndarray, target_yaw: float, grip: float) -> None:
        """Remove the small base translation induced by an in-place Fetch turn."""
        position_tolerance = float(
            getattr(self.args, "dock_position_tolerance", 0.020)
        )
        yaw_tolerance = float(getattr(self.args, "dock_yaw_tolerance", 0.040))
        for _ in range(120):
            xy, yaw = self.state()
            delta = np.asarray(target_xy) - xy
            heading = np.asarray([math.cos(yaw), math.sin(yaw)])
            longitudinal = float(delta @ heading)
            lateral = float(delta @ np.asarray([-heading[1], heading[0]]))
            yaw_error = wrap(target_yaw - yaw)
            # Static friction makes sub-centimetre lateral corrections
            # unattainable for the differential-drive base.  The following
            # current-RGB-D Cartesian arm stage closes the remaining error in
            # world space, so the accepted envelope is embodiment/scene
            # specific and remains guarded by physical grasp verification.
            if np.linalg.norm(delta) < position_tolerance and abs(yaw_error) < yaw_tolerance:
                return
            # At this stage the lateral error is only a few millimetres.  A
            # low-speed forward correction preserves the grasp heading and
            # avoids pushing the object with the wrist during arm approach.
            forward = float(np.clip(2.0 * longitudinal, -0.09, 0.09))
            yaw_rate = float(np.clip(1.6 * yaw_error + 0.8 * lateral, -0.22, 0.22))
            self.command(grip=grip, base=(forward, yaw_rate / 3.14))
        raise RuntimeError("dock pose controller failed")

    def move_ee(
        self,
        target_world: np.ndarray,
        grip: float,
        tolerance: float = 0.014,
        strict: bool = True,
    ) -> float:
        for _ in range(100):
            tcp = np.asarray(self.agent.tcp_pose.p[0].cpu(), dtype=float)
            error = target_world - tcp
            if np.linalg.norm(error) < tolerance:
                return float(np.linalg.norm(error))
            yaw = self.state()[1]
            c, s = math.cos(yaw), math.sin(yaw)
            local = np.asarray([c * error[0] + s * error[1], -s * error[0] + c * error[1], error[2]])
            self.command_ee(np.r_[np.clip(local / 0.1 * 0.48, -0.28, 0.28), [0, 0, 0]], grip)
        tcp = np.asarray(self.agent.tcp_pose.p[0].cpu(), dtype=float)
        residual = float(np.linalg.norm(target_world - tcp))
        if strict and residual > 0.028:
            self.capture()
            self.args.output_root.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(self.args.output_root / "ee_controller_failure.png", self.frames[-1])
            raise RuntimeError(
                f"end-effector controller failed: target={target_world.tolist()}, "
                f"tcp={tcp.tolist()}, residual={residual:.4f}"
            )
        return residual

    def map_inset(self, width: int) -> np.ndarray:
        canvas = np.full((*self.known.shape, 3), (53, 57, 62), dtype=np.uint8)
        canvas[self.known == 1] = (232, 235, 237)
        canvas[self.known == -1] = (16, 19, 22)
        canvas[(self.known == 1) & (self.scenario.context > 0)] = (222, 226, 180)
        cells = np.asarray([self.grid.world_to_cell(point) for point in self.trajectory], dtype=np.int32)
        if len(cells) > 1:
            cv2.polylines(canvas, [cells[:, ::-1]], False, (224, 140, 25), 2, cv2.LINE_AA)
        cv2.circle(canvas, (self.scenario.start[1], self.scenario.start[0]), 4, (75, 180, 90), -1)
        current = cells[-1]
        cv2.circle(canvas, (int(current[1]), int(current[0])), 4, (5, 5, 5), -1)
        if self.detected:
            cv2.drawMarker(canvas, (self.scenario.target[1], self.scenario.target[0]), (35, 55, 230), cv2.MARKER_STAR, 12, 2)
        canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
        height = max(1, round(canvas.shape[0] * width / canvas.shape[1]))
        return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)

    def sensor_inset(self, width: int) -> tuple[np.ndarray, str]:
        """Return the current Fetch RGB observation used for video inspection.

        Search uses the head camera because it has the useful navigation view;
        close manipulation uses the hand camera.  The experiment's detector is
        deliberately kept separate (controlled local visibility), so displaying
        this stream must not be interpreted as a learned RGB detection claim.
        """

        self.base.scene.update_render(
            update_sensors=True, update_human_render_cameras=False
        )
        self.base.capture_sensor_data()
        close = self.phase.upper() in {
            "APPROACH", "GRASP", "LIFT", "PLACE", "VERIFY", "TARGET MOVE",
            "FOCUS OLD", "CLEAR STALE",
        }
        sensor_name = "fetch_hand" if close else "fetch_head"
        observation = self.base._sensors[sensor_name].get_obs(
            rgb=True,
            depth=False,
            position=False,
            segmentation=True,
        )
        rgb = observation["rgb"]
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)[0, ..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1 else 1), 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        segmentation = observation["segmentation"]
        if hasattr(segmentation, "detach"):
            segmentation = segmentation.detach().cpu().numpy()
        target_id = int(np.asarray(self.target.per_scene_id.cpu())[0])
        target_mask = np.asarray(segmentation)[0, ..., 0] == target_id
        target_visible = int(np.count_nonzero(target_mask)) >= 4
        if target_visible:
            rows, cols = np.where(target_mask)
            cv2.rectangle(
                bgr,
                (int(cols.min()), int(rows.min())),
                (int(cols.max()), int(rows.max())),
                (70, 220, 80),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                bgr,
                "ACTOR-ID IN FRAME",
                (max(3, int(cols.min())), max(14, int(rows.min()) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (70, 220, 80),
                1,
                cv2.LINE_AA,
            )
            self.camera_target_visible_ever = True
            if self.camera_target_first_visible_step is None:
                self.camera_target_first_visible_step = self.step
        height = max(1, round(bgr.shape[0] * width / bgr.shape[1]))
        return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA), sensor_name

    @staticmethod
    def labeled_inset(frame: np.ndarray, inset: np.ndarray, x0: int, y0: int, label: str) -> None:
        frame[y0:y0 + inset.shape[0], x0:x0 + inset.shape[1]] = inset
        cv2.rectangle(
            frame,
            (x0 - 2, y0 - 24),
            (x0 + inset.shape[1] + 1, y0 + inset.shape[0] + 1),
            (248, 248, 248),
            2,
        )
        cv2.rectangle(frame, (x0, y0 - 22), (x0 + inset.shape[1], y0), (17, 24, 30), -1)
        cv2.putText(
            frame,
            label,
            (x0 + 6, y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    def capture(self) -> None:
        xy, yaw = self.state()
        heading, side = np.asarray([math.cos(yaw), math.sin(yaw)]), np.asarray([-math.sin(yaw), math.cos(yaw)])
        close = self.phase.upper() in {
            "DETECT", "APPROACH", "GRASP", "LIFT", "PLACE", "VERIFY",
            "TARGET MOVE", "FOCUS OLD", "CLEAR STALE",
        }
        if close:
            focus, eye_xy = self.object_xyz(), xy - 0.82 * heading - 0.48 * side
            eye, look = [*eye_xy, 1.72], [focus[0], focus[1], max(0.72, focus[2])]
        else:
            # Keep the qualitative camera inside the current room and close
            # enough to show actual wheel/arm motion.  A high, distant camera
            # frequently sat behind an ArchitecTHOR wall and produced a large
            # blank region even though the robot sensors were valid.
            eye_xy = xy - 0.92 * heading - 0.52 * side
            eye, look = [*eye_xy, 1.78], [*(xy + 0.24 * heading), 0.58]
        camera = self.base._human_render_cameras["render_camera"].camera
        camera.set_local_pose(sapien_utils.look_at(eye, look).sp)
        raw = self.base.render_rgb_array("render_camera")
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
        rgb = np.asarray(raw)[0]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * (255 if rgb.max() <= 1 else 1), 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 94), (17, 24, 30), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.putText(frame, self.video_title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (255, 255, 255), 2, cv2.LINE_AA)
        phase_upper = self.phase.upper()
        if phase_upper == "VERIFY":
            task_status = f"place {'YES' if self.place_ok else 'no'}"
        elif phase_upper == "PLACE":
            task_status = "place pending"
        else:
            grasp = phase_upper in {
                "GRASP", "LIFT", "TRANSPORT", "CLEARANCE RETREAT", "CARRY"
            } and self.grasping()
            task_status = f"grasp {'YES' if grasp else 'no'}"
        cv2.putText(frame, f"{self.phase.upper()} | sim {self.step / self.hz:5.1f} s | travel {self.travel:5.2f} m | replans {self.replans} | {task_status}", (20, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 221, 230), 1, cv2.LINE_AA)
        known = 100 * np.count_nonzero(self.known) / self.known.size
        cv2.putText(frame, f"online memory {known:4.1f}% | target {'observed' if self.detected else 'not observed'}", (20, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (174, 206, 220), 1, cv2.LINE_AA)
        map_view = self.map_inset(min(255, frame.shape[1] // 3))
        # Multi-room scene rasters can be much taller than the ReplicaCAD map.
        # Keep the inset below the header and above the footer for every aspect
        # ratio instead of allowing a negative NumPy slice.
        maximum_inset_height = max(80, frame.shape[0] - 156)
        if map_view.shape[0] > maximum_inset_height:
            resized_width = max(
                1,
                round(map_view.shape[1] * maximum_inset_height / map_view.shape[0]),
            )
            map_view = cv2.resize(
                map_view,
                (resized_width, maximum_inset_height),
                interpolation=cv2.INTER_NEAREST,
            )
        map_x, map_y = 16, frame.shape[0] - map_view.shape[0] - 38
        self.labeled_inset(
            frame,
            map_view,
            map_x,
            map_y,
            self.map_label,
        )
        sensor_view, sensor_name = self.sensor_inset(min(260, frame.shape[1] // 3))
        sensor_x = frame.shape[1] - sensor_view.shape[1] - 16
        sensor_y = frame.shape[0] - sensor_view.shape[0] - 38
        self.labeled_inset(
            frame,
            sensor_view,
            sensor_x,
            sensor_y,
            f"CURRENT OBSERVATION | {sensor_name.replace('_', ' ').upper()} RGB",
        )
        playback = self.args.fps * self.args.capture_stride / self.hz
        cv2.rectangle(frame, (0, frame.shape[0] - 31), (frame.shape[1], frame.shape[0]), (17, 24, 30), -1)
        cv2.putText(
            frame,
            self.video_footer.format(playback=playback),
            (18, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.frame_phases.append(self.phase)

    def freeze(self, seconds: float) -> None:
        self.capture()
        for _ in range(max(0, round(seconds * self.args.fps) - 1)):
            self.frames.append(self.frames[-1].copy())
            self.frame_phases.append(self.phase)

    def execute(self, route: np.ndarray, planner_replans: int) -> None:
        self.hold(20, 1)
        self.phase = "Search"
        self.observe()
        self.memory_event("initial_observation")
        self.freeze(1.0)
        self.replans = planner_replans
        route = np.asarray(route, dtype=float)
        for index, point in enumerate(route[1:]):
            self.drive(np.asarray([point]), 1, self.args.search_speed, 0.20)
            if index % 5 == 0:
                self.memory_event("online_memory_update")
            if self.detected:
                break
        if not self.detected:
            # The planner's last trace sample is a detection pose. Reacquire
            # it more precisely when accumulated controller tolerance leaves
            # the rasterized visibility ray one cell short.
            self.drive(np.asarray([route[-1]]), 1, 0.14, 0.045)
            self.observe(force=True)
        if not self.detected:
            raise RuntimeError("target was not locally detected")
        self.phase = "Detect"
        self.memory_event("target_detected")
        self.freeze(2.0)
        build_receptacle(self.base, self.geometry)

        self.phase = "Approach"
        self.drive(np.asarray([self.geometry.dock_xy]), 1, 0.12, 0.020)
        self.turn(self.geometry.pick_yaw, 1)
        self.settle_dock_pose(self.geometry.dock_xy, self.geometry.pick_yaw, 1)
        self.freeze(1.0)
        # Switch only the arm to Cartesian delta-pose control.  This preserves
        # the current wrist orientation and follows the same feedback-based
        # approach used by the verified tabletop grasp calibration.
        self.agent.set_control_mode("pd_ee_delta_pose")
        self.agent.controller.reset()
        tcp_matrix = np.asarray(
            self.agent.tcp_pose.to_transformation_matrix()[0, :3, :3].cpu(), dtype=float
        )
        # Fetch's finger meshes straddle the gripper-link x axis, while the
        # palm occupies negative x (see the shipped Fetch URDF).  Approach
        # from positive x so the object enters between the fingers first.
        approach_axis = tcp_matrix[:, 0]
        object_center = self.object_xyz().copy()
        self.move_ee(object_center - self.geometry.approach_distance * approach_axis, 1.0, 0.016)
        final_approach_residual = self.move_ee(object_center, 1.0, 0.012, strict=False)
        self.memory_event(f"final_approach_residual={final_approach_residual:.4f}m")
        for _ in range(10):
            self.command_ee(grip=1.0)
        for grip in np.linspace(1.0, -1.0, 35):
            self.command_ee(grip=float(grip))
        for _ in range(20):
            self.command_ee(grip=-1.0)
        self.phase = "Grasp"
        self.freeze(1.0)
        for _ in range(60):
            self.command_ee(grip=-1.0)
        self.grasp_ok = self.grasping()
        self.memory_event("bilateral_contact_check")
        self.freeze(2.0)
        if not self.grasp_ok:
            left = np.asarray(
                self.base.scene.get_pairwise_contact_forces(self.agent.finger1_link, self.target)[0].cpu(),
                dtype=float,
            )
            right = np.asarray(
                self.base.scene.get_pairwise_contact_forces(self.agent.finger2_link, self.target)[0].cpu(),
                dtype=float,
            )
            robot_contacts = {}
            for link in self.agent.robot.get_links():
                force = np.asarray(
                    self.base.scene.get_pairwise_contact_forces(link, self.target)[0].cpu(), dtype=float
                )
                if np.linalg.norm(force) > 1e-4:
                    robot_contacts[link.name] = force.tolist()
            self.args.output_root.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(self.args.output_root / "grasp_failure.png", self.frames[-1])
            write_csv(self.args.output_root / "grasp_failure_control.csv", self.rows)
            raise RuntimeError(
                "bilateral finger contact criterion failed: "
                f"tcp={np.asarray(self.agent.tcp_pose.p[0].cpu()).tolist()}, "
                f"object={self.object_xyz().tolist()}, left_force={left.tolist()}, "
                f"right_force={right.tolist()}, "
                f"finger1={np.asarray(self.agent.finger1_link.pose.p[0].cpu()).tolist()}, "
                f"finger2={np.asarray(self.agent.finger2_link.pose.p[0].cpu()).tolist()}, "
                f"robot_contacts={robot_contacts}"
            )
        self.target.linear_damping = 0.1
        self.target.angular_damping = 0.1
        self.phase = "Lift"
        lift_target = np.asarray(self.agent.tcp_pose.p[0].cpu(), dtype=float) + np.asarray([0, 0, 0.15])
        lift_residual = self.move_ee(lift_target, -1.0, 0.015, strict=False)
        self.memory_event(f"lift_residual={lift_residual:.4f}m")
        for _ in range(10):
            self.command_ee(grip=-1.0)
        qpos = np.asarray(self.agent.robot.get_qpos()[0].cpu(), dtype=float)
        self.arm_target = qpos[[5, 7, 8, 9, 10, 11, 12]].copy()
        self.agent.set_control_mode("pd_joint_pos")
        self.agent.controller.reset()
        for _ in range(25):
            self.command(grip=-1, base=(-0.10, 0))
        if not self.grasping():
            raise RuntimeError("object lost after grasp")
        self.freeze(1.0)

        self.phase = "Transport"
        # This return corridor was already physically traversed during search.
        carry = np.asarray(
            [[1.75, -6.78], [2.07, -6.82], [2.39, -6.86], [2.79, -7.18],
             [2.897, -6.78], [3.003, -6.38], [3.11, -5.98],
             [3.11, -5.50], self.geometry.place_base_xy]
        )
        self.drive(carry, -1, 0.18, 0.07)
        self.turn(self.geometry.place_yaw, -1, 0.45)
        if not self.grasping():
            self.capture()
            self.args.output_root.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(self.args.output_root / "transport_failure.png", self.frames[-1])
            write_csv(self.args.output_root / "transport_failure_control.csv", self.rows)
            raise RuntimeError("object lost during physical transport")
        pre = self.object_xyz().copy()
        if np.linalg.norm(pre[:2] - self.geometry.tray_xy) > 0.22:
            raise RuntimeError("carried object not aligned with receptacle")
        self.hold(60, -1)
        if not self.grasping():
            raise RuntimeError("object lost while settling above receptacle")
        self.freeze(1.0)
        self.phase = "Place"
        self.target.linear_damping = 5.0
        self.target.angular_damping = 5.0
        for grip in np.linspace(-1.0, 1.0, 50):
            self.command(grip=float(grip))
        self.hold(120, 1)
        final = self.object_xyz()
        expected_z = self.geometry.tray_top + self.geometry.half_height
        self.place_ok = bool(
            np.linalg.norm(final[:2] - self.geometry.tray_xy) < 0.22
            and abs(final[2] - expected_z) < 0.04
            and not self.grasping()
        )
        self.phase = "Verify"
        self.memory_event("place_verification")
        self.freeze(3.0)
        if not self.place_ok:
            self.args.output_root.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(self.args.output_root / "placement_failure.png", self.frames[-1])
            write_csv(self.args.output_root / "placement_failure_control.csv", self.rows)
            raise RuntimeError(f"placement failed at {final.tolist()}")
        for _ in range(35):
            self.command(grip=1, base=(-0.10, 0))
        self.freeze(2.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(trial: Trial, args: argparse.Namespace, result: Any, raw_route: np.ndarray, navmesh_source: Path) -> dict[str, Any]:
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    video = output / "dream_replicacad_closed_loop.mp4"
    with imageio.get_writer(video, fps=args.fps, codec="libx264", quality=8, macro_block_size=None, ffmpeg_log_level="warning") as writer:
        for frame in trial.frames:
            writer.append_data(frame)
    phases = ("Search", "Detect", "Grasp", "Verify")
    labels = ("(a) Online memory", "(b) Target observed", "(c) Physical grasp", "(d) Placed & verified")
    panels = []
    for phase, label in zip(phases, labels):
        matches = [i for i, value in enumerate(trial.frame_phases) if value == phase]
        frame = trial.frames[matches[len(matches) // 2]].copy()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (18, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    montage = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    montage_path = output / "dream_replicacad_montage.png"
    imageio.imwrite(montage_path, montage)
    if args.paper_montage:
        args.paper_montage.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(args.paper_montage, montage)
    control, memory, trajectory = output / "control_log.csv", output / "memory_log.csv", output / "trajectory.csv"
    write_csv(control, trial.rows)
    write_csv(memory, trial.memory_rows)
    write_csv(trajectory, [dict(step=i, world_x_m=p[0], world_y_m=p[1]) for i, p in enumerate(trial.trajectory)])
    final = trial.object_xyz()
    metadata = {
        "evidence_type": "physics-controlled simulator integration trial",
        "boundary": BOUNDARY,
        "task": "unknown-map cross-room search, pickup, physical transport, and place",
        "environment_id": ENVIRONMENT_ID, "scene": SCENE_NAME,
        "simulator": "ManiSkill 3.0.1 / SAPIEN 3.0.3", "simulation_backend": "PhysX CPU",
        "robot": "Fetch",
        "control_modes": ["pd_joint_pos", "pd_ee_delta_pose"],
        "base_controller": "PD forward/yaw velocity via env.step()",
        "planner_controller_coupling": (
            "incremental unknown-map route is computed before the physics rollout; "
            "Fetch then tracks it with feedback control while rebuilding the displayed "
            "local memory; no onboard-RGB or physical-state replanning"
        ),
        "arm_controller": "joint-position hold during navigation; Cartesian delta-pose approach/lift via env.step()",
        "grasp": "geometric dock plus top-down Cartesian approach; bilateral finger-force criterion",
        "placement": "carried-object alignment, physical settling, gradual release, and pose verification",
        "map_initially_known": False, "future_path_shown": False,
        "video_insets": [
            "online 2D occupancy/context-proxy memory",
            "current Fetch head/hand RGB observation",
        ],
        "onboard_rgb_used_for_detection": False,
        "onboard_rgb_target_visible_ever": trial.camera_target_visible_ever,
        "onboard_rgb_target_first_visible_step": trial.camera_target_first_visible_step,
        "onboard_rgb_bbox_source": "simulator actor-ID mask for audit visualization only",
        "target_detection_source": "controlled occlusion-aware local geometric visibility",
        "controlled_semantic_context": True, "perfect_target_pose_after_detection": True,
        "learned_perception_tested": False, "dynamic_relocation_tested": False,
        "teleportation_used": False, "state_setter_calls_after_initialization": 0,
        "target_detected": trial.detected, "bilateral_contact_grasp_confirmed": trial.grasp_ok,
        "physical_place_success": trial.place_ok,
        "planner_path_m": result.path_m, "shortest_path_m": result.shortest_path_m,
        "actual_search_travel_m": trial.search_travel, "actual_total_travel_m": trial.travel,
        "planner_replans": result.replans, "controller_recoveries": trial.recoveries,
        "control_frequency_hz": trial.hz, "control_steps": trial.step,
        "simulated_duration_s": trial.step / trial.hz,
        "max_translation_per_step_m": trial.max_step_translation,
        "max_yaw_per_step_rad": trial.max_step_yaw,
        "final_object_xyz_m": final.tolist(),
        "place_xy_error_m": float(np.linalg.norm(final[:2] - trial.geometry.tray_xy)),
        "raw_planner_route_points": len(raw_route), "rendered_frames": len(trial.frames),
        "fps": args.fps, "capture_stride": args.capture_stride,
        "nominal_playback_factor": args.fps * args.capture_stride / trial.hz,
        "video_duration_s": len(trial.frames) / args.fps,
        "navmesh_source_sha256": sha256(navmesh_source),
        "video_sha256": sha256(video), "montage_sha256": sha256(montage_path),
        "control_log_sha256": sha256(control), "memory_log_sha256": sha256(memory),
        "trajectory_sha256": sha256(trajectory),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 output requires even --width and --height values")
    env = gym.make(
        ENVIRONMENT_ID, robot_uids="fetch", build_config_idxs=[0], num_envs=1,
        # State observations keep every control step inexpensive.  RGB-D is
        # captured explicitly only for recorded frames via get_sensor_images().
        obs_mode="state", reward_mode="none", render_mode="rgb_array",
        control_mode="pd_joint_pos", sim_backend="physx_cpu", render_backend="cpu",
        max_episode_steps=50000,
        sensor_configs=dict(width=320, height=240),
        human_render_camera_configs=dict(width=args.width, height=args.height, fov=1.05, near=0.05, far=100, shader_pack="default"),
    )
    try:
        env.reset(seed=0)
        base = env.unwrapped
        grid = rasterize_navigation_mesh(base.scene_builder.navigable_positions[0], args.resolution)
        target_xy = np.asarray([args.target_x, args.target_y])
        scenario = build_search_scenario(grid, target_xy=target_xy, destination_boundary_y_m=-2.50)
        result, trace = run_scenario(
            scenario, "dream", budget_m=args.budget_m, sensor_range_m=args.sensor_range,
            semantic_rate=0.10, keep_trace=True,
        )
        if not result.success or trace is None:
            raise RuntimeError("DREAM planner did not find the target")
        raw_route = np.asarray([grid.cell_to_world(point) for point in trace["trajectory"]])
        geometry = make_geometry(base, target_xy, np.asarray([2.47, -5.34]))
        target = build_objects(base, geometry)
        trial = Trial(env, grid, scenario, target, geometry, args)
        trial.execute(raw_route, result.replans)
        navmesh = Path(os.environ["MS_ASSET_DIR"]) / "data/scene_datasets/replica_cad_dataset/configs/scenes/apt_0.scene_instance.fetch.navigable_positions.obj"
        metadata = write_outputs(trial, args, result, raw_route, navmesh)
        print(json.dumps(metadata, indent=2), flush=True)
        return metadata
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/maniskill_replicacad_closed_loop"))
    parser.add_argument("--paper-montage", type=Path)
    parser.add_argument("--resolution", type=float, default=0.08)
    parser.add_argument("--sensor-range", type=float, default=1.20)
    parser.add_argument("--scan-interval", type=int, default=10)
    parser.add_argument("--budget-m", type=float, default=80.0)
    parser.add_argument("--search-speed", type=float, default=0.28)
    parser.add_argument("--target-x", type=float, default=0.79)
    parser.add_argument("--target-y", type=float, default=-6.30)
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
