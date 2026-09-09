#!/usr/bin/env python3
"""Run the sensor-driven DREAM dynamic trial in one ArchitecTHOR home."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if not os.environ.get("MS_ASSET_DIR"):
    os.environ["MS_ASSET_DIR"] = os.environ.get(
        "DREAM_MANISKILL_ASSET_DIR", str(ROOT / ".maniskill_assets")
    )
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")

import gymnasium as gym
import numpy as np
from scipy.ndimage import distance_transform_edt

import mani_skill  # noqa: F401

from architecthor_navigation import (
    FETCH_START_XY,
    SCENE_IDS,
    choose_episode_layout,
    derive_navigation_grid,
    _free_at,
    _free_segment,
)
from dream_simulation_core import SimulationSpec, sha256
from houseexpo_cross_room import weighted_distances
from maniskill_dynamic_dream import (
    DynamicTrial,
    DYNAMIC_TARGET_HALF_HEIGHT_M,
    DYNAMIC_TARGET_HALF_SIZE_M,
    PHYSICAL_OFFSET,
    _array,
    _write_csv,
    add_common_arguments,
    build_moving_target,
    write_outputs,
)
from maniskill_replicacad_closed_loop import make_geometry


ENVIRONMENT_ID = "SceneManipulation-v1"
BOUNDARY = (
    "Controlled sensor-driven dynamic trial in an official ArchitecTHOR scene. "
    "Target localization and semantic memory use only current rendered actor-ID "
    "pixels, RGB-D and camera calibration; simulator target pose is reserved for "
    "scoring. Traversability is conservatively derived from the exact stage and "
    "object GLBs because the official archive contains no Fetch navigation file. "
    "This isolates DREAM memory/planning/control/manipulation from learned "
    "open-vocabulary perception."
)


class ArchitecTHORTrial(DynamicTrial):
    experiment_boundary = BOUNDARY
    traversability_source = (
        "floor triangles minus projected wall/furniture geometry from official GLBs"
    )

    def move_target_continuously(self, destination_logical_xy: np.ndarray) -> np.ndarray:
        """Select a collision-clear event direction away from the observing base."""

        support_xy = _array(self.support.pose.p)[0, :2].astype(float)
        base_xy, _ = self.state()
        preferred = np.asarray(destination_logical_xy, dtype=float) + PHYSICAL_OFFSET
        preferred_angle = math.atan2(*(preferred - support_xy)[::-1])

        default_pick_yaw = math.atan2(0.48, -0.96)
        # Prefer the nominal real-system-style approach, while allowing seven
        # rotated base headings in layouts where furniture blocks that side.
        # The hand RGB-D refinement below the planner recomputes the Cartesian
        # object centre at every heading and the physical bilateral-contact
        # check remains the success gate.
        pick_yaws = tuple(default_pick_yaw + index * math.pi / 4 for index in range(8))
        angles = np.linspace(0, 2 * math.pi, 64, endpoint=False)
        angles = sorted(
            angles,
            key=lambda angle: abs(math.atan2(math.sin(angle - preferred_angle), math.cos(angle - preferred_angle))),
        )
        base_cell = self.grid.world_to_cell(base_xy)
        connected_distances, _ = weighted_distances(self.grid.free, base_cell)
        clearance_m = distance_transform_edt(self.grid.free) * self.grid.resolution_m
        candidates = []
        selected_relocation_distance = self.args.relocation_distance
        # Some furnished homes place the farthest valid search point close to
        # a wall.  Preserve the requested 0.85 m event whenever feasible, then
        # deterministically shorten it (still above DREAM's 0.30 m relocation
        # threshold) instead of rejecting an otherwise useful dynamic trial.
        for relocation_distance in (
            self.args.relocation_distance,
            0.75,
            0.65,
            0.55,
            0.45,
        ):
            for pick_yaw in pick_yaws:
                rotation = np.asarray(
                    [
                        [math.cos(pick_yaw), -math.sin(pick_yaw)],
                        [math.sin(pick_yaw), math.cos(pick_yaw)],
                    ]
                )
                dock_offset = -(rotation @ np.asarray([0.550, 0.038]))
                yaw_difference = abs(
                    math.atan2(
                        math.sin(pick_yaw - default_pick_yaw),
                        math.cos(pick_yaw - default_pick_yaw),
                    )
                )
                for angle in angles:
                    direction = np.asarray([math.cos(angle), math.sin(angle)])
                    final = support_xy + relocation_distance * direction
                    initial_robot_clearance = float(np.linalg.norm(support_xy - base_xy))
                    final_robot_clearance = float(np.linalg.norm(final - base_xy))
                    moves_away = float((support_xy - base_xy) @ direction) > 0.0
                    robot_is_near = initial_robot_clearance <= 1.20
                    safe_relative_motion = (
                        moves_away
                        and final_robot_clearance >= initial_robot_clearance + 0.25
                        if robot_is_near
                        else final_robot_clearance >= initial_robot_clearance - 0.10
                    )
                    if (
                        not safe_relative_motion
                        or final_robot_clearance < self.args.dynamic_robot_clearance
                    ):
                        continue
                    dock = final + dock_offset
                    retreat = dock - self.args.post_grasp_retreat * np.asarray(
                        [math.cos(pick_yaw), math.sin(pick_yaw)]
                    )
                    dock_cell = self.grid.world_to_cell(dock)
                    retreat_cell = self.grid.world_to_cell(retreat)
                    final_cell = self.grid.world_to_cell(final)
                    if (
                        _free_at(self.grid, final)
                        and _free_segment(self.grid, support_xy, final)
                        and _free_at(self.grid, dock)
                        and _free_at(self.grid, retreat)
                        and _free_segment(self.grid, dock, retreat)
                        and np.isfinite(connected_distances[dock_cell])
                    ):
                        minimum_static_clearance = float(
                            min(
                                clearance_m[final_cell],
                                clearance_m[dock_cell],
                                clearance_m[retreat_cell],
                            )
                        )
                        summed_static_clearance = float(
                            clearance_m[final_cell]
                            + clearance_m[dock_cell]
                            + clearance_m[retreat_cell]
                        )
                        candidates.append(
                            (
                                minimum_static_clearance,
                                summed_static_clearance,
                                -yaw_difference,
                                final,
                                initial_robot_clearance,
                                final_robot_clearance,
                                moves_away,
                                float(pick_yaw),
                            )
                        )
            if candidates:
                selected_relocation_distance = relocation_distance
                break
        if not candidates:
            raise RuntimeError("no collision-clear relocation and RGB-D-refined dock pair")
        ranked_candidates = sorted(
            candidates,
            key=lambda item: (item[0], item[1], item[2], -item[5]),
            reverse=True,
        )
        distinct_candidates = []
        for candidate in ranked_candidates:
            if all(
                np.linalg.norm(candidate[3] - prior[3])
                >= self.args.minimum_relocation_layout_separation
                for prior in distinct_candidates
            ):
                distinct_candidates.append(candidate)
        if self.args.relocation_layout_index >= len(distinct_candidates):
            raise RuntimeError(
                "requested relocation layout index is unavailable: "
                f"{self.args.relocation_layout_index} >= {len(distinct_candidates)}"
            )
        (
            chosen_static_clearance,
            chosen_summed_static_clearance,
            _,
            chosen,
            chosen_initial_robot_clearance,
            chosen_final_robot_clearance,
            chosen_moves_away,
            chosen_pick_yaw,
        ) = distinct_candidates[self.args.relocation_layout_index]
        self.selected_pick_yaw = chosen_pick_yaw
        chosen_logical = chosen - PHYSICAL_OFFSET
        self.event(
            "dynamic_destination_safety_selection",
            requested_destination_xy_m=preferred.tolist(),
            selected_destination_xy_m=chosen.tolist(),
            initial_robot_clearance_m=chosen_initial_robot_clearance,
            final_robot_clearance_m=chosen_final_robot_clearance,
            minimum_static_grid_clearance_m=chosen_static_clearance,
            summed_static_grid_clearance_m=chosen_summed_static_clearance,
            candidate_count=len(candidates),
            distinct_relocation_candidate_count=len(distinct_candidates),
            relocation_layout_index=self.args.relocation_layout_index,
            motion_direction_is_away=chosen_moves_away,
            selected_pick_yaw_rad=chosen_pick_yaw,
            selected_relocation_distance_m=selected_relocation_distance,
        )
        self.extra_manifest_fields["episode_layout"]["runtime_selected_final_target_xy_m"] = (
            chosen_logical.tolist()
        )
        return super().move_target_continuously(chosen_logical)


def _failure(trial: ArchitecTHORTrial, args: argparse.Namespace, error: Exception) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "failure_control.csv", trial.rows)
    _write_csv(args.output_root / "failure_events.csv", trial.events)
    _write_csv(args.output_root / "failure_memory.csv", trial.memory_updates)
    base_xy, base_yaw = trial.state()
    payload = {
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
        "execution_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in sorted(vars(args).items())
        },
        "implementation_sha256": {
            path.name: sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("maniskill_dynamic_dream.py"),
                Path(__file__).with_name("maniskill_replicacad_closed_loop.py"),
                Path(__file__).with_name("dream_simulation_core.py"),
                Path(__file__).resolve().parents[1] / "src/dream/dynamic_memory.py",
            )
        },
    }
    payload.update(trial.extra_manifest_fields)
    (args.output_root / "failure.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    if trial.frames:
        import imageio.v2 as imageio

        imageio.imwrite(args.output_root / "failure_last_frame.png", trial.frames[-1])
    print(json.dumps(payload, indent=2), flush=True)


def run(args: argparse.Namespace) -> dict:
    try:
        build_config_idx = SCENE_IDS.index(args.scene_id)
    except ValueError as error:
        raise ValueError(f"unknown ArchitecTHOR scene: {args.scene_id}") from error
    asset_root = args.asset_root
    grid, grid_report = derive_navigation_grid(
        asset_root,
        args.scene_id,
        build_config_idx,
        resolution_m=args.resolution,
        robot_clearance_m=args.robot_clearance,
    )
    layout = choose_episode_layout(
        grid,
        build_config_idx,
        relocation_distance_m=args.relocation_distance,
        layout_index=args.layout_index,
        minimum_layout_separation_m=args.minimum_layout_separation,
        place_near_target=args.place_near_target,
    )
    if (args.place_base_override_x is None) != (args.place_base_override_y is None):
        raise ValueError("both placement-base override coordinates are required")
    if args.place_base_override_x is not None:
        override = np.asarray(
            [args.place_base_override_x, args.place_base_override_y], dtype=float
        )
        # The green receptacle is centred 0.696 m behind the pi-facing Fetch
        # placement pose.  Overrides are accepted only when both robot and
        # receptacle locations pass the same conservative derived grid.
        tray = override + np.asarray([-0.696, 0.009])
        if not _free_at(grid, override) or not _free_at(grid, tray):
            raise ValueError("placement override or its receptacle is outside the free grid")
        layout = replace(layout, place_base_xy=override)
    spec = SimulationSpec(
        "architecthor",
        args.scene_id,
        args.seed,
        args.protocol,
        args.variant,
        args.sensor_range,
        args.resolution,
    )
    env = gym.make(
        ENVIRONMENT_ID,
        scene_builder_cls="ArchitecTHOR",
        robot_uids="fetch",
        build_config_idxs=[build_config_idx],
        num_envs=1,
        obs_mode="state",
        reward_mode="none",
        render_mode="rgb_array",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        render_backend="cpu",
        max_episode_steps=40000,
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
    trial: ArchitecTHORTrial | None = None
    try:
        env.reset(seed=args.seed)
        base = env.unwrapped
        geometry = replace(
            make_geometry(base, layout.initial_target_xy, layout.place_base_xy),
            half_size=DYNAMIC_TARGET_HALF_SIZE_M,
            half_height=DYNAMIC_TARGET_HALF_HEIGHT_M,
        )
        support, target = build_moving_target(
            base, layout.initial_target_xy, geometry.target_z
        )
        trial = ArchitecTHORTrial(
            env,
            grid,
            layout.search_scenario,
            target,
            support,
            geometry,
            args,
            spec,
        )
        trial.video_title = (
            f"DREAM dynamic reacquisition + grasp | {args.scene_id} / Fetch"
        )
        trial.extra_manifest_fields = {
            "architecthor_adapter_sha256": sha256(Path(__file__)),
            "architecthor_navigation_sha256": sha256(
                Path(__file__).with_name("architecthor_navigation.py")
            ),
            "architecthor_build_config_idx": build_config_idx,
            "official_scene_id": args.scene_id,
            "official_fetch_start_xy_m": list(FETCH_START_XY[build_config_idx]),
            "derived_grid": asdict(grid_report),
            "episode_layout": {
                "layout_index": args.layout_index,
                "minimum_layout_separation_m": args.minimum_layout_separation,
                "relocation_layout_index": args.relocation_layout_index,
                "minimum_relocation_layout_separation_m": (
                    args.minimum_relocation_layout_separation
                ),
                "place_near_target": args.place_near_target,
                "initial_target_xy_m": layout.initial_target_xy.tolist(),
                "final_target_xy_m": layout.final_target_xy.tolist(),
                "place_base_xy_m": layout.place_base_xy.tolist(),
                "place_base_is_validated_override": bool(
                    args.place_base_override_x is not None
                ),
                "relocation_distance_m": layout.relocation_distance_m,
                "shortest_path_to_initial_target_m": layout.shortest_path_to_target_m,
            },
        }
        trial.execute_dynamic(layout.final_target_xy, layout.place_base_xy)
        return write_outputs(trial, args)
    except Exception as error:
        if trial is not None:
            _failure(trial, args, error)
        raise
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = add_common_arguments(argparse.ArgumentParser())
    parser.set_defaults(
        output_root=Path("experiments/results/maniskill_architecthor_dynamic"),
        max_replans=140,
        # ArchitecTHOR furniture is denser than the ReplicaCAD smoke scene.
        # Retreat far enough that a held payload can rotate without sweeping
        # through the force-driven support; the candidate selector validates
        # this entire segment before the dynamic event is accepted.
        # A controller-driven 0.40 m retreat gives the payload clearance before
        # turning while remaining feasible in the validated furnished scene.
        post_grasp_retreat=0.40,
        # The artificial moving support has a 0.24 m tabletop.  Excluding a
        # 0.34 m disk accounts for its footprint plus base tracking error when
        # the global approach route is needed.
        grid_target_exclusion=0.34,
        lift_height=0.06,
        lift_stages=1,
        minimum_measured_lift=0.03,
        retreat_speed=0.03,
        retreat_max_steps=600,
        carry_speed=0.04,
        dock_position_tolerance=0.055,
        # Several official homes contain a single narrow doorway in the
        # already 0.30 m-inflated Fetch grid.  Keep one additional raster cell
        # for payload transport and rely on the longer pre-turn retreat for
        # support clearance.
        carry_additional_clearance=0.08,
        cartesian_carry_hold=False,
    )
    parser.add_argument("--scene-id", choices=SCENE_IDS, default=SCENE_IDS[0])
    parser.add_argument("--layout-index", type=int, default=0)
    parser.add_argument("--minimum-layout-separation", type=float, default=1.0)
    parser.add_argument("--relocation-layout-index", type=int, default=0)
    parser.add_argument(
        "--minimum-relocation-layout-separation", type=float, default=0.45
    )
    parser.add_argument(
        "--place-near-target", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--place-base-override-x", type=float, default=None)
    parser.add_argument("--place-base-override-y", type=float, default=None)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path("../.maniskill_assets/data/scene_datasets/ai2thor"),
    )
    parser.add_argument("--robot-clearance", type=float, default=0.30)
    parser.add_argument("--relocation-distance", type=float, default=0.85)
    parser.add_argument("--dynamic-robot-clearance", type=float, default=0.70)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
