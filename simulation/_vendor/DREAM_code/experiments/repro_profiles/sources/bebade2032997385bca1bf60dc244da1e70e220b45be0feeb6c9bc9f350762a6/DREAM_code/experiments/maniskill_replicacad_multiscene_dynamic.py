#!/usr/bin/env python3
"""Run the physical DREAM dynamic pipeline in any ReplicaCAD apartment.

Unlike the original apartment-0 smoke test, this adapter derives a distant
cross-region target, a force-driven relocation, and a placement goal from the
official Fetch navigation mesh of each requested apartment.  The robot and
both dynamic actors are controlled only through ``env.step``/PhysX after
initialization.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

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
import mani_skill  # noqa: F401

from architecthor_navigation import (
    EpisodeLayout,
    _free_at,
    _free_segment,
    choose_episode_layout,
)
from dream_simulation_core import SimulationSpec, sha256
from maniskill_architecthor_dynamic import ArchitecTHORTrial, _failure
from maniskill_dynamic_dream import (
    DYNAMIC_TARGET_HALF_HEIGHT_M,
    DYNAMIC_TARGET_HALF_SIZE_M,
    PHYSICAL_OFFSET,
    add_common_arguments,
    build_moving_target,
    write_outputs,
)
from maniskill_replicacad_closed_loop import ENVIRONMENT_ID, make_geometry
from maniskill_replicacad_search_video import build_search_scenario, rasterize_navigation_mesh


SCENE_IDS = tuple(f"apt_{index}" for index in range(6))
BOUNDARY = (
    "Controlled sensor-driven dynamic trial in an official furnished ReplicaCAD "
    "apartment. Target localization and semantic memory use only current rendered "
    "actor-ID pixels, RGB-D and camera calibration; simulator target pose is reserved "
    "for scoring. Traversability uses the official Fetch navigation mesh. This "
    "isolates DREAM memory/planning/control/manipulation from learned open-vocabulary "
    "perception."
)


class ReplicaCADMultiSceneTrial(ArchitecTHORTrial):
    """Reuse the collision-aware physical policy on an official ReplicaCAD mesh."""

    experiment_boundary = BOUNDARY
    traversability_source = "official ReplicaCAD Fetch navigable-position mesh"


def run(args: argparse.Namespace) -> dict:
    scene_id = f"apt_{args.build_config_index}"
    spec = SimulationSpec(
        "replicacad", scene_id, args.seed, args.protocol, args.variant,
        args.sensor_range, args.resolution,
    )
    env = gym.make(
        ENVIRONMENT_ID,
        robot_uids="fetch",
        build_config_idxs=[args.build_config_index],
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
    trial: ReplicaCADMultiSceneTrial | None = None
    try:
        env.reset(seed=args.seed)
        base = env.unwrapped
        navmesh = base.scene_builder.navigable_positions[0]
        grid = rasterize_navigation_mesh(navmesh, args.resolution)

        # ``choose_episode_layout(..., 0)`` uses ReplicaCAD's common Fetch
        # spawn (-1, 0).  Replace only descriptive fields in the generated
        # scenario; its metric grid, context, and target remain unchanged.
        if args.layout_mode in {"legacy", "custom"}:
            initial_xy = np.asarray(
                [args.initial_target_x, args.initial_target_y], dtype=float
            )
            initial_physical_xy = initial_xy + PHYSICAL_OFFSET
            spawn_xy = np.asarray([-1.0, 0.0], dtype=float)
            place_xy = np.asarray(
                [args.place_base_x, args.place_base_y], dtype=float
            )
            tray_xy = place_xy + np.asarray([-0.696, 0.009])
            if not _free_at(grid, initial_physical_xy):
                raise ValueError("custom target is outside the official Fetch navmesh")
            if _free_segment(grid, spawn_xy, initial_physical_xy):
                raise ValueError("custom target does not require a cross-region turn")
            if not _free_at(grid, tray_xy):
                raise ValueError("custom placement tray is outside the official Fetch navmesh")
            scenario = build_search_scenario(
                grid, target_xy=initial_xy, destination_boundary_y_m=-2.50
            )
            if scenario.shortest_cells * scenario.resolution_m < 3.0:
                raise ValueError("custom target path is shorter than 3 m")
            nominal_final_xy = np.asarray(
                [args.final_target_x, args.final_target_y], dtype=float
            )
            layout = EpisodeLayout(
                initial_target_xy=initial_xy,
                final_target_xy=nominal_final_xy,
                place_base_xy=place_xy,
                search_scenario=scenario,
                relocation_distance_m=float(
                    np.linalg.norm(nominal_final_xy - initial_xy)
                ),
                shortest_path_to_target_m=(
                    scenario.shortest_cells * scenario.resolution_m
                ),
            )
        else:
            layout = choose_episode_layout(
                grid,
                0,
                relocation_distance_m=args.relocation_distance,
                layout_index=args.layout_index,
                minimum_layout_separation_m=args.minimum_layout_separation,
                place_near_target=args.place_near_target,
            )
        scenario = replace(
            layout.search_scenario,
            scale="ReplicaCAD",
            house_id=scene_id,
        )
        geometry = replace(
            make_geometry(base, layout.initial_target_xy, layout.place_base_xy),
            half_size=DYNAMIC_TARGET_HALF_SIZE_M,
            half_height=DYNAMIC_TARGET_HALF_HEIGHT_M,
        )
        support, target = build_moving_target(
            base, layout.initial_target_xy, geometry.target_z
        )
        trial = ReplicaCADMultiSceneTrial(
            env, grid, scenario, target, support, geometry, args, spec
        )
        trial.video_title = (
            f"DREAM dynamic reacquisition + grasp | ReplicaCAD {scene_id} / Fetch"
        )
        scene_config = (
            Path(os.environ["MS_ASSET_DIR"])
            / "data/scene_datasets/replica_cad_dataset/configs/scenes"
            / f"{scene_id}.scene_instance.json"
        )
        navmesh_path = scene_config.with_name(
            f"{scene_id}.scene_instance.fetch.navigable_positions.obj"
        )
        trial.extra_manifest_fields = {
            "replicacad_multiscene_adapter_sha256": sha256(Path(__file__)),
            "layout_selector_sha256": sha256(
                Path(__file__).with_name("architecthor_navigation.py")
            ),
            "replicacad_build_config_index": args.build_config_index,
            "layout_index": args.layout_index,
            "minimum_layout_separation_m": args.minimum_layout_separation,
            "relocation_layout_index": args.relocation_layout_index,
            "minimum_relocation_layout_separation_m": (
                args.minimum_relocation_layout_separation
            ),
            "layout_mode": args.layout_mode,
            "place_near_target": args.place_near_target,
            "official_scene_id": scene_id,
            "official_scene_config_sha256": sha256(scene_config),
            "official_fetch_navmesh_sha256": sha256(navmesh_path),
            "episode_layout": {
                "initial_target_xy_m": layout.initial_target_xy.tolist(),
                "nominal_final_target_xy_m": layout.final_target_xy.tolist(),
                "place_base_xy_m": layout.place_base_xy.tolist(),
                "requested_relocation_distance_m": layout.relocation_distance_m,
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
    parser.add_argument("--build-config-index", type=int, choices=range(6), required=True)
    parser.add_argument("--layout-index", type=int, default=0)
    parser.add_argument("--minimum-layout-separation", type=float, default=1.0)
    parser.add_argument("--relocation-layout-index", type=int, default=0)
    parser.add_argument(
        "--minimum-relocation-layout-separation", type=float, default=0.45
    )
    parser.add_argument(
        "--layout-mode", choices=("auto", "legacy", "custom"), default="auto"
    )
    parser.add_argument(
        "--place-near-target", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--relocation-distance", type=float, default=0.85)
    parser.add_argument("--dynamic-robot-clearance", type=float, default=0.70)
    parser.set_defaults(
        output_root=Path("experiments/results/maniskill_replicacad_multiscene_dynamic"),
        max_replans=140,
        post_grasp_retreat=0.40,
        grid_target_exclusion=0.34,
        lift_height=0.06,
        lift_stages=1,
        minimum_measured_lift=0.03,
        retreat_speed=0.03,
        retreat_max_steps=600,
        carry_speed=0.04,
        carry_yaw_rate=0.06,
        dock_position_tolerance=0.055,
        carry_additional_clearance=0.08,
        cartesian_carry_hold=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
