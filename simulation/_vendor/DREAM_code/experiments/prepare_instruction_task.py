#!/usr/bin/env python3
"""Create a new instruction fixture from existing house geometry, not results."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from instruction_environment import sample_start


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-task",type=Path,required=True)
    p.add_argument("--recipe",default="tomato_bowl_white_table")
    p.add_argument("--recipes-json",type=Path,default=Path(__file__).parent/"configs/instruction_pickplace_v1/recipes.json")
    p.add_argument("--fit-furniture",action="store_true",help="Fit the recipe's actual support mesh into native free-space geometry")
    p.add_argument("--seed",type=int,default=100)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads(args.reference_task.read_text())
    recipes=json.loads(args.recipes_json.read_text())["recipes"]
    recipe=next(r for r in recipes if r["id"]==args.recipe)
    if not args.fit_furniture and recipe["id"] not in ("tomato_bowl_white_table","bread_plate_white_table"):
        raise ValueError("This initial fixture uses a visually inspected white-cloth table; other support recipes need matched furniture before construction")
    table_xy=np.asarray(source["bin_xy"])
    direction=np.asarray(source["spawn_xy"])-table_xy
    direction/=np.linalg.norm(direction)
    placement_xy=table_xy+.20*direction
    room_source=args.reference_task.parent/source["room_map_file"]
    shutil.copy2(room_source,args.output/"evaluator_room_map.npz")
    with np.load(room_source) as data:
        room_map={key:data[key] for key in data.files}
    support_asset="Dining_Table_9_1";table_radius=.40;furniture_checks={}
    if args.fit_furniture:
        from instruction_assets import inspect_asset
        from instruction_fixture_geometry import mesh_footprint,furniture_center_candidates,edge_placement
        support_asset=recipe["environment_assets"]["support"]
        directory=Path(__file__).resolve().parents[2]/".maniskill_assets/data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects"
        support=inspect_asset(directory,support_asset)
        destination=inspect_asset(directory,recipe["environment_assets"]["placement"])
        footprint,equations,local_center=mesh_footprint(support)
        table_radius=float(np.max(np.linalg.norm(footprint,axis=1)))+.02
        receptacle_radius=float(np.max((destination.upper[:2]-destination.lower[:2])/2))
        candidates,furniture_checks=furniture_center_candidates(room_map,footprint,source["initial_room"],table_xy)
        for center in candidates:
            proposed=edge_placement(center,equations,np.asarray(source["spawn_xy"]),receptacle_radius)
            # The actual externally driven cart route remains unobstructed.
            cart_path=np.array([source["target_xy"],*source.get("disturbance_waypoints",[]),source["endpoint"]])
            if np.min(np.linalg.norm(cart_path-center,axis=1))<table_radius+.22:
                continue
            try:
                sample_start(room_map,source["initial_room"],args.seed,[(center,table_radius)],
                    target_xy=source["target_xy"],docking_xy=source["endpoint"])
            except ValueError:
                continue
            table_xy=center-local_center;placement_xy=proposed
            furniture_checks.update(footprint_vertices_xy_m=footprint.tolist(),
                source_asset_sha256=support.sha256,conservative_radius_m=table_radius,
                status="geometry_only_not_robot_task_success")
            break
        else:
            raise ValueError("No furniture placement preserves valid connected starts and cart clearance")
    task=dict(protocol="instruction_pickplace_development_v1",status="constructed_not_executed",
        scene=source["scene"],instruction=recipe["instruction"],recipe=recipe,seed=args.seed,
        initial_room=source["initial_room"],target_xy=source["target_xy"],endpoint=source["endpoint"],
        disturbance_waypoints=source.get("disturbance_waypoints",[]),
        placement_table_asset=support_asset,placement_table_xy=table_xy.tolist(),
        placement_table_collision_radius_m=table_radius,placement_xy=placement_xy.tolist(),
        furniture_construction=furniture_checks,
        room_map_file="evaluator_room_map.npz",maximum_sim_seconds=1200,cart_rim_height_m=.02,
        construction_boundary="Development only: native house geometry reused, not historical robot routes/results. Random initial state and new query-driven destinations.")
    # The movable cart can initially block a doorway. It is not a permanent
    # obstacle for environment-level reachability screening: the disturbance
    # moves it. Initial start clearance is ensured by >=2 m pickup separation.
    discs=[(table_xy,table_radius)]
    spawn,yaw,sampling=sample_start(room_map,task["initial_room"],args.seed,discs,
                                  target_xy=task["target_xy"],docking_xy=task["endpoint"])
    task["initial_sample_preview"]=dict(spawn_xy=spawn.tolist(),yaw_rad=yaw,**sampling)
    (args.output/"task.json").write_text(json.dumps(task,indent=2)+"\n")
    print(json.dumps(task,indent=2),flush=True)


if __name__=="__main__":
    main()
