#!/usr/bin/env python3
"""Environment-only deterministic task construction, never an agent map.

Uses official geometry to place collision-clear fixtures and a force-controller
path. The policy must not load this raster, endpoint, or waypoints. All ten
official scene IDs are retained, including configuration failures.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import math

import cv2
import numpy as np

from architecthor_navigation import SCENE_IDS,derive_navigation_grid,_free_segment
from houseexpo_cross_room import weighted_distances,_reconstruct_path


def choose_task(grid,report,index):
    cells=np.argwhere(grid.free)
    home=np.asarray(report.raster_spawn_xy_m)
    start=grid.world_to_cell(home)
    home_dist,parents=weighted_distances(grid.free,start)
    xy=np.asarray([grid.cell_to_world(tuple(cell)) for cell in cells])
    radii=np.linalg.norm(xy-home,axis=1)
    # Initial memory target is within head RGB-D reach and in unoccluded free
    # space; no positive perception outcome is assumed or used for selection.
    candidate_ids=np.flatnonzero((radii>=1.25)&(radii<=1.65))
    candidate_ids=sorted(candidate_ids,key=lambda i:abs(radii[i]-1.45))
    initial_id=next((i for i in candidate_ids if _free_segment(grid,home,xy[i])),None)
    if initial_id is None:
        raise RuntimeError("No collision-clear initial memory fixture 1.25–1.65 m from spawn")
    initial=xy[initial_id]
    d,parents=weighted_distances(grid.free,tuple(cells[initial_id]))
    geodesic=d[cells[:,0],cells[:,1]]*grid.resolution_m
    euclidean=np.linalg.norm(xy-initial,axis=1)
    choices=np.flatnonzero((geodesic>=3.)&(geodesic<=4.5)&(euclidean>=2.)&(radii>=2.8))
    compact_fallback=False
    if not len(choices):
        # Some official start components are compact. Keep that scene with an
        # explicitly shorter task, rather than silently excluding the scene.
        choices=np.flatnonzero((geodesic>=1.5)&(geodesic<=4.5)&(euclidean>=1.)&(radii>=1.5))
        compact_fallback=True
    if not len(choices):
        raise RuntimeError("No connected relocation even under the disclosed compact-scene fallback")
    # Prefer a genuine obstacle detour, then proximity to the preselected 4 m
    # path-length target. This geometric choice is made before any policy run.
    rank=sorted(choices,key=lambda i:(_free_segment(grid,initial,xy[i]),
        abs(geodesic[i]-4.),-geodesic[i]/max(euclidean[i],.01)))
    endpoint_id=rank[0]
    path=_reconstruct_path(parents,tuple(cells[initial_id]),tuple(cells[endpoint_id]))
    waypoint_cells=path[3::3]
    waypoints=[np.asarray(grid.cell_to_world(tuple(c))).tolist() for c in waypoint_cells]
    near=np.flatnonzero((radii>=.62)&(radii<=.72))
    bin_id=next((i for i in sorted(near,key=lambda i:np.dot(xy[i]-home,initial-home))
        if _free_segment(grid,home,xy[i])),None)
    if bin_id is None:
        raise RuntimeError("No task receptacle dock near spawn")
    return dict(scene=report.scene_id,seed=17+index,spawn_xy=home.tolist(),
        target_xy=initial.tolist(),endpoint=xy[endpoint_id].tolist(),bin_xy=xy[bin_id].tolist(),
        disturbance_waypoints=waypoints,disturbance_step=480,
        query_step=480+int(math.ceil(geodesic[endpoint_id]/.05*20))+600,
        planned_relocation_path_m=float(geodesic[endpoint_id]),
        planned_relocation_euclidean_m=float(euclidean[endpoint_id]),
        relocation_has_obstacle_detour=not _free_segment(grid,initial,xy[endpoint_id]),
        compact_scene_fallback=compact_fallback,
        construction="Official geometry used ONLY to construct valid environment fixtures; policy has RGB-D and proprioception, no task geometry")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,default=Path(__file__).resolve().parents[2]/".maniskill_assets/data/scene_datasets/ai2thor")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--only",choices=SCENE_IDS)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for index,scene in enumerate(SCENE_IDS):
        if args.only and scene!=args.only:
            continue
        try:
            grid,report=derive_navigation_grid(args.assets,scene,index)
            task=choose_task(grid,report,index)
            (args.output/f"{index+1:02d}_{scene}.json").write_text(json.dumps(task,indent=2)+"\n")
            canvas=np.full((*grid.free.shape,3),235,np.uint8)
            canvas[grid.free]=(210,222,210)
            for name,color in (("spawn_xy",(220,90,30)),("target_xy",(40,180,240)),("endpoint",(40,70,230)),("bin_xy",(70,150,30))):
                cell=grid.world_to_cell(task[name])
                cv2.circle(canvas,tuple(cell[::-1]),3,color,-1)
            path=[grid.world_to_cell(p)[::-1] for p in [task["target_xy"],*task["disturbance_waypoints"],task["endpoint"]]]
            cv2.polylines(canvas,[np.asarray(path,np.int32)],False,(40,70,230),1)
            cv2.imwrite(str(args.output/f"{index+1:02d}_{scene}_environment_only.png"),cv2.resize(canvas,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
            row=dict(status="constructed_not_tested",**asdict(report),**task)
        except Exception as error:
            row=dict(scene=scene,status="configuration_failure",error=repr(error))
        reports.append(row)
        print(json.dumps(row),flush=True)
    (args.output/"construction_report.json").write_text(json.dumps(reports,indent=2)+"\n")


if __name__=="__main__":
    main()
