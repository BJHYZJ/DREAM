#!/usr/bin/env python3
"""Construct v2 fixtures with robot/bin clearance, before any learned run.

Only the environment uses official geometry. No generated grid, future cart
waypoint or target endpoint may enter the learned policy. Every scene is kept,
including a configuration failure; this is development, not held-out testing.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from architecthor_navigation import SCENE_IDS,derive_navigation_grid,_free_segment
from houseexpo_cross_room import weighted_distances,_reconstruct_path


def construct(grid,report,index):
    cells=np.argwhere(grid.free)
    xy=np.array([grid.cell_to_world(tuple(c)) for c in cells])
    original=np.asarray(report.raster_spawn_xy_m)
    clearance=distance_transform_edt(grid.free)*grid.resolution_m
    # Grid already includes 0.30 m furniture inflation. Extra clearance is
    # needed for the initial physical arm scan and not just the base centre.
    home_ids=np.flatnonzero(clearance[cells[:,0],cells[:,1]]>=.22)
    home_ids=sorted(home_ids,key=lambda i:np.linalg.norm(xy[i]-original))[:40]
    for home_id in home_ids:
        home=xy[home_id]
        radii=np.linalg.norm(xy-home,axis=1)
        initial_ids=np.flatnonzero((radii>=1.25)&(radii<=1.65))
        initial_ids=sorted(initial_ids,key=lambda i:abs(radii[i]-1.45))[:50]
        for initial_id in initial_ids:
            initial=xy[initial_id]
            if not _free_segment(grid,home,initial):
                continue
            bins=np.flatnonzero((radii>=.67)&(radii<=.76))
            bins=sorted(bins,key=lambda i:np.dot(xy[i]-home,initial-home))
            bin_id=next((i for i in bins if _free_segment(grid,home,xy[i])),None)
            if bin_id is None:
                continue
            bin_xy=xy[bin_id]
            cart_free=grid.free.copy()
            forbidden=(radii<.80)|(np.linalg.norm(xy-bin_xy,axis=1)<.60)
            cart_free[cells[forbidden,0],cells[forbidden,1]]=False
            if not cart_free[tuple(cells[initial_id])]:
                continue
            distances,parents=weighted_distances(cart_free,tuple(cells[initial_id]))
            geodesic=distances[cells[:,0],cells[:,1]]*grid.resolution_m
            separation=np.linalg.norm(xy-initial,axis=1)
            candidates=np.flatnonzero((geodesic>=3.)&(geodesic<=4.5)&(separation>=2.)&(radii>=2.8))
            compact=False
            if not len(candidates):
                candidates=np.flatnonzero((geodesic>=1.5)&(geodesic<=4.5)&(separation>=1.)&(radii>=1.5))
                compact=True
            if not len(candidates):
                continue
            end=min(candidates,key=lambda i:(_free_segment(grid,initial,xy[i]),abs(geodesic[i]-4.)))
            path=_reconstruct_path(parents,tuple(cells[initial_id]),tuple(cells[end]))
            waypoints=[list(grid.cell_to_world(tuple(c))) for c in path[3::3]]
            path_xy=np.array([grid.cell_to_world(tuple(c)) for c in path])
            return dict(scene=report.scene_id,seed=17+index,spawn_xy=home.tolist(),
                target_xy=initial.tolist(),endpoint=xy[end].tolist(),bin_xy=bin_xy.tolist(),
                disturbance_waypoints=waypoints,disturbance_step=600,
                query_step=600+int(math.ceil(geodesic[end]/.05*20))+600,
                planned_relocation_path_m=float(geodesic[end]),
                planned_relocation_euclidean_m=float(separation[end]),
                relocation_has_obstacle_detour=not _free_segment(grid,initial,xy[end]),
                compact_scene_fallback=compact,
                minimum_cart_path_home_distance_m=float(np.linalg.norm(path_xy-home,axis=1).min()),
                minimum_cart_path_bin_distance_m=float(np.linalg.norm(path_xy-bin_xy,axis=1).min()),
                home_extra_raster_clearance_m=float(clearance[tuple(cells[home_id])]),
                construction="v2 environment-only geometry: rotation clearance and robot/bin exclusion from force path; no policy performance used to select configurations")
    raise RuntimeError("No valid fixture configuration under fixed geometric rules")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,default=Path(__file__).resolve().parents[2]/".maniskill_assets/data/scene_datasets/ai2thor")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for index,scene in enumerate(SCENE_IDS):
        try:
            grid,report=derive_navigation_grid(args.assets,scene,index)
            task=construct(grid,report,index)
            (args.output/f"{index+1:02d}_{scene}.json").write_text(json.dumps(task,indent=2)+"\n")
            canvas=np.full((*grid.free.shape,3),240,np.uint8)
            canvas[grid.free]=(200,220,200)
            for field,color in (("spawn_xy",(220,70,30)),("bin_xy",(30,130,30)),("target_xy",(0,170,240)),("endpoint",(20,20,230))):
                cv2.circle(canvas,tuple(grid.world_to_cell(task[field]))[::-1],3,color,-1)
            points=[grid.world_to_cell(p)[::-1] for p in [task["target_xy"],*task["disturbance_waypoints"],task["endpoint"]]]
            cv2.polylines(canvas,[np.array(points,np.int32)],False,(30,70,220),1)
            cv2.imwrite(str(args.output/f"{index+1:02d}_{scene}_environment_only.png"),cv2.resize(canvas,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
            row=dict(status="constructed_not_tested",**asdict(report),**task)
        except Exception as error:
            row=dict(scene=scene,status="configuration_failure",error=repr(error))
        rows.append(row)
        print(json.dumps(row),flush=True)
    (args.output/"construction_report.json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
