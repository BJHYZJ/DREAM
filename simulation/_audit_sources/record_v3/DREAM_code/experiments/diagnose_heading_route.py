#!/usr/bin/env python3
"""Plan and sweep-check a heading route on recorded observations; DO NOT execute."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from dream_fetch_navigation import FetchObservedMap
from dream_fetch_heading_astar import FetchHeadingAStar
from houseexpo_cross_room import weighted_distances
from maniskill_learned_probe import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",type=Path,required=True)
    parser.add_argument("--connectivity",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--maximum-goals",type=int,default=12)
    args=parser.parse_args();started=time.monotonic()
    footprint=json.loads((args.replay/"robot_footprint.json").read_text())
    diagnostic=json.loads((args.connectivity/"diagnostic.json").read_text())
    if diagnostic["robot_footprint_sha256"]!=sha(args.replay/"robot_footprint.json"):
        raise ValueError("Footprint differs from the map-connectivity diagnostic")
    with np.load(args.connectivity/"diagnostic_maps.npz") as data:
        observed=FetchObservedMap(origin=data["origin"],size=len(data["known"]),resolution=float(data["resolution"]))
        observed.known=data["known"].copy();observed.inferred_blocked=data["inferred_blocked"].copy()
    base=np.asarray(footprint["base_xyyaw"])
    planner=FetchHeadingAStar(observed,footprint,base)
    union=planner.pose_masks.any(axis=0);start=tuple(observed.cells(base[:2]))
    distances,_=weighted_distances(union,start)
    cells=np.argwhere(union&np.isfinite(distances))
    positions=observed.world(cells)+planner.offset
    query=np.asarray(diagnostic["query_from_actual_memory_event"]["detection"]["point_world"])
    ranges=np.linalg.norm(positions-query[:2],axis=1)
    candidates=cells[(ranges>=.58)&(ranges<=.64)]
    order=np.argsort(distances[candidates[:,0],candidates[:,1]])
    trials=[];route=[]
    for cell in candidates[order]:
        xy=observed.world(cell)+planner.offset;delta=query[:2]-xy
        goal=np.r_[xy,math.atan2(delta[1],delta[0])]
        if planner.point_is_occupied(*planner.to_pt(goal)):continue
        route=planner.path(goal)
        valid=bool(route) and planner.validate_route(route)
        trials.append(dict(goal_xyyaw=goal.tolist(),states=len(route),swept_route_valid=valid,
                           expanded=planner.expanded,budget_exceeded=planner.search_budget_exceeded))
        if valid:break
        route=[]
        if len(trials)>=args.maximum_goals:break
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(source_run=footprint["source_run"],source_footprint_sha256=sha(args.replay/"robot_footprint.json"),
        connectivity_diagnostic_sha256=sha(args.connectivity/"diagnostic.json"),
        diagnostic_maps_sha256=sha(args.connectivity/"diagnostic_maps.npz"),
        source_query=diagnostic["query_from_actual_memory_event"],
        planner_method=planner.run_astar.__func__.__module__+".AStar.run_astar",
        grid_offset_xy=planner.offset.tolist(),candidate_docking_cells=len(candidates),
        trials=trials,route_xyyaw=route,recorded_map_swept_route_found=bool(route),
        wall_time_s=time.monotonic()-started,physical_motion_executed=False,task_success_verified=False,
        boundary="Saved observed occupancy, measured robot footprint and learned memory query only. DREAM A* with heading-state motion edges; every returned edge sweep checked. Offline route feasibility is not task execution or success.")
    (args.output/"route_diagnostic.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_query","route_xyyaw")},indent=2))


if __name__=="__main__":main()
