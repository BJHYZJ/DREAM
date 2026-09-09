#!/usr/bin/env python3
"""Construct geometry-screened tasks in distinct native ManiSkill houses.

This creates fixtures, not results. Room maps and support relocation paths are
environment/evaluator inputs only. Full learned-policy episodes are separate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_crossroom_tasks import layers, room_partition, construct_remembered_room


def worker(arguments):
    assets, output, index, record = arguments
    scene=record["scene"]
    grid,stage,walls,furniture=layers(assets,scene,config_relative=record["config_file"])
    cores,labels,sizes=room_partition(stage,grid.resolution_m)
    np.savez_compressed(output/f"{scene}_environment_rooms.npz",raw_free=grid.free,stage_free=stage,
        room_cores=cores,room_labels=labels,minimum_xy=grid.minimum_xy,maximum_xy=grid.maximum_xy,
        resolution=grid.resolution_m)
    colors=np.random.default_rng(42).integers(140,235,size=(len(sizes)+1,3),dtype=np.uint8)
    colors[0]=(245,245,245)
    canvas=colors[labels]
    canvas[walls.astype(bool)]=(30,40,45)
    canvas[furniture.astype(bool)]=(95,105,115)
    rows=[]
    for variant in range(2):
        try:
            task=construct_remembered_room(grid,cores,labels,0,variant,scene_id=scene,
                preferred_spawn=(grid.minimum_xy+grid.maximum_xy)/2,whole_house=True)
            task["seed"]=417+index*3+variant
            task["native_scene_config"]=record["config_file"]
            name=f"{index+1:02d}_{scene}_layout{variant+1:02d}.json"
            (output/name).write_text(json.dumps(task,indent=2)+"\n")
            rows.append(dict(status="constructed_not_executed",file=name,**task))
            if variant==0:
                pts=np.array([grid.world_to_cell(p)[::-1] for p in [task["target_xy"],*task["disturbance_waypoints"],task["endpoint"]]],np.int32)
                cv2.polylines(canvas,[pts],False,(20,40,210),2)
                for field,color in (("spawn_xy",(230,70,20)),("target_xy",(0,200,240)),("endpoint",(20,20,230)),("bin_xy",(20,150,20))):
                    cv2.circle(canvas,grid.world_to_cell(task[field])[::-1],4,color,-1)
        except Exception as error:
            rows.append(dict(status="geometry_configuration_failed",variant=variant,error=repr(error)))
    for label in range(1,len(sizes)+1):
        center=tuple(np.mean(np.argwhere(cores==label),axis=0).astype(int)[::-1])
        cv2.putText(canvas,f"R{label}",center,cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
    cv2.imwrite(str(output/f"{scene}_environment_rooms.png"),cv2.resize(canvas,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
    report=dict(scene=scene,native_scene_config=record["config_file"],room_core_count=len(sizes),
                room_core_areas_m2=sizes,stage_floor_area_m2=float(stage.sum()*grid.resolution_m**2),tasks=rows)
    (output/f"{scene}_construction.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(scene=scene,statuses=[r["status"] for r in rows])),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,default=Path(__file__).resolve().parents[2]/".maniskill_assets/data/scene_datasets/ai2thor")
    parser.add_argument("--selected",type=Path,default=Path(__file__).with_name("configs")/"procthor_stage_geometry_v3/selected_scenes.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    selected=json.loads(args.selected.read_text())["selected"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports=list(pool.map(worker,[(args.assets,args.output,i,r) for i,r in enumerate(selected)]))
    (args.output/"construction_report.json").write_text(json.dumps(reports,indent=2)+"\n")


if __name__=="__main__":
    main()
