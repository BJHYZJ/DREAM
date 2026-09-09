#!/usr/bin/env python3
"""Select multi-room native stages by geometry before running learned tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from prepare_crossroom_tasks import layers,room_partition


def worker(item):
    assets,output,config=item
    grid,stage,walls,_=layers(assets,config["scene"],config_relative=config["config_file"],include_objects=False)
    cores,labels,sizes=room_partition(stage,grid.resolution_m)
    clearance=distance_transform_edt(stage)*grid.resolution_m
    count,connected,stats,_=cv2.connectedComponentsWithStats((stage&(clearance>=.42)).astype(np.uint8),8)
    main=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])) if count>1 else 0
    room_ids=sorted(set(cores[(connected==main)&(cores>0)].tolist())) if main else []
    np.savez_compressed(output/f"{config['scene']}_stage_rooms.npz",stage_free=stage,room_labels=labels,
                        room_cores=cores,minimum_xy=grid.minimum_xy,maximum_xy=grid.maximum_xy,resolution=grid.resolution_m)
    rng=np.random.default_rng(42)
    colors=rng.integers(140,235,size=(max(2,len(sizes)+1),3),dtype=np.uint8)
    colors[0]=(245,245,245)
    canvas=colors[labels]
    canvas[walls.astype(bool)]=(35,45,50)
    cv2.imwrite(str(output/f"{config['scene']}_stage_rooms.png"),cv2.resize(canvas,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
    return dict(**config,room_cores=len(sizes),connected_room_cores=room_ids,floor_area_m2=float(stage.sum()*grid.resolution_m**2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=Path(__file__).resolve().parents[2]/".maniskill_assets/data/scene_datasets/ai2thor")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    configs=json.loads(args.candidates.read_text())["configs"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports=list(pool.map(worker,[(args.assets,args.output,c) for c in configs]))
    candidates=[row for row in reports if len(row["connected_room_cores"])>=2]
    candidates.sort(key=lambda row:(-len(row["connected_room_cores"]),-row["floor_area_m2"]))
    selected=candidates[:14]
    result=dict(selected_configs=[row["config_file"] for row in selected],selected=selected,all_candidates=reports,
                scope="Native stage geometry selection before learned execution, not a performance-selected benchmark split; furniture clearance is checked after referenced assets are downloaded.")
    (args.output/"selected_scenes.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
