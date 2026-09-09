#!/usr/bin/env python3
"""Matched learned-memory replay from actual saved simulator RGB-D.

This isolates ray clearing on an identical observation stream, NOT navigation
success, SLAM/RMP, or an independent new robotic episode. Ground truth is used
only after integration to score obsolete support/object-region voxels.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

from maniskill_learned_probe import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dream_learned_core import RGBDObservation,LearnedPerception,SemanticMemory


def read_observation(path):
    with np.load(path,allow_pickle=False) as data:
        return RGBDObservation(int(data["frame_id"]),int(data["sim_step"]),str(data["sensor"]),
            data["rgb"],data["depth_m"],data["intrinsics"],data["camera_to_world_cv"],data["base_xyyaw"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--post-views",type=int,default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    config=json.loads((args.source_run/"configuration.json").read_text())
    events=[json.loads(line) for line in (args.source_run/"events.jsonl").read_text().splitlines()]
    observations=[e for e in events if e["event"]=="observation"]
    disturbance=config.get("disturbance_step",480)
    initial=[e for e in observations if e["step"]<disturbance]
    after=[e for e in observations if e["step"]>=config["query_step"]][:args.post_views]
    if not initial or not after:
        raise ValueError("Replay needs pre-relocation and post-request observations")
    perception=LearnedPerception(threshold=config["threshold"],model_set=config["model_set"])
    memories={v:SemanticMemory(perception,dynamic=v=="dynamic") for v in ("dynamic","static")}
    trajectory=json.loads((args.source_run/"evaluator_trajectory.json").read_text())
    evaluation_by_step={r["step"]:r for r in trajectory}
    old_center=np.asarray(evaluation_by_step[initial[-1]["step"]]["target_xyz"])
    report=[]
    sources={}
    snapshots={}
    for index,event in enumerate(initial+after):
        path=args.source_run/f"observation_{event['frame_id']:05d}.npz"
        obs=read_observation(path)
        sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        features=perception.dense(obs)
        for variant,memory in memories.items():
            update=memory.integrate(obs,features)
            points=memory.cloud.points.numpy()
            ghost=(np.linalg.norm(points[:,:2]-old_center[:2],axis=1)<=.18)&(points[:,2]>=.88)&(points[:,2]<=1.08)
            row=dict(variant=variant,frame_id=obs.frame_id,sim_step=obs.sim_step,
                obsolete_region_voxels=int(ghost.sum()),total_voxels=len(memory),removed=update["removed"])
            report.append(row)
            if index==len(initial)-1 or index==len(initial+after)-1:
                key=("before" if index==len(initial)-1 else "after")+"_"+variant
                snapshots[key]=dict(points=points.copy(),scores=memory.alignments(config["query"]).numpy().copy())
        print(json.dumps(report[-2:]),flush=True)
    # Evaluator annotations are kept outside both replay memory inputs.
    new_center=np.asarray(evaluation_by_step[after[-1]["step"]]["target_xyz"])
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.3),constrained_layout=True)
    padding=np.array([.65,.65])
    low=np.minimum(old_center[:2],new_center[:2])-padding
    high=np.maximum(old_center[:2],new_center[:2])+padding
    for ax,key,title in zip(axes,("before_dynamic","after_static","after_dynamic"),
                            ("Before relocation","Same RGB-D: no clearing","Same RGB-D: ray clearing")):
        points=snapshots[key]["points"]
        scores=snapshots[key]["scores"]
        selected=(points[:,2]>=.88)&(points[:,2]<=1.08)
        selected&=(points[:,:2]>=low).all(axis=1)&(points[:,:2]<=high).all(axis=1)
        plotted=ax.scatter(points[selected,0],points[selected,1],c=scores[selected],s=14,
                           marker="s",vmin=.05,vmax=.26,cmap="viridis")
        ax.add_patch(plt.Circle(old_center[:2],.18,fill=False,color="#d65b43",ls="--",lw=1.4))
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel="World x (m)",title=title)
        ax.set_aspect("equal")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("World y (m)")
    fig.colorbar(plotted,ax=axes,shrink=.65,label="Text-feature alignment")
    fig.savefig(args.output/"matched_memory_replay.pdf")
    fig.savefig(args.output/"matched_memory_replay.png",dpi=180)
    plt.close(fig)
    np.savez_compressed(args.output/"plotted_arrays.npz",**{
        f"{key}_{name}":value for key,data in snapshots.items() for name,value in data.items()})
    result=dict(source_run=str(args.source_run.resolve()),source_observation_sha256=sources,
        model_set=config["model_set"],query=config["query"],same_features_shared_between_variants=True,
        rejection_filter_used=False,old_target_region_evaluator_xyz=old_center.tolist(),
        after_target_evaluator_xyz=new_center.tolist(),pre_views=len(initial),post_views=len(after),
        final=[r for r in report if r["frame_id"]==after[-1]["frame_id"]],timeline=report,
        boundary="Paired replay of one existing development episode, not independent trials or a navigation performance comparison. Obsolete-region count includes support/target-height voxels; colors are actual text-feature alignments, not detector accuracy.")
    (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("timeline","source_observation_sha256")},indent=2))


if __name__=="__main__":
    main()
