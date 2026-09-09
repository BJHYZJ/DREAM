#!/usr/bin/env python3
"""Plot recorded observations and controls from one disclosed example episode."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    events=[json.loads(x) for x in (args.run/"events.jsonl").read_text().splitlines()]
    config=json.loads((args.run/"configuration.json").read_text())
    actions=json.loads((args.run/"actions.json").read_text())
    truth=json.loads((args.run/"evaluator_trajectory.json").read_text())
    truth_by_step={x["step"]:x for x in truth}
    observations=[x for x in events if x["event"]=="observation"]
    initial=[x for x in observations if x["step"]<config["disturbance_step"] and any(
        np.linalg.norm(np.array(d["point_world"])-truth_by_step[x["step"]]["target_xyz"])<.15 for d in x["detections"])]
    first=max(initial,key=lambda x:max(d["score"] for d in x["detections"]))
    focus=next(x for x in events if x["event"]=="focused_verification")
    focused=next(x for x in observations if x["step"]==focus["step"])
    approach=next(x for x in events if x["event"]=="visual_approach_complete")
    found=next(x for x in observations if x["frame_id"]==approach["detection"]["observation_id"])
    frames=json.loads((args.run/"video_frames.json").read_text())
    placed=next(x for x in events if x["event"]=="placement_attempt_complete")
    last=min(frames,key=lambda x:abs(x["sim_step"]-placed["step"]))
    cap=cv2.VideoCapture(str(args.run/"observation_raw.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES,last["frame"])
    ok,release=cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot decode recorded release observation")
    provenance=[]
    fig,axes=plt.subplots(1,4,figsize=(12,2.9),constrained_layout=True)
    for ax,event,title in zip(axes[:3],(first,focused,found),("Stored observation","Check remembered place","Reacquired target")):
        source=args.run/f"observation_{event['frame_id']:05d}.npz"
        with np.load(source,allow_pickle=False) as data:
            rgb=data["rgb"]
        ax.imshow(rgb)
        for d in event["detections"]:
            x0,y0,x1,y1=d["box_xyxy"]
            ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec="#f29b38",lw=1.2))
        ax.set_title(title,fontsize=10)
        ax.axis("off")
        provenance.append(dict(title=title,source=source.name,step=event["step"],
            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    axes[3].imshow(cv2.cvtColor(release,cv2.COLOR_BGR2RGB))
    axes[3].set_title("Released in task bin",fontsize=10)
    axes[3].axis("off")
    provenance.append(dict(title="Released in task bin",source="observation_raw.mp4",frame=last))
    fig.savefig(args.output/"recorded_observation_sequence.pdf")
    fig.savefig(args.output/"recorded_observation_sequence.png",dpi=200)
    plt.close(fig)
    base=np.array([x["base_xyyaw"][:2] for x in actions])
    target=np.array([x["target_xyz"] for x in truth])
    support=np.array([x["support_xyz"] for x in truth])
    times=np.array([x["step"] for x in truth])/20
    contact=np.array([x["bilateral_contact"] for x in truth])
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    axes[0].plot(base[:,0],base[:,1],color="#2d64a5",label="Measured robot route")
    axes[0].plot(support[:,0],support[:,1],"--",color="#d38637",label="Support motion (evaluator)")
    axes[0].scatter(*base[0],marker="s",c="#2d64a5",s=30)
    axes[0].set(xlabel="World x (m)",ylabel="World y (m)")
    axes[0].set_aspect("equal",adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[1].plot(times,target[:,2],label="Target height (evaluator)",color="#2d64a5")
    axes[1].fill_between(times,.65,1.3,where=contact,color="#82b28e",alpha=.22,label="Bilateral contact")
    axes[1].axvline(config["query_step"]/20,color=".4",ls=":",lw=1,label="Query issued")
    axes[1].set(xlabel="Simulation time (s)",ylabel="Height (m)",ylim=(.65,1.3))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.savefig(args.output/"executed_motion_and_contact.pdf")
    fig.savefig(args.output/"executed_motion_and_contact.png",dpi=180)
    plt.close(fig)
    (args.output/"provenance.json").write_text(json.dumps(dict(source_run=str(args.run.resolve()),
        observations=provenance,scope="One illustrated episode, not additional independent trials; boxes are recorded OWL detections; target/support truth is plotted offline only"),indent=2)+"\n")


if __name__=="__main__":
    main()
