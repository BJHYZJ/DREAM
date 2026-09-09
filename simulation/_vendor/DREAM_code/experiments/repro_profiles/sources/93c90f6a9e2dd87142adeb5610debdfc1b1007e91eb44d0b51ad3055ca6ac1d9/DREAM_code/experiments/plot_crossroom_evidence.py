#!/usr/bin/env python3
"""Plot recorded trajectories and simulator pixels from an audited gallery.

Floor plans and target traces are explicitly offline evaluator illustrations,
never the policy's observed semantic map. No synthesized scene images are used.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def read(path):
    return json.loads(path.read_text())


def video_frame(run,step,filename):
    frames=read(run/"video_frames.json")
    frame=min(frames,key=lambda row:abs(row["sim_step"]-step))
    cap=cv2.VideoCapture(str(run/filename))
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame["frame"])
    ok,pixels=cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode {run}/{filename} at {step}")
    return cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB),dict(video=filename,**frame)


def save(fig,output,name):
    fig.savefig(output/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(output/f"{name}.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    gallery=read(args.gallery/"audited_successes.json")
    cases=gallery["cases"]
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({"font.family":"DejaVu Serif","font.size":8,"pdf.fonttype":42})
    columns=min(5,len(cases))
    rows=math.ceil(len(cases)/columns)
    fig,axes=plt.subplots(rows,columns,figsize=(7.16,2.25*rows),squeeze=False)
    provenance=[]
    for index,(case,ax) in enumerate(zip(cases,axes.flat),1):
        run=Path(case["source_run"])
        actions=read(run/"actions.json")
        truth=read(run/"evaluator_trajectory.json")
        events=[json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()]
        config=read(run/"configuration.json")
        task=read(run/"environment_task.json")
        with np.load(run/"evaluator_room_map.npz") as room:
            free=room["stage_free"]
            furnished=room["raw_free"]
            labels=room["room_labels"]
            rgb=np.full((*free.shape,3),.30)
            rgb[free]=[.97,.97,.97]
            rgb[labels==task["initial_room"]]=[.88,.93,.99]
            rgb[labels==task["destination_room"]]=[.94,.92,.85]
            rgb[free & ~furnished]=[.70,.72,.73]
            lo,hi=room["minimum_xy"],room["maximum_xy"]
            ax.imshow(rgb,extent=[lo[0],hi[0],lo[1],hi[1]],origin="upper",interpolation="nearest")
        xy=np.asarray([row["base_xyyaw"][:2] for row in actions])
        phase=np.asarray([row["phase"] for row in truth])
        times=np.asarray([row["step"] for row in actions])
        search=(times>=config["query_step"]) & ~np.isin(phase,["Carry","Place","Check placement"])
        carry=np.isin(phase,["Carry","Place","Check placement"])
        for mask,color in ((search,"#2274ad"),(carry,"#229365")):
            path=xy.copy()
            path[~mask]=np.nan
            ax.plot(path[:,0],path[:,1],color=color,lw=1.25,zorder=3)
        support=np.asarray([row["support_xyz"][:2] for row in truth])
        interval=(times>=config["disturbance_step"]) & (times<=config["query_step"])
        ax.plot(support[interval,0],support[interval,1],"--",color="#c77828",lw=1,zorder=4)
        ax.scatter(*xy[0],s=16,color="#22313c",marker="o",zorder=5)
        ax.scatter(*task["bin_xy"],s=20,facecolors="none",edgecolors="#229365",marker="s",zorder=5)
        end=truth[config["query_step"]-1]["target_xyz"][:2]
        ax.scatter(*end,s=22,color="#ba4a32",marker="*",zorder=5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{index:02d}  {case['scene'].replace('ProcTHOR-','P-').replace('ArchitecTHOR-','A-')}\n"
                     f"{case['metrics']['evaluator_base_travel_m']:.1f} m travel",fontsize=8)
        # A physical scale is explicit; different house extents are not equal.
        x0=lo[0]+.07*(hi[0]-lo[0])
        y0=lo[1]+.05*(hi[1]-lo[1])
        ax.plot([x0,x0+2],[y0,y0],color="black",lw=2)
        ax.text(x0+1,y0+.18,"2 m",ha="center",fontsize=6.5)
        provenance.append(dict(scene=case["scene"],source_run=str(run),
            plot_inputs=["actions.json","evaluator_trajectory.json","evaluator_room_map.npz","environment_task.json"],
            total_house_floor_area_m2=case["house_stage_floor_area_m2"],
            interpretation="Whole-house geometry is offline evaluator context, not explored area or policy input."))
    for ax in list(axes.flat)[len(cases):]:
        ax.axis("off")
    handles=[Line2D([0],[0],color="#2274ad",label="Search"),Line2D([0],[0],color="#229365",label="Carry / return"),
             Line2D([0],[0],color="#c77828",ls="--",label="External support motion"),
             Line2D([0],[0],color="#22313c",marker="o",lw=0,label="Start"),
             Line2D([0],[0],color="#229365",marker="s",fillstyle="none",lw=0,label="Known bin"),
             Line2D([0],[0],color="#ba4a32",marker="*",lw=0,label="Relocated target")]
    fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=7,frameon=False)
    fig.tight_layout(rect=[0,.16 if rows==1 else .085,1,1],h_pad=2.1)
    save(fig,args.output,"crossroom_layouts")

    # One transparent illustrative case; never counted as extra trials.
    run=Path(cases[0]["source_run"])
    result=read(run/"result.json")
    config=read(run/"configuration.json")
    events=[json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()]
    remembered=result["before"]["observation_id"]
    memory_event=next(e for e in events if e["event"]=="observation" and e["frame_id"]==remembered)
    focus=next(e for e in events if e["event"]=="focused_verification")
    found=next(e for e in events if e["event"]=="visual_approach_complete")
    snapshots=[("Remember",memory_event["step"]),("Check old location",focus["step"]),
               ("Find in next room",found["step"]),("Released in bin",result["env_steps"]-8)]
    fig,axes=plt.subplots(1,4,figsize=(7.16,1.65))
    references=[]
    for ax,(title,step) in zip(axes,snapshots):
        pixels,ref=video_frame(run,step,"observation_raw.mp4")
        ax.imshow(pixels)
        ax.axis("off")
        ax.set_title(f"{title}\n{ref['sim_time_s']:.1f} s",fontsize=8)
        references.append(dict(label=title,**ref))
    fig.tight_layout(pad=.35)
    save(fig,args.output,"crossroom_observation_sequence")

    truth=read(run/"evaluator_trajectory.json")
    t=np.asarray([r["step"]/20 for r in truth])
    z=np.asarray([r["target_xyz"][2] for r in truth])
    hold=np.asarray([r["bilateral_contact"] for r in truth],dtype=float)
    fig,axes=plt.subplots(2,1,figsize=(7.16,2.2),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    axes[0].plot(t,z,color="#2274ad",lw=1)
    axes[0].set_ylabel("Target height (m)")
    axes[1].fill_between(t,0,hold,color="#229365",alpha=.75,step="post")
    axes[1].set_ylabel("Bilateral\ncontact")
    axes[1].set_yticks([0,1])
    axes[1].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.axvspan(config["disturbance_step"]/20,result["disturbance_done_step"]/20,color="#c77828",alpha=.12)
        ax.axvline(config["query_step"]/20,color="gray",ls=":",lw=.8)
        ax.grid(alpha=.16)
    fig.tight_layout(pad=.6)
    save(fig,args.output,"crossroom_contact_timeline")
    (args.output/"figure_provenance.json").write_text(json.dumps(dict(partial_preview=gallery["partial_preview"],
        cases=provenance,illustration_source_run=str(run),observation_frames=references,
        boundary="Curated actual successes; generated figures are not extra trials. Geometry/target truth is evaluator-only."),indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output),cases=len(cases),partial_preview=gallery["partial_preview"])))


if __name__=="__main__":
    main()
