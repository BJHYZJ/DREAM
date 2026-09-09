#!/usr/bin/env python3
"""Matched observation/feature replay from all ten dynamic suite attempts.

Independent of task success: no scene is selected because its robot succeeded.
This measures retention in the former support/target-height region, not
perception accuracy, new navigation trials, or pose-graph RMP robustness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from maniskill_learned_probe import ROOT
from dream_learned_core import LearnedPerception,SemanticMemory
from replay_learned_memory_evidence import read_observation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--wait-for-completion",action="store_true")
    args=parser.parse_args()
    while args.wait_for_completion and not (args.suite/"suite_result.json").is_file():
        time.sleep(10)
    suite=json.loads((args.suite/"suite_result.json").read_text())
    attempts=[x for x in suite["attempts"] if x["variant"]=="dynamic"]
    if len(attempts)!=10:
        raise RuntimeError("Ten scheduled dynamic attempts required, including failures")
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    perception=LearnedPerception(threshold=.25,model_set="production")
    results=[]
    for attempt in attempts:
        run=args.suite/attempt["name"]
        config=json.loads((run/"configuration.json").read_text())
        events=[json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()]
        views=[x for x in events if x["event"]=="observation"]
        initial=[x for x in views if x["step"]<config["disturbance_step"]][:8]
        post=[x for x in views if x["step"]>=config["query_step"]][:8]
        if len(initial)!=8 or len(post)!=8:
            results.append(dict(scene=attempt["scene"],status="insufficient_observations",pre_views=len(initial),post_views=len(post)))
            continue
        truths={x["step"]:x for x in json.loads((run/"evaluator_trajectory.json").read_text())}
        old=np.asarray(truths[initial[-1]["step"]]["target_xyz"])
        memories={variant:SemanticMemory(perception,dynamic=variant=="dynamic") for variant in ("dynamic","static")}
        timeline=[]
        hashes={}
        for index,event in enumerate(initial+post):
            source=run/f"observation_{event['frame_id']:05d}.npz"
            observation=read_observation(source)
            hashes[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
            features=perception.dense(observation)
            for variant,memory in memories.items():
                memory.integrate(observation,features)
                points=memory.cloud.points.numpy()
                region=(np.linalg.norm(points[:,:2]-old[:2],axis=1)<=.18)&(points[:,2]>=.88)&(points[:,2]<=1.08)
                scores=memory.alignments("apple").numpy()
                timeline.append(dict(index=index,frame_id=event["frame_id"],step=event["step"],variant=variant,
                    region_voxels=int(region.sum()),region_feature_score_sum=float(scores[region].sum()),
                    total_voxels=len(memory)))
        row=dict(scene=attempt["scene"],status="replayed",source_run=str(run.resolve()),
            source_attempt_status=attempt["status"],source_task_success=attempt.get("protocol_success",False),
            source_observation_sha256=hashes,old_region_evaluator_xyz=old.tolist(),
            before=[x for x in timeline if x["index"]==7],final=[x for x in timeline if x["index"]==15],timeline=timeline)
        results.append(row)
        (args.output/f"{attempt['scene']}.json").write_text(json.dumps(row,indent=2)+"\n")
        print(json.dumps({k:v for k,v in row.items() if k not in ("timeline","source_observation_sha256")}),flush=True)
    replayed=[x for x in results if x["status"]=="replayed"]
    counts={variant:np.array([next(f["region_voxels"] for f in x["final"] if f["variant"]==variant) for x in replayed]) for variant in ("dynamic","static")}
    differences=counts["static"]-counts["dynamic"]
    p=float(wilcoxon(counts["static"],counts["dynamic"],zero_method="wilcox").pvalue) if np.any(differences) else 1.
    fig,ax=plt.subplots(figsize=(9.5,3.0),constrained_layout=True)
    positions=np.arange(len(replayed))
    ax.bar(positions-.18,counts["static"],.36,color="#8797a5",label="No ray clearing")
    ax.bar(positions+.18,counts["dynamic"],.36,color="#2d64a5",label="Ray clearing")
    ax.set_xticks(positions,[x["scene"].replace("ArchitecTHOR-","") for x in replayed],rotation=30,ha="right")
    ax.set_ylabel("Former-region voxels")
    ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.15)
    fig.savefig(args.output/"matched_ten_scene_memory.pdf")
    fig.savefig(args.output/"matched_ten_scene_memory.png",dpi=180)
    plt.close(fig)
    report=dict(scheduled=10,replayed=len(replayed),same_rgbd_and_features=True,image_region_rejection_used=False,
        fixed_pre_views=8,fixed_post_views=8,old_region_radius_m=.18,height_slab_m=[.88,1.08],
        model_set="production",model_lock=json.loads((ROOT/".dream_model_cache/dream_models.lock.json").read_text()),
        median_counts={v:float(np.median(c)) for v,c in counts.items()},
        lower_with_clearing=int((differences>0).sum()),equal=int((differences==0).sum()),
        higher_with_clearing=int((differences<0).sum()),descriptive_wilcoxon_p=p,runs=results,
        boundary="Matched component replays of existing dynamic-policy streams, including failed tasks; not new independent robot trials, a perception benchmark, or held-out generalization. Region includes support/target-height surfaces, not perfect object-instance segmentation.")
    (args.output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="runs"},indent=2),flush=True)


if __name__=="__main__":
    main()
