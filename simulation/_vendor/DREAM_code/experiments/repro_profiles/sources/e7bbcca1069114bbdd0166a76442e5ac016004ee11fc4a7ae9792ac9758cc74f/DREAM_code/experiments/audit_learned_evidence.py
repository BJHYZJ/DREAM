#!/usr/bin/env python3
"""Check saved hashes, physical success, camera provenance and stage coverage.

Does not infer algorithm effectiveness from successful videos or voxel counts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from learned_evaluation import evaluate_episode


def audit(run):
    result=json.loads((run/"result.json").read_text())
    config=json.loads((run/"configuration.json").read_text())
    rows=json.loads((run/"evaluator_trajectory.json").read_text())
    actions=json.loads((run/"actions.json").read_text())
    events=[json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()]
    hashes=json.loads((run/"sha256.json").read_text())
    mismatches=[]
    for name,expected in hashes.items():
        path=run/"source_snapshot"/name[7:] if name.startswith("source:") else run/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            mismatches.append(name)
    positions=np.asarray([r["base_xyyaw"][:2] for r in actions])
    max_delta=float(np.linalg.norm(np.diff(positions,axis=0),axis=1).max()) if len(positions)>1 else 0.
    report=dict(run=str(run),variant=config["variant"],model_set=config["model_set"],
        hash_mismatches=mismatches,action_steps_contiguous=[r["step"] for r in actions]==list(range(1,len(actions)+1)),
        evaluator_steps_match_actions=[r["step"] for r in rows]==[r["step"] for r in actions],
        max_base_step_displacement_m=max_delta,
        wrist_memory_observations=sum(e["event"]=="observation" and e["memory"]["sensor"]=="fetch_hand" for e in events))
    if rows and "phase" in rows[0]:
        score=evaluate_episode(rows,actions,events,np.asarray(result["task_known_bin_xy"]),
                               require_delivery=config.get("deliver",False))
        report["recomputed_score"]=score
        report["recorded_score_matches"]=all(result.get(k)==v for k,v in score.items())
    if (run/"video_frames.json").is_file():
        frames=json.loads((run/"video_frames.json").read_text())
        import cv2
        video=cv2.VideoCapture(str(run/"reviewer_view.mp4"))
        count=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=video.get(cv2.CAP_PROP_FPS)
        video.release()
        report["video"]=dict(frame_count=count,manifest_count=len(frames),fps=fps,
            all_source_gaps_four_steps=bool(np.all(np.diff([f["sim_step"] for f in frames])==4)),
            frame_count_matches=count==len(frames),phases=sorted({f["phase"] for f in frames}),
            duration_s=count/fps if fps else None)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    reports=[audit(run) for run in args.runs]
    value=json.dumps(reports,indent=2)+"\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(value)
    print(value)


if __name__=="__main__":
    main()
