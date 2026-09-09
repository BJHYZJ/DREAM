#!/usr/bin/env python3
"""Extract exact recorded frames around events; never interpolate scene pixels."""
import argparse
import json
from pathlib import Path

import cv2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    events=[json.loads(line) for line in (args.run/"events.jsonl").read_text().splitlines()]
    frames=json.loads((args.run/"video_frames.json").read_text())
    selected={"depth_grasp_center","grasp_tcp_approach","gripper_closed","lift_attempt_complete",
              "visual_approach_complete","placement_tcp_approach","placement_attempt_complete",
              "arm_camera_scan_complete"}
    video=cv2.VideoCapture(str(args.run/"reviewer_view.mp4"))
    provenance=[]
    try:
        for event in events:
            if event["event"] not in selected:
                continue
            row=min(frames,key=lambda f:abs(f["sim_step"]-event["step"]))
            video.set(cv2.CAP_PROP_POS_FRAMES,row["frame"])
            ok,image=video.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {row}")
            name=f"{event['step']:05d}_{event['event']}.png"
            cv2.imwrite(str(args.output/name),image)
            provenance.append(dict(file=name,event=event,source_frame=row))
    finally:
        video.release()
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__=="__main__":
    main()
