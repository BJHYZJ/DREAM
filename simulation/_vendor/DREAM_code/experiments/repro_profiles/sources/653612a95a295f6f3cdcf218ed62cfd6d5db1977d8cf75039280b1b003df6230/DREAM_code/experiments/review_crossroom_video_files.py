#!/usr/bin/env python3
"""Decode complete videos and extract unmodified recorded review keyframes.

Contact sheets are chapter previews, not replacements for uninterrupted videos.
The carrying check uses measured base displacement, not arm-only object motion.
"""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

from plot_crossroom_evidence import video_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cases=json.loads((args.gallery/"audited_successes.json").read_text())["cases"]
    reports=[]
    for index,case in enumerate(cases,1):
        run=Path(case["source_run"])
        log=args.output/f"{index:02d}_decode.log"
        with log.open("w") as stream:
            decode=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-xerror","-threads","2",
                "-i",str(run/"reviewer_view.mp4"),"-f","null","-"],stdout=stream,stderr=subprocess.STDOUT)
        if decode.returncode:
            raise RuntimeError(f"Complete video decode failed: {run}")
        events=[json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()]
        scenes=[("visual_approach_complete","Fresh visual approach"),("lift_attempt_complete","Lift"),
                ("held_arm_joint_fold","Folded-arm carry"),("placement_attempt_complete","Released placement")]
        sheet=np.full((664,1440,3),245,np.uint8)
        references=[]
        for i,(key,label) in enumerate(scenes):
            event=next(e for e in events if e["event"]==key)
            pixels,ref=video_frame(run,event["step"],"reviewer_view.mp4")
            pixels=cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR)
            x,y=(i%2)*720,(i//2)*332
            sheet[y+32:y+332,x:x+720]=cv2.resize(pixels,(720,300),interpolation=cv2.INTER_AREA)
            cv2.putText(sheet,f"{label} | {ref['sim_time_s']:.1f} s",(x+8,y+23),cv2.FONT_HERSHEY_SIMPLEX,.55,(35,45,50),1,cv2.LINE_AA)
            references.append(dict(label=label,**ref))
        cv2.imwrite(str(args.output/f"{index:02d}_{case['scene']}_keyframes.png"),sheet)
        actions=json.loads((run/"actions.json").read_text())
        truth=json.loads((run/"evaluator_trajectory.json").read_text())
        delta=np.linalg.norm(np.diff(np.asarray([a["base_xyyaw"][:2] for a in actions]),axis=0),axis=1)
        translating=(delta>.001) & np.asarray([r["phase"]=="Carry" for r in truth[1:]])
        held=np.asarray([r["bilateral_contact"] for r in truth[1:]])
        distance=float(delta[translating].sum())
        held_distance=float(delta[translating & held].sum())
        fraction=held_distance/distance if distance else 0.
        reports.append(dict(scene=case["scene"],source_run=str(run),full_video_decode_passed=True,
            measured_carry_base_distance_m=distance,measured_bilateral_carry_base_distance_m=held_distance,
            carry_translation_contact_fraction_distance_weighted=fraction,keyframes=references))
        print(json.dumps({k:v for k,v in reports[-1].items() if k!="keyframes"}),flush=True)
    (args.output/"video_review_measurements.json").write_text(json.dumps(dict(
        boundary="Read-only diagnostics and actual source-frame extraction; visual interpretation still requires inspection.",
        cases=reports),indent=2)+"\n")


if __name__=="__main__":
    main()
