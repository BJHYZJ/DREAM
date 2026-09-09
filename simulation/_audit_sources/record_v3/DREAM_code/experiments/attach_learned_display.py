#!/usr/bin/env python3
"""Attach verified display re-executions without changing recorded outcomes."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery",type=Path,required=True)
    parser.add_argument("--display",type=Path,required=True)
    parser.add_argument("--manuscript",type=Path)
    args=parser.parse_args()
    audit=json.loads((args.gallery/"audit_and_results.json").read_text())
    reports=json.loads((args.display/"all_reexecutions.json").read_text())
    by_name={Path(r["source_run"]).name:r for r in reports}
    runs=sorted((r for r in audit["runs"] if r["variant"]=="dynamic"),key=lambda r:r["name"])
    if len(runs)!=10 or len(reports)!=10:
        raise RuntimeError("Ten source recordings and verified re-executions required")
    sheet=np.full((480,2000,3),245,np.uint8)
    for index,run in enumerate(runs):
        report=by_name[run["name"]]
        if not report["passed"] or report["preview_only"] or report["executed_steps"]!=run["env_steps"]:
            raise RuntimeError(f"Incomplete display verification: {run['name']}")
        source=args.display/run["name"]/"reviewer_view.mp4"
        cap=cv2.VideoCapture(str(source))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if count!=run["video"]["frames"] or fps!=5:
            raise RuntimeError("Display/source frame counts differ")
        link=args.gallery/f"{index+1:02d}_{run['scene']}.mp4"
        if not link.is_symlink():
            raise RuntimeError(f"Refuse to replace a material video file: {link}")
        # Replace only a validated viewing shortcut; source movies are kept.
        link.unlink()
        link.symlink_to(source.resolve())
        frame=cv2.imread(str(args.display/run["name"]/"scene_step_00360.png"))
        if frame is None:
            raise RuntimeError("Missing recorded fixed-step poster")
        frame=cv2.resize(frame,(400,225))
        cv2.imwrite(str(args.gallery/f"{index+1:02d}_scene.jpg"),frame)
        tile=np.full((240,400,3),245,np.uint8)
        tile[:225]=frame
        label=f"{index+1:02d} {run['scene'].replace('ArchitecTHOR-','')} | {'PASS' if run['protocol_success'] else 'NOT COMPLETED'}"
        cv2.putText(tile,label,(6,237),cv2.FONT_HERSHEY_SIMPLEX,.38,(35,45,50),1,cv2.LINE_AA)
        sheet[index//5*240:(index//5+1)*240,index%5*400:(index%5+1)*400]=tile
    cv2.imwrite(str(args.gallery/"learned_scene_contact_sheet.png"),sheet)
    (args.gallery/"display_reexecution_verification.json").write_text(json.dumps(reports,indent=2)+"\n")
    note=("The left overview is re-rendered by re-executing every saved physical action with a lower display camera; "
        "per-step base/yaw/target/TCP agreement is verified. The right observation/map panel is from the original policy recording. "
        "No physical steps are removed, no recorded actor poses are set, and this is not a new algorithm trial.")
    html=args.gallery/"index.html"
    content=html.read_text()
    content=content.replace("<h1>Learned DREAM / ManiSkill</h1>","<h1>Learned DREAM / ManiSkill</h1><p>"+note+"</p>")
    html.write_text(content)
    readme=args.gallery/"README.md"
    readme.write_text(readme.read_text()+"\n## Display re-execution\n\n"+note+"\n")
    if args.manuscript:
        shutil.copy2(args.gallery/"learned_scene_contact_sheet.png",args.manuscript/"figs/learned_scene_contact_sheet.png")
    print(json.dumps(dict(videos=10,all_reexecutions_verified=True)),flush=True)


if __name__=="__main__":
    main()
