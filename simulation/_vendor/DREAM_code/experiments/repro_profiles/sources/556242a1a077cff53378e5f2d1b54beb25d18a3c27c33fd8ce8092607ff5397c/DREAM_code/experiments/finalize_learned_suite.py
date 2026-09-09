#!/usr/bin/env python3
"""Build an all-attempts report and chaptered video gallery after a suite ends."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.stats import binomtest

from learned_evaluation import evaluate_episode,evaluate_dynamic_protocol


def load(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def inspect_run(run,attempt):
    result=load(run/"result.json",{})
    config=load(run/"configuration.json",{})
    actions=load(run/"actions.json",[])
    trajectory=load(run/"evaluator_trajectory.json",[])
    events=[json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()] if (run/"events.jsonl").is_file() else []
    task=load(run/"environment_task.json",{})
    scores={}
    if trajectory and config:
        start=np.asarray(actions[0]["base_xyyaw"][:2])
        destination=np.asarray(task.get("bin_xy",result.get("task_known_bin_xy",start+[-.7,0.])))
        scores=evaluate_episode(trajectory,actions,events,destination,require_delivery=True)
        scores.update(evaluate_dynamic_protocol(trajectory,events,
            disturbance_step=config.get("disturbance_step",480),query_step=config["query_step"],endpoint=np.asarray(config["endpoint"])))
    hashes=load(run/"sha256.json",{})
    mismatches=[]
    for key,expected in hashes.items():
        path=run/"source_snapshot"/key[7:] if key.startswith("source:") else run/key
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            mismatches.append(key)
    frames=load(run/"video_frames.json",[])
    movie=None
    if frames:
        cap=cv2.VideoCapture(str(run/"reviewer_view.mp4"))
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps=cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        movie=dict(frames=count,manifest_frames=len(frames),fps=fps,
            duration_s=count/fps if fps else None,
            continuous=bool(count==len(frames) and np.all(np.diff([f["sim_step"] for f in frames])==4)))
    passed=bool(attempt["status"]=="completed" and scores.get("evaluator_task_success",False)
        and scores.get("evaluator_initial_target_observed",False) and scores.get("evaluator_disturbance_valid",False))
    if passed:
        reason="Pass"
    elif attempt["status"]=="timeout":
        reason="Host wall-time limit"
    elif not scores.get("evaluator_initial_target_observed",False):
        reason="Initial target not observed"
    elif not scores.get("evaluator_disturbance_valid",False):
        reason="Disturbance precondition failed"
    elif attempt["status"]!="completed":
        reason="Execution/controller exception"
    elif not result.get("visual_approach_complete",False):
        reason="Search/approach incomplete"
    elif scores.get("evaluator_sustained_lift_s",0)<1.:
        reason="Grasp/lift incomplete"
    else:
        reason="Carry/place incomplete"
    base=np.asarray([r["base_xyyaw"][:2] for r in actions])
    truths={row["step"]:row for row in trajectory}
    reacquired=False
    approached=False
    for event in events:
        if event["step"]<config.get("query_step",0) or event["step"] not in truths:
            continue
        truth=np.asarray(truths[event["step"]]["target_xyz"])
        if event["event"]=="observation":
            reacquired |= any(np.linalg.norm(np.asarray(d["point_world"])-truth)<.15
                for d in event.get("detections",[]))
        if event["event"]=="visual_approach_complete":
            approached |= np.linalg.norm(np.asarray(event["detection"]["point_world"])-truth)<.15
    stages=dict(reacquired=bool(reacquired),approached=bool(approached),
        lifted=scores.get("evaluator_sustained_lift_s",0)>=1.,placed=scores.get("evaluator_stable_placement",False))
    return dict(name=attempt["name"],scene=attempt["scene"],variant=attempt["variant"],
        execution_status=attempt["status"],protocol_success=passed,reason=reason,
        metrics=scores,diagnostic_stages=stages,env_steps=len(actions),
        max_base_step_m=float(np.linalg.norm(np.diff(base,axis=0),axis=1).max()) if len(base)>1 else 0.,
        action_steps_contiguous=[r["step"] for r in actions]==list(range(1,len(actions)+1)),
        trajectory_aligned=[r["step"] for r in actions]==[r["step"] for r in trajectory],
        hash_mismatches=mismatches,hash_manifest_present=bool(hashes),video=movie,
        wrist_keyframes=sum(e["event"]=="observation" and e["memory"]["sensor"]=="fetch_hand" for e in events),
        result_scores_match=all(result.get(k)==v for k,v in scores.items()) if result else None,
        events=events,frames=frames)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--manuscript",type=Path)
    args=parser.parse_args()
    suite=load(args.suite/"suite_result.json")
    if suite is None:
        raise RuntimeError("Suite is still running; do not publish a partial aggregate")
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[inspect_run(args.suite/a["name"],a) for a in suite["attempts"]]
    scenes=sorted({r["scene"] for r in reports})
    by_key={(r["scene"],r["variant"]):r for r in reports}
    pairs=[(by_key[(s,"dynamic")]["protocol_success"],by_key[(s,"static")]["protocol_success"]) for s in scenes]
    wins=sum(d and not s for d,s in pairs)
    losses=sum(s and not d for d,s in pairs)
    p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.
    counts={v:dict(scheduled=len(scenes),success=sum(r["variant"]==v and r["protocol_success"] for r in reports),
        **{stage:sum(r["variant"]==v and r["diagnostic_stages"][stage] for r in reports)
           for stage in ("reacquired","approached","lifted","placed")},
        timeouts=sum(r["variant"]==v and r["execution_status"]=="timeout" for r in reports),
        exceptions=sum(r["variant"]==v and r["execution_status"]=="execution_failure" for r in reports)) for v in ("dynamic","static")}
    compact=[{k:v for k,v in r.items() if k not in ("events","frames")} for r in reports]
    aggregate=dict(counts=counts,paired_wins=wins,paired_losses=losses,descriptive_exact_mcnemar_p=p,runs=compact,
        interpretation="Small fixed development suite; not held-out generalization. Host timeouts/exceptions remain explicit; do not attribute them unconditionally to memory quality or claim universal/statistical superiority from videos.")
    (args.output/"audit_and_results.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    # A fixed early frame for scene diversity, not a hand-picked successful pose.
    sheet=np.full((2*240,5*400,3),245,np.uint8)
    cards=[]
    markdown=["# Learned ManiSkill: ten scenes, all attempts", "",
        f"Dynamic-memory full gate: {counts['dynamic']['success']}/{len(scenes)}; accumulation-only: {counts['static']['success']}/{len(scenes)}.",
        "", "These are real executed attempts, not ten guaranteed successes. Use the gallery chapters to seek within uninterrupted videos; no steps are removed from the source recordings.", "",
        "| Video | Scene | Outcome |", "|---|---|---|"]
    rows=[]
    for index,r in enumerate(sorted((r for r in reports if r["variant"]=="dynamic"),key=lambda r:r["name"])):
        run=args.suite/r["name"]
        video_name=f"{index+1:02d}_{r['scene']}.mp4"
        source=run/"reviewer_view.mp4"
        if not source.is_file() or not r["video"] or not r["video"]["continuous"]:
            raise RuntimeError(f"Missing/noncontinuous recording: {source}")
        (args.output/video_name).symlink_to(source.resolve())
        cap=cv2.VideoCapture(str(run/"overview_raw.mp4"))
        chosen=min(r["frames"],key=lambda f:abs(f["sim_step"]-360))
        cap.set(cv2.CAP_PROP_POS_FRAMES,chosen["frame"])
        ok,frame=cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot extract fixed source frame: {run}")
        frame=cv2.resize(frame,(400,225))
        tile=np.full((240,400,3),245,np.uint8)
        tile[:225]=frame
        label=f"{index+1:02d} {r['scene'].replace('ArchitecTHOR-','')} | {'PASS' if r['protocol_success'] else 'NOT COMPLETED'}"
        cv2.putText(tile,label,(6,237),cv2.FONT_HERSHEY_SIMPLEX,.38,(35,45,50),1,cv2.LINE_AA)
        sheet[(index//5)*240:(index//5+1)*240,(index%5)*400:(index%5+1)*400]=tile
        poster=f"{index+1:02d}_scene.jpg"
        cv2.imwrite(str(args.output/poster),frame)
        chapters=[("Initial scan",0.)]
        labels={"memory_retrieval":"Request / retrieval","visual_approach_complete":"Target approached",
            "depth_grasp_center":"Grasp","lift_attempt_complete":"Lift / carry","placement_tcp_approach":"Place"}
        seen=set()
        for event in r["events"]:
            name=event["event"]
            if name in labels and name not in seen:
                chapters.append((labels[name],max(0.,event["step"]/20.-1.)))
                seen.add(name)
        buttons=" ".join(f'<button onclick="document.getElementById(\'v{index}\').currentTime={time:.2f};document.getElementById(\'v{index}\').play()">{html.escape(label)}</button>' for label,time in chapters)
        cards.append(f'<article><h2>{index+1:02d} · {html.escape(r["scene"])} <small>{html.escape(r["reason"])}</small></h2><video id="v{index}" controls preload="none" poster="{poster}" src="{video_name}"></video><p>{buttons}</p></article>')
        markdown.append(f"| [{index+1:02d}]({video_name}) | {r['scene']} | {r['reason']} |")
        static=by_key[(r["scene"],"static")]
        rows.append(f"{index+1} & {r['scene'].replace('ArchitecTHOR-','')} & {'Yes' if r['protocol_success'] else 'No'} & {'Yes' if static['protocol_success'] else 'No'} & {r['metrics'].get('evaluator_base_travel_m',0):.2f} & {static['metrics'].get('evaluator_base_travel_m',0):.2f} & {r['reason']} " + chr(92)*2)
    cv2.imwrite(str(args.output/"learned_scene_contact_sheet.png"),sheet)
    gallery='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Learned DREAM evidence</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#223039;background:#f5f7f8}article{background:white;padding:18px;margin:24px 0;border-radius:8px}video{width:100%;display:block}h2{font-size:20px}small{font-size:14px;font-weight:400;margin-left:12px}button{margin:4px;padding:8px;cursor:pointer}p{line-height:1.6}</style><h1>Learned DREAM / ManiSkill</h1><p>Ten different furnished scenes; all attempts retained. These recordings are continuous at 1× simulation time. Chapter buttons only seek the original video. Success/failure comes from independent physical checks, not captions. Known simulator odometry and a round-object grasp adapter are disclosed in the README.</p>'+"\n".join(cards)+'</html>'
    (args.output/"index.html").write_text(gallery)
    (args.output/"README.md").write_text("\n".join(markdown)+"\n\nSee `audit_and_results.json` for every dynamic/static outcome, setup validity, timestamps, hashes and video checks. No submission-readiness claim is implied by a successful render.\n")
    if args.manuscript:
        shutil.copy2(args.output/"learned_scene_contact_sheet.png",args.manuscript/"figs/learned_scene_contact_sheet.png")
        text=("\\latestchange{Across the ten scheduled scenes, the complete dynamic-task gate is met in "
            f"{counts['dynamic']['success']}/10 dynamic-memory and {counts['static']['success']}/10 accumulation-only attempts. "
            f"As a diagnostic stage breakdown, verified current-view reacquisition occurs in {counts['dynamic']['reacquired']}/10 and {counts['static']['reacquired']}/10, respectively; this does not replace the complete-task gate. "
            f"There are {counts['dynamic']['timeouts']} and {counts['static']['timeouts']} host wall-time terminations, respectively; these remain separately identified in the all-attempts audit. "
            "These descriptive development results and the accompanying failure cases do not establish general superiority. Per-scene stages, trajectories and physical checks are reported in the response and released logs.}\n")
        (args.manuscript/"learned_suite_summary.tex").write_text(text)
        table="\\begin{table}[H]\n\\centering\n\\caption{\\latestchange{All ten preselected scenes. D: dynamic maintenance; A: accumulation-only. Success is the complete dynamic-task gate, not just grasping. Exceptions/timeouts and all stage metrics are retained in the released audit.}}\n\\label{tab:maniskill_runs}\n\\begingroup\\color{red}\\scriptsize\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{clccrrl}\n\\toprule\nVideo & Scene & D success & A success & D travel (m) & A travel (m) & D outcome \\\\\n\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}}\n\\endgroup\n\\end{table}\n"
        (args.manuscript/"learned_suite_rows.tex").write_text(table)
    print(json.dumps(dict(counts=counts,output=str(args.output)),indent=2))


if __name__=="__main__":
    main()
