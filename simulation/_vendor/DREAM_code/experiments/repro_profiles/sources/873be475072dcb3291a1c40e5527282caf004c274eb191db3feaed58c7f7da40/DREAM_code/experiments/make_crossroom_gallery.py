#!/usr/bin/env python3
"""Publish independently checked, explicitly curated cross-room videos.

Default requires ten distinct successful houses. --preview is a clearly
labelled partial development preview, never a completed ten-video delivery.
"""
import argparse
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from audit_crossroom_attempts import summarize
from finalize_learned_suite import inspect_run
from crossroom_evaluation import evaluate_room_transitions


def audit(run):
    result=inspect_run(run,dict(name=run.name,scene=json.loads((run/"configuration.json").read_text())["scene"],
                                variant="dynamic",status="completed"))
    config=json.loads((run/"configuration.json").read_text())
    task=json.loads((run/"environment_task.json").read_text())
    actions=json.loads((run/"actions.json").read_text())
    with np.load(run/"evaluator_room_map.npz") as data:
        rooms=evaluate_room_transitions(actions,result["events"],task,{k:data[k] for k in data.files},config["query_step"])
        area=float(data["stage_free"].sum()*float(data["resolution"])**2)
        count=int(data["room_cores"].max())
    wrist_scans_requested=config.get("wrist_scans",True)
    sensing_matches=(result["wrist_keyframes"]>0 if wrist_scans_requested else
        not any(e["event"]=="arm_camera_scan_complete" for e in result["events"]) and
        result["wrist_keyframes"]==0 and
        any(e["event"]=="head_camera_scan_complete" for e in result["events"]))
    checks=dict(physical_and_setup=result["protocol_success"],cross_room=rooms["evaluator_cross_room_success"],
        aligned_controls=result["action_steps_contiguous"] and result["trajectory_aligned"],
        provenance=result["hash_manifest_present"] and not result["hash_mismatches"],
        continuous_video=result["video"] is not None and result["video"]["continuous"],
        scores_match=result["result_scores_match"],correct_fresh_target=result["diagnostic_stages"]["approached"],
        requested_navigation_sensing=sensing_matches)
    if not all(checks.values()):
        raise RuntimeError(f"Selected video fails audit: {run}: {checks}")
    return dict(source_run=str(run.resolve()),scene=config["scene"],checks=checks,
        metrics=result["metrics"],room_metrics=rooms,house_stage_floor_area_m2=area,
        house_room_core_count=count,video=result["video"],wrist_keyframes=result["wrist_keyframes"],
        max_base_step_m=result["max_base_step_m"],events=result["events"],frames=result["frames"],
        source_generation=run.parent.name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--preview",action="store_true")
    parser.add_argument("--audit-reports",type=Path,nargs="+",
                        help="Completed audit_crossroom_success.py audit.json reports; required for a final release")
    args=parser.parse_args()
    houses=[summarize(run)["scene"] for run in args.runs]
    if len(houses)!=len(set(houses)):
        raise ValueError("Distinct houses required; repeated layouts do not increase the house count")
    if not args.preview and len(houses)!=10:
        raise ValueError("Final release requires exactly ten independently successful houses")
    audits={}
    for path in args.audit_reports or []:
        report=json.loads(path.read_text())
        if not report.get("passed",False):
            raise ValueError(f"Failed independent physical audit: {path}")
        audits[report["source_run"]]=(path,report)
    if not args.preview and any(str(run.resolve()) not in audits for run in args.runs):
        raise ValueError("Each final video needs independent physical control/contact audit")
    if not args.preview and any(not audits[str(run.resolve())][1].get("batch_provenance",{}).get("passed",False) for run in args.runs):
        raise ValueError("Each final video needs checked pre/post frozen-source and task provenance")
    reports=[audit(run) for run in args.runs]
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.ledger,args.output/"development_attempt_ledger.json")
    cards=[]
    markdown=["# 跨房间成功视频"+("：部分预览，尚未完成十条" if args.preview else "：十条精选成功案例"),
        "", "这里只收录通过完整物理任务与跨房间检查的实际运行。它们是开发后的精选成功展示，不能当作无偏 100% 成功率。",
        "", "[打开视频网页](index.html)；全部尝试见 [开发记录](development_attempt_ledger.json)。"]
    for index,(run,report) in enumerate(zip(args.runs,reports),1):
        folder=args.output/f"{index:02d}_{report['scene']}"
        folder.mkdir()
        shutil.copy2(run/"reviewer_view.mp4",folder/"reviewer_view.mp4")
        for name in ("result.json","configuration.json","environment_task.json","video_frames.json","events.jsonl"):
            shutil.copy2(run/name,folder/name)
        if str(run.resolve()) in audits:
            audit_path,physical=audits[str(run.resolve())]
            shutil.copy2(audit_path,folder/"independent_physical_audit.json")
            report["independent_physical_checks"]=physical["checks"]
            report["batch_provenance"]=physical.get("batch_provenance")
        movie=cv2.VideoCapture(str(folder/"reviewer_view.mp4"))
        count=int(movie.get(cv2.CAP_PROP_FRAME_COUNT))
        for name,frame in (("start",0),("finished",max(0,count-8))):
            movie.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,pixels=movie.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {name} frame in {run}")
            cv2.imwrite(str(folder/f"{name}.png"),pixels)
        movie.release()
        chapter_events={"focused_verification":"Verify memory","visual_approach_complete":"Found in next room",
            "depth_grasp_center":"Grasp","held_arm_joint_fold":"Carry back","placement_tcp_approach":"Place"}
        chapters=[("Build memory",0.),("Target relocation",json.loads((run/"configuration.json").read_text())["disturbance_step"]/20.)]
        for event,label in chapter_events.items():
            found=next((e for e in report["events"] if e["event"]==event),None)
            if found:
                chapters.append((label,max(0.,found["step"]/20.-2.)))
        buttons="".join(f'<button onclick="document.getElementById(\'v{index}\').currentTime={when:.1f}">{html.escape(label)}</button>' for label,when in chapters)
        prefix=folder.name
        metric=report["metrics"]
        cards.append(f'<article><h2>{index:02d} · {html.escape(report["scene"])}</h2>'
            f'<p>Cross-room search → physical grasp → return → stable placement · base travel {metric["evaluator_base_travel_m"]:.2f} m</p>'
            f'<video id="v{index}" controls preload="none" poster="{prefix}/finished.png" src="{prefix}/reviewer_view.mp4"></video>'
            f'<div>{buttons}</div><p><a href="{prefix}/result.json">Task checks</a> · <a href="{prefix}/events.jsonl">Recorded events</a>'
            +(f' · <a href="{prefix}/independent_physical_audit.json">Independent replay/contact audit</a>' if str(run.resolve()) in audits else '')+'</p></article>')
        markdown.extend(["",f"{index}. [{report['scene']} 完整视频]({prefix}/reviewer_view.mp4)：底盘路径 {metric['evaluator_base_travel_m']:.2f} m，跨房间与稳定放置检查均通过。"])
        report["chapters"]=chapters
        report.pop("events")
        report.pop("frames")
    title=f"Cross-room DREAM: {len(reports)} curated successful demonstration"+("s" if len(reports)>1 else "")
    boundary=("Partial development preview — the requested ten-video set is not complete." if args.preview else
              "Ten curated successful demonstrations after development; not an unbiased success-rate benchmark.")
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+'</title>'
    page+='<style>body{font:16px system-ui;max-width:1150px;margin:28px auto;padding:0 20px;background:#f4f6f7;color:#223039}article{background:white;padding:18px;margin:24px 0;border-radius:8px}video{width:100%}button{margin:4px;padding:8px;cursor:pointer}p{line-height:1.6}</style>'
    page+=f'<h1>{html.escape(title)}</h1><p>{boundary}</p><p>Continuous 1× simulation time. Chapter buttons only seek the original video. Green: A* plan / goal; blue: executed path; orange: measured text-feature alignment. Room maps and target truth are evaluator-only. See the <a href="development_attempt_ledger.json">all-attempt development ledger</a>.</p>'+"".join(cards)
    (args.output/"index.html").write_text(page+"\n")
    (args.output/"README.md").write_text("\n".join(markdown)+"\n")
    (args.output/"audited_successes.json").write_text(json.dumps(dict(partial_preview=args.preview,
        boundary=boundary,distinct_successful_houses=len(reports),cases=reports),indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output.resolve()),distinct_successful_houses=len(reports),partial_preview=args.preview)))


if __name__=="__main__":
    main()
