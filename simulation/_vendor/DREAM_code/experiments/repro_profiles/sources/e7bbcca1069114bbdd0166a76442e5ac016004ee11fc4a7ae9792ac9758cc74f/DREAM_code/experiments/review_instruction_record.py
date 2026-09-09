#!/usr/bin/env python3
"""Audit recorded observation/identity/video provenance and add logged captions.

No simulation is run here. Annotated footage keeps every source frame at 1x;
captions are derived from saved events, never supplied to the acting policy.
Original recordings remain untouched. This is not an extra algorithm trial.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from dream_learned_core import RGBDObservation
from instruction_geometry import point_observed_empty
from instruction_evaluation import chronological_reacquisition
from instruction_task import parse_instruction
from learned_video import letterbox


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def observation(path):
    with np.load(path,allow_pickle=False) as saved:
        values={key:saved[key] for key in saved.files}
    for key in ("frame_id","sim_step"):values[key]=int(values[key])
    values["sensor"]=str(values["sensor"])
    return RGBDObservation(**values)


def bbox_iou(mask,box):
    pixels=np.argwhere(mask)
    if not len(pixels):return 0.
    lo=pixels.min(0)[::-1];hi=pixels.max(0)[::-1]+1
    box=np.asarray(box)
    overlap=np.maximum(0,np.minimum(hi,box[2:])-np.maximum(lo,box[:2])).prod()
    union=(hi-lo).prod()+np.maximum(0,box[2:]-box[:2]).prod()-overlap
    return float(overlap/max(1.,union))


def logged_captions(step,result,events,*,hz=20):
    captions=[]
    start,end=result.get("relocation_start_step"),result.get("relocation_done_step")
    if start is not None and start<step and (end is None or step<=end):
        captions.append(("External: target/support moving",start))
    changes=[e for e in events if e["event"]=="observed_memory_update" and
             e.get("task_stage")=="pickup_search" and 0<=step-e["step"]<3*hz]
    if changes:
        event=changes[-1]
        if event.get("new_point") is None:
            label=("Observed: target missing; memory updated" if event.get("dynamic_updates_enabled") else
                   "Observed: target missing; static memory kept")
        else:
            label=("Observed: memory updated" if event.get("dynamic_updates_enabled") else
                   "Observed: change; static memory retained")
        captions.append((label,event["step"]))
    reacquired=result.get("reacquisition_step")
    if reacquired is not None and 0<=step-reacquired<3*hz:
        captions.append(("Observed: target found again",reacquired))
    return captions


def audit_record(run,physical):
    result=json.loads((run/"result.json").read_text())
    task=json.loads((run/"environment_task.json").read_text())
    config=json.loads((run/"configuration.json").read_text())
    events=[json.loads(line) for line in (run/"events.jsonl").read_text().splitlines()]
    rows=json.loads((run/"evaluator_trajectory.json").read_text())
    actions=json.loads((run/"actions.json").read_text())
    before=json.loads((run/"source_hashes_before.json").read_text())
    after=json.loads((run/"source_hashes_after.json").read_text())
    p=json.loads(physical.read_text())
    checks=dict(protocol_scoring=bool(result.get("evaluator_protocol_success")),
        source_hashes_match=before==after,
        saved_source_matches_hashes=all(sha(run/"source_snapshot"/key)==value for key,value in before.items()),
        instruction_first=events[0].get("instruction")==parse_instruction(task["instruction"]).policy_payload()
            and events[0]["step"]==0,
        actual_control_reexecution=bool(p["physical_reexecution_passed"]),
        reexecuted_protocol_success=bool(p["evaluation"]["evaluator_protocol_success"]),
        replay_sources_verified=all(p[k] for k in ("source_recording_unchanged",
            "replay_environment_source_matches_recording","replay_source_unchanged")),
        replay_refers_to_this_run=Path(p["source_run"]).resolve()==run.resolve(),
        no_native_contact_above_threshold=p["contact_audit"]["native_environment_contact_control_steps"]==0,
        complete_aligned_traces=len(rows)==len(actions) and all(r["step"]==a["step"]==i+1
            for i,(r,a) in enumerate(zip(rows,actions))))
    detections=json.loads((run/"evaluator_discoveries.json").read_text())
    identity=[]
    for record in detections:
        obs=observation(run/f"observation_{record['observation_id']:05d}.npz")
        with np.load(run/"evaluator_visibility"/f"visibility_{record['observation_id']:05d}.npz") as data:
            mask=data["target_mask"]
        detection=record["detection"]
        truth=rows[record["step"]-1]
        error=float(np.linalg.norm(np.asarray(detection["point_world"])-truth["target_xyz"]))
        iou=bbox_iou(mask,detection["box_xyxy"])
        correct=bool(error<.15 and mask.sum()>=12 and iou>=.25)
        consistent=(record["step"]==obs.sim_step and record["observation_id"]==obs.frame_id==detection["observation_id"]
            and record["evaluator_correct_identity"]==correct
            and abs(error-record["evaluator_position_error_m"])<1e-5
            and abs(iou-record["evaluator_detection_box_iou"])<1e-5
            and int(mask.sum())==record["evaluator_visible_target_pixels"]
            and mask.shape==obs.depth_m.shape and detection["query"]==parse_instruction(task["instruction"]).pickup_query)
        identity.append(dict(step=record["step"],consistent=consistent,correct=correct,
                             score=detection["score"],observation_id=obs.frame_id))
    checks["identity_records_consistent_with_saved_masks_and_replayed_positions"]=bool(identity) and all(r["consistent"] for r in identity)
    start=result["relocation_start_step"]
    discoveries=[r for r in identity if r["correct"] and 0<=start-r["step"]<=120
                 and r["score"]>=config["threshold"]]
    if discoveries:
        event=detections[next(i for i,r in enumerate(identity) if r==discoveries[-1])]
        point=np.asarray(event["detection"]["point_world"][:2])
        base=np.asarray(rows[start-1]["base_xyyaw"][:2])
        translating=np.linalg.norm(base-np.asarray(rows[start-2]["base_xyyaw"][:2]))*20>.02
        checks["fresh_visual_discovery_at_standoff_during_translation"]=bool(np.linalg.norm(point-base)>=1 and translating)
    else:checks["fresh_visual_discovery_at_standoff_during_translation"]=False
    updates=[]
    for event in events:
        if event["event"]!="observed_memory_update" or event.get("task_stage")!="pickup_search":continue
        obs=observation(run/f"observation_{event['evidence_observation_id']:05d}.npz")
        updates.append(dict(step=event["step"],observation_id=obs.frame_id,
            measured_free_depth=point_observed_empty(obs,event["old_point"]),time_aligned=obs.sim_step==event["step"]))
    checks["old_location_update_has_measured_free_depth"]=bool(updates) and all(r["measured_free_depth"] and r["time_aligned"] for r in updates)
    initial=json.loads((run/"initial_conditions.json").read_text())["fixtures"]["target"]["initial_origin_xyz"]
    temporal=chronological_reacquisition(events,detections,start,initial)
    checks["reacquisition_follows_observed_invalidation"]=bool(temporal["strict_loss_then_reacquisition"] and
        any(r["step"]==temporal["observed_loss_step"] and r["measured_free_depth"] and r["time_aligned"]
            for r in updates))
    # Preserve original result files/verdicts. Reannotate from the explicitly
    # versioned stricter event definition, never the first moving-target frame.
    reviewed_result=dict(result,reacquisition_step=temporal["reacquisition_step"])
    return dict(checks=checks,identity_checks=identity,memory_evidence=updates,
        temporal_evidence=temporal,review_definition_version=2,
        original_reacquisition_step=result.get("reacquisition_step")),reviewed_result,task,events,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--physical-audit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--annotate",action="store_true")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report,result,task,events,rows=audit_record(args.run,args.physical_audit)
    frames=json.loads((args.run/"video_frames.json").read_text())
    report["checks"]["continuous_1x_frame_steps"]=len(frames)==len(rows)//4 and all(
        f["sim_step"]==4*(i+1) and abs(f["sim_time_s"]-f["sim_step"]/20)<1e-9
        and f["phase"]==rows[f["sim_step"]-1]["phase"] for i,f in enumerate(frames))
    captures={name:cv2.VideoCapture(str(args.run/(name+".mp4"))) for name in
              ("overview_raw","observation_raw","reviewer_view")}
    writer=None;decoded=0;errors=[];caption_records=[]
    if args.annotate:
        writer=imageio.get_writer(args.output/"reviewer_view.mp4",fps=5,codec="libx264",quality=8,
                                  macro_block_size=2,ffmpeg_log_level="error")
    try:
        for record in frames:
            images={}
            for name,capture in captures.items():
                ok,frame=capture.read()
                if not ok:raise ValueError(f"Video ended early: {name}, frame {decoded}")
                images[name]=frame
            expected=letterbox(images["observation_raw"],480,270)
            error=float(np.mean(np.abs(expected.astype(float)-images["reviewer_view"][60:330,960:].astype(float))))
            errors.append(error)
            labels=logged_captions(record["sim_step"],result,events)
            if writer:
                frame=images["reviewer_view"].copy()
                # Restore original raw overview, removing the earlier overlay
                # that accidentally hid simultaneous observed-memory events.
                frame[36:576,:960]=images["overview_raw"]
                if record["sim_step"]<=300:
                    frame[:36,:960]=(248,247,245)
                    cv2.putText(frame,task["instruction"],(14,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(53,46,32),1,cv2.LINE_AA)
                for i,(caption,event_step) in enumerate(labels):
                    top=46+34*i
                    cv2.rectangle(frame,(12,top),(650,top+30),(204,235,245),-1)
                    cv2.putText(frame,caption,(22,top+21),cv2.FONT_HERSHEY_SIMPLEX,.55,(53,46,32),1,cv2.LINE_AA)
                    caption_records.append(dict(frame=decoded,sim_step=record["sim_step"],caption=caption,event_step=event_step))
                writer.append_data(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                if labels and not (args.output/"memory_update_frame.png").exists() and any("memory updated" in c for c,_ in labels):
                    cv2.imwrite(str(args.output/"memory_update_frame.png"),frame)
            decoded+=1
        report["checks"]["all_source_videos_decode_completely"]=all(not c.read()[0] for c in captures.values())
        report["checks"]["matching_5fps_playback"]=all(abs(c.get(cv2.CAP_PROP_FPS)-5)<1e-6 for c in captures.values())
    finally:
        for capture in captures.values():capture.release()
        if writer:writer.close()
    report["checks"]["current_observation_panel_matches_raw_camera"]=bool(errors) and max(errors)<12.
    report.update(decoded_frames=decoded,maximum_observation_panel_mae=max(errors),
        source_run=str(args.run.resolve()),annotated=args.annotate,
        source_video_sha256={name:sha(args.run/(name+".mp4")) for name in captures},
        events_sha256=sha(args.run/"events.jsonl"),physical_audit_sha256=sha(args.physical_audit),
        review_script_sha256=sha(__file__),
        review_dependency_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ("instruction_evaluation.py","instruction_geometry.py","instruction_task.py")},release_ready=False,
        boundary="Recorded-data/visibility/physical-replay/video check, not an additional task. Visual inspection and final frozen-study release remain separate.")
    if args.annotate:
        report["checks"]["observed_memory_update_caption_visible"]=any("memory updated" in c["caption"] for c in caption_records)
        cap=cv2.VideoCapture(str(args.output/"reviewer_view.mp4"));n=0
        while cap.read()[0]:n+=1
        cap.release()
        report["checks"]["annotated_video_retains_all_frames"]=n==decoded
        report["annotated_video_sha256"]=sha(args.output/"reviewer_view.mp4")
    report["record_review_passed"]=all(report["checks"].values())
    (args.output/"caption_provenance.json").write_text(json.dumps(caption_records,indent=2)+"\n")
    (args.output/"record_review.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("identity_checks",)},indent=2),flush=True)
    if not report["record_review_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
