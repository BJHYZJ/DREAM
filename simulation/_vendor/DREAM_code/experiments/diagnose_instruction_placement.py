"""Local recovery diagnostic AFTER exact full saved-control physics replay.

This is deliberately not a complete instruction-policy execution. Only two past
RGB-D destination sightings are reconstructed, not the full original memory.
No navigation is requested. New actual head/body observations must ground the
destination before the ordinary physical placement controller may release.
Actor handles below are exclusively for evaluator checks, never policy inputs.
"""
import json
from pathlib import Path
import time

import numpy as np


def validate_prefix(report):
    for key in ("physical_reexecution_passed","source_recording_unchanged",
                "replay_environment_source_matches_recording","replay_source_unchanged"):
        if report.get(key) is not True:
            raise ValueError(f"Placement diagnostic requires a verified prefix: {key}")
    errors=report.get("maximum_error",{})
    if not errors or not all(np.isfinite(v) and v<=report["tolerance"] for v in errors.values()):
        raise ValueError("Invalid or non-exact prefix state errors")


def prior_destination_events(events,step):
    """Select past learned sightings only, never evaluator destination data."""
    selected={}
    for event in events:
        if event.get("event")=="placement_observed_during_initial_search" and event["step"]<=step:
            selected[event["detection"]["observation_id"]]=event
    result=sorted(selected.values(),key=lambda e:e["step"])[-2:]
    if not result:
        raise ValueError("No prior learned destination sighting for local diagnostic")
    return result


def placement_start_step(prefix,source,navigation_record=None):
    """Validate an optional actual local-navigation chain; never relabel its prefix.

    The simulator is already at the end of those controls. This helper returns
    the true continued step, not an edited copy of the original replay audit.
    """
    start=int(prefix["steps"])
    if navigation_record is None:return start,[]
    from maniskill_learned_probe import sha
    directory=Path(navigation_record)
    config=json.loads((directory/"configuration.json").read_text())
    result=json.loads((directory/"result.json").read_text())
    controls=json.loads((directory/"actions.json").read_text())
    if Path(config["prefix_run"]).resolve()!=Path(source).resolve() or config["prefix_steps"]!=start:
        raise ValueError("Local navigation belongs to a different replay prefix")
    if not all(result.get(k) is True for k in
               ("navigation_diagnostic_passed","source_files_unchanged","source_inputs_unchanged")):
        raise ValueError("Only an independently guarded successful local navigation may precede placement")
    if not all(sha(Path(p))==expected for p,expected in config["inputs_sha256"].items()):
        raise ValueError("Local navigation inputs changed")
    if (not controls or len(controls)!=result["additional_controls"]
            or any(c["step"]!=start+i+1 for i,c in enumerate(controls))
            or result["final_step"]!=start+len(controls)):
        raise ValueError("Local navigation control chain is incomplete or misaligned")
    return result["final_step"],[directory/name for name in
        ("configuration.json","result.json","actions.json","contact_audit.json")]


def diagnostic_outcome(rows,hz,motion_completed):
    consecutive=best=0
    for row in rows:
        stable=(row["inside_receptacle"] and row["receptacle_contact_force_n"]>.01
                and not row["bilateral_contact"]
                and np.linalg.norm(row["target_velocity"])<.025)
        consecutive=consecutive+1 if stable else 0
        best=max(best,consecutive)
    return dict(placement_motion_completed=bool(motion_completed),
                diagnostic_stable_placement_s=best/hz,
                diagnostic_placement_passed=bool(motion_completed and best/hz>=2.),
                complete_instruction_task_success=False,release_ready=False,
                boundary="LOCAL CONTROLLER DIAGNOSTIC, not a full policy trial or release video")


def run_diagnostic(io,source,output,prefix,instruction,target,destination,
                   pickup_spec,receptacle_spec,step_budget=2400,navigation_record=None):
    from dream_learned_core import LearnedPerception,Detection,RGBDObservation
    from instruction_policy import InstructionSearchPilot
    from instruction_evaluation import placement_geometry
    from learned_video import EvidenceVideo
    from maniskill_learned_probe import array,sha,ROOT
    from physics_contact_audit import ContactAudit

    validate_prefix(prefix)
    source=Path(source);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    start_step,chain_inputs=placement_start_step(prefix,source,navigation_record)
    if navigation_record is not None:
        nav_result=json.loads((Path(navigation_record)/"result.json").read_text())
        if io.step_id!=start_step or not np.allclose(io.pose(),nav_result["final_base_xyyaw"],atol=1e-5,rtol=0):
            raise ValueError("Live robot does not match the recorded end of local navigation")
    events=[json.loads(line) for line in (source/"events.jsonl").read_text().splitlines()]
    sightings=prior_destination_events(events,start_step)
    fit=next(e for e in reversed(events) if e["event"]=="depth_grasp_center" and e["step"]<=start_step)
    inputs=[source/name for name in ("events.jsonl","actions.json","environment_task.json")]
    inputs.extend(chain_inputs)
    inputs.extend(source/f"observation_{e['detection']['observation_id']:05d}.npz" for e in sightings)
    source_files=[*Path(__file__).parent.glob("*.py"),*(ROOT/"DREAM_code/src/dream").rglob("*.py")]
    before={str(p):sha(p) for p in source_files}
    configuration=dict(entrypoint="diagnose_instruction_placement.py",prefix_run=str(source.resolve()),
        prefix_steps=prefix["steps"],local_placement_start_step=start_step,
        preceding_navigation_record=str(Path(navigation_record).resolve()) if navigation_record else None,
        maximum_additional_steps=step_budget,
        memory_reconstruction="Only two past learned RGB-D destination sightings; NOT the full original memory",
        navigation_requested=False,inputs_sha256={str(p):sha(p) for p in inputs},
        policy_inputs="Language, past/current RGB-D, proprioception. Actor handles evaluator-only.",
        complete_instruction_task_success=False,release_ready=False)
    (output/"configuration.json").write_text(json.dumps(configuration,indent=2)+"\n")
    (output/"source_hashes_before.json").write_text(json.dumps(before,indent=2)+"\n")
    policy=None;video=None;rows=[];completed=False;started=time.monotonic();error=None
    contact=ContactAudit(io.base,target_name=target.name,fixture_names={destination.name})
    try:
        if not bool(io.robot.is_grasping(target)[0]):
            raise ValueError("Exact replay ended without physical target retention; no placement diagnostic")
        io.step_id=start_step
        io.frame_id=max(io.frame_id,max(int(p.stem.rsplit("_",1)[1]) for pattern in
                        ("observation_*.npz","navigation_depth_*.npz") for p in source.glob(pattern)))
        # Synchronize controller targets to current measured joints, not actor
        # state. The original initialization grasp seed is retained unchanged.
        q=array(io.robot.robot.get_qpos())[0]
        io.arm=q[[5,7,8,9,10,11,12]].copy();io.body=q[[4,6,3]].copy();io.grip=-1.
        io.trace=[];io.review_sensor="fetch_head";io.display_sensor="fetch_head"
        io.stabilize_stationary_base=True;io.ceiling_safe_overview=True
        io.hold_measured_arm()
        perception=LearnedPerception(threshold=.15,model_set="production")
        policy=InstructionSearchPilot(io,perception,output,instruction,dynamic=True)
        policy.grasp_motion_attempted=True
        policy.perceived_payload_radius=float(fit["radius_m"])
        policy.perceived_payload_height=float(fit.get("height_m",2*fit["radius_m"]))
        for event in sightings:
            path=source/f"observation_{event['detection']['observation_id']:05d}.npz"
            with np.load(path,allow_pickle=False) as saved:
                values={key:saved[key] for key in saved.files}
            for key in ("frame_id","sim_step"):values[key]=int(values[key])
            values["sensor"]=str(values["sensor"])
            if values["sim_step"]>start_step:
                raise ValueError("Future observation cannot seed a diagnostic")
            observation=RGBDObservation(**values)
            observation.save(output/path.name)
            policy.memory.integrate(observation)
            policy.placement_sightings.append(Detection(**event["detection"]))
        video=EvidenceVideo(output,hz=io.base.control_freq)
        def before_step():
            if io.step_id-start_step>=step_budget:
                raise TimeoutError("Declared local placement diagnostic budget exceeded")
            contact.current_step=io.step_id+1;contact.phase=policy.phase
        def after_step():
            force=io.base.scene.get_pairwise_contact_forces(target,destination)
            rows.append(dict(step=io.step_id,task_stage=policy.task_stage,phase=policy.phase,
                base_xyyaw=io.pose().tolist(),target_xyz=array(target.pose.p)[0].tolist(),
                target_velocity=array(target.get_linear_velocity())[0].tolist(),
                bilateral_contact=bool(io.robot.is_grasping(target)[0]),
                receptacle_contact_force_n=float(np.linalg.norm(array(force))),
                inside_receptacle=placement_geometry(target,destination,pickup_spec,receptacle_spec,
                                                     instruction.placement_relation)))
            video.capture(io,policy.occupancy,"DIAGNOSTIC: "+policy.phase,memory=policy.memory,
                          query=policy.query,cached=policy.cached,
                          disturbance_label="Local recovery test; not a full-task result")
        io.before_step=before_step;io.after_step=after_step
        policy.start_placement_search()
        policy.event("local_diagnostic_started",partial_memory=True,navigation_requested=False)
        policy.focus()
        completed=policy.place_observed() if policy.current_detection is not None else False
        for _ in range(40):io.command()
    except Exception as caught:
        error=repr(caught)
    finally:
        io.before_step=None;io.after_step=None
        if video:video.close()
        outcome=diagnostic_outcome(rows,io.base.control_freq,completed)
        outcome.update(error=error,additional_steps=len(rows),wall_time_s=time.monotonic()-started,
                       source_files_unchanged=all(sha(p)==before[str(p)] for p in source_files),
                       contact_audit=contact.report())
        for name,value in (("result.json",outcome),("actions.json",io.trace),
                           ("evaluator_trajectory.json",rows)):
            (output/name).write_text(json.dumps(value,indent=2)+"\n")
        print(json.dumps(outcome,indent=2),flush=True)
    return outcome
