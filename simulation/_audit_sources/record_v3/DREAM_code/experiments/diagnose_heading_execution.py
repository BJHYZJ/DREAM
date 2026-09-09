"""Navigation-only diagnostic after an exact full saved-control prefix.

Uses a recorded observed map, learned memory goal and actual fresh depth hits.
The saved 2-D obstacles are conservatively retained; no full semantic-memory
resume or clearing of old 3-D evidence is claimed. Never a full task success.
"""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np


def run_diagnostic(io,source,output,prefix,route_path,target,destination,cart,step_budget=2400):
    from diagnose_instruction_placement import validate_prefix
    from dream_fetch_footprint import capture_robot_footprint
    from dream_fetch_navigation import FetchObservedMap
    from fetch_heading_control import HeadingRouteExecutor,route_chunks
    from learned_video import EvidenceVideo
    from maniskill_crossroom_policy import CrossRoomSearchPilot
    from maniskill_learned_probe import array,sha,ROOT
    from physics_contact_audit import ContactAudit

    validate_prefix(prefix)
    route_path=Path(route_path);route_report=json.loads(route_path.read_text())
    if Path(route_report["source_run"]).resolve()!=source.resolve() or not route_report["recorded_map_swept_route_found"]:
        raise ValueError("A verified offline route for this exact source is required")
    route=route_report["route_xyyaw"]
    if np.linalg.norm(np.asarray(route[0][:2])-io.pose()[:2])>1e-4:
        raise ValueError("Replay does not end at the planned measured start")
    events=[json.loads(line) for line in (source/"events.jsonl").read_text().splitlines()]
    if route_report["source_query"] not in events:
        raise ValueError("Route query is not an actual learned event from this prefix")
    query=np.asarray(route_report["source_query"]["detection"]["point_world"])
    if np.linalg.norm(np.asarray(route[-1][:2])-query[:2])>.68:
        raise ValueError("Offline route does not terminate near its learned query")
    held=[e for e in events if e["event"]=="measured_carry_envelope"]
    fits=[e for e in events if e["event"]=="depth_grasp_center"]
    last_stage=json.loads((source/"evaluator_trajectory.json").read_text())[-1]["task_stage"]
    radius=held[-1]["payload_radius_from_depth_m"] if held and last_stage=="placement_search" else None
    height=fits[-1].get("height_m") if radius and fits else None
    with np.load(source/"observed_occupancy.npz") as data:
        occupancy=FetchObservedMap(origin=data["origin"],size=len(data["known"]),
            resolution=float(data["resolution"]),radius=float(data["radius"]))
        occupancy.known=data["known"].copy();occupancy.last_seen=data["last_seen"].copy()
    if any(e["event"]=="controller_stall" for e in events):
        raise ValueError("This bounded diagnostic currently requires a prefix without omitted stall-inferred cells")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    inputs=[route_path,source/"actions.json",source/"events.jsonl",source/"observed_occupancy.npz"]
    sources=[*Path(__file__).parent.glob("*.py"),*(ROOT/"DREAM_code/src/dream").rglob("*.py")]
    hashes={str(p):sha(p) for p in sources}
    configuration=dict(prefix_run=str(source.resolve()),prefix_steps=prefix["steps"],
        inputs_sha256={str(p):sha(p) for p in inputs},maximum_additional_controls=step_budget,
        geometry_memory="Saved observed 2-D obstacles retained; fresh depth may add obstacles/free evidence, never clear old obstacles",
        semantic_memory="Only saved learned query and offline route; NOT a full policy memory resume",
        complete_instruction_task_success=False,release_ready=False)
    (output/"configuration.json").write_text(json.dumps(configuration,indent=2)+"\n")
    (output/"source_hashes_before.json").write_text(json.dumps(hashes,indent=2)+"\n")
    io.step_id=prefix["steps"];start_step=io.step_id
    io.frame_id=max(int(p.stem.rsplit("_",1)[1]) for pattern in
        ("observation_*.npz","navigation_depth_*.npz") for p in source.glob(pattern))
    q=array(io.robot.robot.get_qpos())[0]
    io.arm=q[[5,7,8,9,10,11,12]].copy();io.body=q[[4,6,3]].copy();io.grip=-1. if radius else 1.
    io.trace=[];io.review_sensor="fetch_head";io.display_sensor="fetch_head"
    io.stabilize_stationary_base=True;io.ceiling_safe_overview=True;io.hold_measured_arm()
    event_rows=[];rows=[];completed=False;error=None;video=None;started=time.monotonic()
    contact=ContactAudit(io.base,target_name=target.name,fixture_names={cart.name,destination.name,
        "instruction_placement_"+json.loads((source/"environment_task.json").read_text())["placement_table_asset"]})
    def event(name,**values):
        event_rows.append(dict(step=io.step_id,event=name,**values))
    def footprint():
        return capture_robot_footprint(io,payload_radius=radius,payload_height=height)
    def geometry_observe():
        obs=io.capture("fetch_head");obs.save(output/f"navigation_depth_{obs.frame_id:05d}.npz")
        fresh=FetchObservedMap(origin=occupancy.origin,size=len(occupancy.known),resolution=occupancy.resolution)
        spheres=CrossRoomSearchPilot.robot_filter_spheres(SimpleNamespace(io=io))
        fresh.integrate(obs,self_spheres=spheres)
        occupancy.known[(occupancy.known==0)&(fresh.known==1)]=1
        occupancy.known[fresh.known==-1]=-1
        occupancy.last_seen[fresh.known!=0]=obs.frame_id;occupancy._clearance=None
        event("fresh_conservative_depth_guard",observation_id=obs.frame_id)
    controller=HeadingRouteExecutor(io,occupancy,footprint,geometry_observe,event)
    chunks=route_chunks(route);path_xy=[p[:2] for p in chunks]
    try:
        video=EvidenceVideo(output,hz=io.base.control_freq)
        def before_step():
            if io.step_id-start_step>=step_budget:raise TimeoutError("Navigation-only diagnostic budget exceeded")
            contact.current_step=io.step_id+1;contact.phase=controller.phase
        def after_step():
            # Actor access below is independent evaluation only. No values
            # from this callback are returned to the navigation controller.
            rows.append(dict(step=io.step_id,base_xyyaw=io.pose().tolist(),
                bilateral_contact=bool(io.robot.is_grasping(target)[0]),target_xyz=array(target.pose.p)[0].tolist()))
            video.capture(io,occupancy,"LOCAL TEST: "+controller.phase,
                planned_path=path_xy,waypoint=controller.waypoint,navigation_goal=path_xy[-1],
                disturbance_label="Navigation diagnostic; not a full-task result")
        io.before_step=before_step;io.after_step=after_step
        initial=footprint();(output/"initial_robot_footprint.json").write_text(json.dumps(initial,indent=2)+"\n")
        geometry_observe()
        completed=controller.execute(route)
        for _ in range(20):io.command()
    except Exception as caught:error=repr(caught)
    finally:
        io.before_step=None;io.after_step=None
        if video:video.close()
        contact_report=contact.report();goal_error=float(np.linalg.norm(io.pose()[:2]-np.asarray(route[-1][:2])))
        retained=radius is None or (bool(rows) and all(r["bilateral_contact"] for r in rows))
        unchanged=all(sha(p)==hashes[str(p)] for p in sources)
        inputs_unchanged=all(sha(p)==configuration["inputs_sha256"][str(p)] for p in inputs)
        result=dict(navigation_motion_completed=completed,goal_error_m=goal_error,
            native_contact_control_steps=contact_report["native_environment_contact_control_steps"],
            additional_controls=len(rows),last_failure=controller.last_failure,error=error,
            held_contact_fraction=float(np.mean([r["bilateral_contact"] for r in rows])) if rows and radius else None,
            payload_retention_check_passed=retained,
            navigation_diagnostic_passed=bool(completed and goal_error<.06 and retained and unchanged
                and inputs_unchanged and not contact_report["native_environment_contact_control_steps"]),
            source_files_unchanged=unchanged,source_inputs_unchanged=inputs_unchanged,
            wall_time_s=time.monotonic()-started,complete_instruction_task_success=False,release_ready=False,
            boundary="Actual local navigation after exact replay, with conservative fresh-depth guards. Not a new full policy trial or semantic-memory resume; no grasp/place success claimed.")
        for name,value in (("result.json",result),("actions.json",io.trace),("evaluator_trajectory.json",rows),
                           ("events.json",event_rows),("contact_audit.json",contact_report)):
            (output/name).write_text(json.dumps(value,indent=2)+"\n")
        print(json.dumps(result,indent=2),flush=True)
    return result
