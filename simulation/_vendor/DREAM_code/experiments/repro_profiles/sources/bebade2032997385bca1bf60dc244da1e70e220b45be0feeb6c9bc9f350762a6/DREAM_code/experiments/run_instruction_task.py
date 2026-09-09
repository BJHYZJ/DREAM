#!/usr/bin/env python3
"""Execute a real instruction-first learned DREAM/Fetch task; success is audited.

This development runner does not freeze a benchmark by itself. Its task actors,
room raster and event controller are environment/evaluator state only. The
search policy receives a language contract and observed robot data.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

from maniskill_learned_probe import ROOT,SimulatorIO,array,make_env,initialize_compact_arm,LearnedPerception,sha
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

from instruction_task import parse_instruction
from instruction_policy import InstructionSearchPilot
from instruction_environment import create_fixture,sample_start,DiscoveryTriggeredMove,exclude_initial_categories
from instruction_evaluation import evaluate_instruction,placement_geometry
from learned_video import EvidenceVideo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--variant",choices=("dynamic","static"),default="dynamic")
    parser.add_argument("--threshold",type=float,default=.15)
    parser.add_argument("--navigation-budget",type=int,default=110)
    parser.add_argument("--video",action="store_true")
    args=parser.parse_args()
    task=json.loads(args.task_json.read_text())
    instruction=parse_instruction(task["instruction"])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"environment_task.json").write_text(json.dumps(task,indent=2)+"\n")
    (args.output/"configuration.json").write_text(json.dumps(dict(
        variant=args.variant,threshold=args.threshold,navigation_budget=args.navigation_budget,
        video=args.video,entrypoint="run_instruction_task.py",status="development_not_frozen_benchmark"),indent=2)+"\n")
    room_path=args.task_json.parent/task["room_map_file"]
    shutil.copy2(room_path,args.output/"evaluator_room_map.npz")
    with np.load(room_path) as data:
        room_map={key:data[key] for key in data.files}
    sources=list(Path(__file__).parent.glob("*.py"))+list((ROOT/"DREAM_code/src/dream").rglob("*.py"))
    hashes={str(p.relative_to(ROOT/"DREAM_code")):sha(p) for p in sources}
    for source in sources:
        relative=source.relative_to(ROOT/"DREAM_code")
        destination=args.output/"source_snapshot"/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    (args.output/"source_hashes_before.json").write_text(json.dumps(hashes,indent=2)+"\n")
    env=None; io=None; policy=None; video=None; move=None; rows=[]; result={}
    trajectory_stream=(args.output/"evaluator_trajectory.jsonl").open("x",buffering=1)
    started=time.monotonic()
    torch.set_num_threads(2)
    try:
        perception=LearnedPerception(threshold=args.threshold,model_set="production")
        env=make_env(task["scene"],width=640,height=480)
        env.reset(seed=task["seed"])
        base=env.unwrapped
        spawn,yaw,sampling=sample_start(room_map,task["initial_room"],task["seed"],
            [(task["placement_table_xy"],task["placement_table_collision_radius_m"])],
            target_xy=task["target_xy"],docking_xy=task["endpoint"])
        base.agent.robot.set_pose(Pose.create_from_pq(p=[[*spawn,.02]]))
        q=base.agent.robot.get_qpos().clone(); q[:,2]=yaw
        base.agent.robot.set_qpos(q); base.agent.controller.reset()
        io=SimulatorIO(env)
        io.review_sensor="fetch_head"
        initialize_compact_arm(base)
        io.arm=array(io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].copy()
        io.stabilize_stationary_base=True; io.ceiling_safe_overview=True
        categories={task["recipe"]["environment_assets"][role].rsplit("_",1)[0]
                    for role in ("pickup","placement")}
        excluded=exclude_initial_categories(base,categories)
        directory=ROOT/".maniskill_assets/data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects"
        cart,target,receptacle,pickup_spec,receptacle_spec,metadata=create_fixture(base,directory,task)
        (args.output/"initial_conditions.json").write_text(json.dumps(dict(spawn_xy=spawn.tolist(),yaw_rad=yaw,
            sampling=sampling,fixtures=metadata,excluded_initial_actors=excluded,
            boundary="Environment/evaluator only. All pose initialization precedes control step one."),indent=2)+"\n")
        policy=InstructionSearchPilot(io,perception,args.output,instruction,dynamic=args.variant=="dynamic")
        move=DiscoveryTriggeredMove(io,cart,target,task["endpoint"],task["disturbance_waypoints"],
                                    query=instruction.pickup_query,threshold=args.threshold,
                                    evaluation_output=args.output/"evaluator_visibility")
        policy.observation_listener=move.observed
        maximum_steps=int(task.get("maximum_sim_seconds",1200)*base.control_freq)
        def before():
            if io.step_id>=maximum_steps:
                raise TimeoutError("Declared simulation-time budget exceeded")
            move.before_step(policy.task_stage)
        io.before_step=before
        if args.video:
            video=EvidenceVideo(args.output,hz=base.control_freq)
        def after():
            force=base.scene.get_pairwise_contact_forces(target,receptacle)
            row=dict(step=io.step_id,task_stage=policy.task_stage,phase=policy.phase,
                base_xyyaw=io.pose().tolist(),target_xyz=array(target.pose.p)[0].tolist(),
                target_velocity=array(target.get_linear_velocity())[0].tolist(),
                bilateral_contact=bool(base.agent.is_grasping(target)[0]),
                receptacle_contact_force_n=float(np.linalg.norm(array(force))),
                inside_receptacle=placement_geometry(target,receptacle,pickup_spec,receptacle_spec,instruction.placement_relation))
            rows.append(row)
            trajectory_stream.write(json.dumps(row)+"\n")
            if video:
                caption=[]
                if move.gate.started_step is not None and move.done_step is None:
                    caption.append("External: target/support moving")
                if policy.last_memory_change and io.step_id-policy.last_memory_change["step"]<100:
                    caption.append("Observed: memory updated" if policy.memory.dynamic else
                                   "Observed: change; static memory retained")
                video.capture(io,policy.occupancy,policy.phase,memory=policy.memory,query=policy.query,
                    cached=policy.cached,planned_path=policy.planned_path,waypoint=policy.waypoint,
                    navigation_goal=policy.navigation_goal,disturbance_label=caption)
        io.after_step=after
        policy.survey()
        approached=False
        # Target motion is judged only by repeated visual observations. The
        # policy is not told whether the environment controller reached its end.
        for attempt in range(20):
            if not policy.navigate(budget=args.navigation_budget):
                break
            first=policy.current_detection
            for _ in range(20): io.command()
            second=policy.observe()
            stable=first is not None and second is not None and np.linalg.norm(
                np.asarray(first.point_world)-np.asarray(second.point_world))<.025
            policy.event("pregrasp_visual_motion_check",stationary=bool(stable),
                first=asdict(first) if first else None,second=asdict(second) if second else None)
            if stable:
                approached=True
                break
        result["pickup_approached"]=approached
        held=policy.attempt_grasp() if approached else False
        result["grasp_proprioception_retained"]=held
        if held:
            policy.prepare_carry()
            policy.start_placement_search()
            placement_approach=policy.navigate(budget=args.navigation_budget)
            result["placement_approached"]=placement_approach
            result["placement_motion_completed"]=policy.place_observed() if placement_approach else False
        for _ in range(40): io.command()
    except Exception as error:
        result["error"]=repr(error)
        (args.output/"failure.json").write_text(json.dumps(dict(error=repr(error)),indent=2)+"\n")
    finally:
        trajectory_stream.close()
        if video: video.close()
        if io:
            (args.output/"actions.json").write_text(json.dumps(io.trace)+"\n")
        if policy:
            (args.output/"memory_updates.json").write_text(json.dumps(policy.memory.updates,indent=2)+"\n")
            np.savez_compressed(args.output/"observed_occupancy.npz",known=policy.occupancy.known,
                last_seen=policy.occupancy.last_seen,origin=policy.occupancy.origin,
                resolution=policy.occupancy.resolution,radius=policy.occupancy.radius)
        if move:
            (args.output/"disturbance_forces.json").write_text(json.dumps(move.trace)+"\n")
            (args.output/"evaluator_discoveries.json").write_text(json.dumps(move.observation_checks)+"\n")
            try:
                result.update(evaluate_instruction(rows,policy.events,io.trace,move,room_map,hz=io.base.control_freq))
            except Exception as error:
                result.update(evaluator_protocol_success=False,evaluator_error=repr(error))
        else:
            result["evaluator_protocol_success"]=False
        (args.output/"evaluator_trajectory.json").write_text(json.dumps(rows)+"\n")
        post={str(p.relative_to(ROOT/"DREAM_code")):sha(p) for p in sources}
        result.update(wall_time_s=time.monotonic()-started,status="finished_development_attempt",
                      source_files_unchanged=post==hashes,release_ready=False)
        (args.output/"source_hashes_after.json").write_text(json.dumps(post,indent=2)+"\n")
        (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2),flush=True)
        if env: env.close()


if __name__=="__main__":
    main()
