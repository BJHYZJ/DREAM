#!/usr/bin/env python3
"""Learned RGB-D and physical household manipulation calibration, not search.

The robot starts next to a real official table mesh. No cross-room navigation
or relocation is claimed. Geometry/grasp/place decisions receive only RGB-D
detections; independent actor contacts/poses are used exclusively for scoring.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from maniskill_learned_probe import ROOT,SimulatorIO,array,initialize_compact_arm,LearnedPerception,sha
import cv2
import numpy as np
import torch
from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.agents.robots.fetch.fetch import Fetch

from instruction_task import parse_instruction
from instruction_assets import inspect_asset,build_household_asset
from instruction_policy import InstructionSearchPilot
from instruction_geometry import tabletop_grasp,receptacle_region
from instruction_evaluation import placement_geometry
from learned_video import EvidenceVideo


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--recipe",default="potato_plate_wooden_table")
    p.add_argument("--table-asset",default=None)
    p.add_argument("--recipes-json",type=Path,default=Path(__file__).parent/"configs/instruction_pickplace_v1/recipes.json")
    p.add_argument("--threshold",type=float,default=.15,
                   help="Recorded OWL score threshold; .15 is the learned adapter's original default")
    p.add_argument("--grasp",action="store_true")
    p.add_argument("--place",action="store_true")
    p.add_argument("--video",action="store_true")
    p.add_argument("--carry-yaw",choices=("preserve","base-lateral"),default="preserve",
                   help="Explicit calibration-only gripper-yaw candidate; full-task default is unchanged")
    args=p.parse_args()
    if not 0<args.threshold<1:
        p.error("--threshold must lie between zero and one")
    if args.place and not args.grasp:
        p.error("--place requires --grasp")
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    recipes=json.loads(args.recipes_json.read_text())["recipes"]
    recipe=next(r for r in recipes if r["id"]==args.recipe)
    args.table_asset=args.table_asset or recipe["environment_assets"].get("support","Dining_Table_9_1")
    instruction=parse_instruction(recipe["instruction"])
    source_paths=[Path(__file__),*[Path(__file__).with_name(n) for n in
        ("instruction_policy.py","instruction_geometry.py","instruction_task.py","instruction_assets.py",
         "maniskill_learned_dynamic.py","maniskill_crossroom_policy.py","maniskill_learned_probe.py",
         "dream_learned_core.py","learned_video.py","instruction_evaluation.py")]]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    (args.output/"source_hashes_before.json").write_text(json.dumps(source_hashes,indent=2)+"\n")
    (args.output/"configuration.json").write_text(json.dumps(dict(recipe=recipe,table=args.table_asset,
        grasp=args.grasp,place=args.place,video=args.video,threshold=args.threshold,carry_yaw=args.carry_yaw,
        scope="Near-table learned manipulation calibration; not a cross-room or dynamic task"),indent=2)+"\n")
    env=None; io=None; policy=None; video=None; rows=[]; report={}
    started=time.monotonic()
    try:
        perception=LearnedPerception(threshold=args.threshold,model_set="production")
        env=EmptyEnv(robot_uids="fetch",num_envs=1,obs_mode="state",reward_mode="none",
            render_mode="rgb_array",control_mode="pd_joint_pos",sim_backend="physx_cpu",render_backend="cpu",
            sensor_configs=dict(width=640,height=480,fetch_head=dict(fov=1.10),fetch_hand=dict(fov=1.50)),
            human_render_camera_configs=dict(width=960,height=540,fov=1.05,near=.05,far=100))
        env.reset(seed=91)
        base=env.unwrapped
        base.agent.reset(Fetch.keyframes["rest"].qpos)
        io=SimulatorIO(env)  # Preserve native rest seed for later physical IK.
        io.review_sensor="fetch_head"
        initialize_compact_arm(base)
        io.arm=array(io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].copy()
        io.stabilize_stationary_base=True
        io.ceiling_safe_overview=True
        io.body[1]=.70
        directory=ROOT/".maniskill_assets/data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects"
        table=inspect_asset(directory,args.table_asset)
        # Environment construction only: place each asset's near edge at the
        # same physical standoff, so a larger table cannot overlap the robot.
        table_xy=(.50-float(table.lower[0]),-float((table.lower[1]+table.upper[1])/2))
        _,table_meta=build_household_asset(base,table,xy=table_xy,support_height=0.,role="placement")
        table_z=float(table.upper[2]-table.lower[2]+.002)
        pickup=inspect_asset(directory,recipe["environment_assets"]["pickup"],recipe.get("pickup_scale",1.))
        receptacle=inspect_asset(directory,recipe["environment_assets"]["placement"])
        target,target_meta=build_household_asset(base,pickup,xy=(.65,-.18),support_height=table_z,role="pickup")
        destination,destination_meta=build_household_asset(base,receptacle,xy=(.65,.18),support_height=table_z,role="placement")
        (args.output/"environment_only.json").write_text(json.dumps(dict(
            table=table_meta,target=target_meta,destination=destination_meta),indent=2)+"\n")
        policy=InstructionSearchPilot(io,perception,args.output,instruction)
        if args.carry_yaw=="base-lateral":
            from instruction_geometry import base_lateral_carry_rotation
            policy.carry_phase_label="Calibrate gravity-preserving carry yaw"
            policy.carry_posture_description="gravity_preserving_base_lateral_calibration"
            def carry_rotation(measured):
                candidate,angle=base_lateral_carry_rotation(measured,io.pose()[2])
                policy.event("calibration_carry_yaw_candidate",yaw_change_rad=angle,
                    measured_rotation=measured.tolist(),requested_rotation=candidate.tolist(),
                    source="measured robot/gripper orientation only; no task actor pose")
                return candidate
            policy.carry_rotation=carry_rotation
        if args.video:
            video=EvidenceVideo(args.output,hz=base.control_freq)
        def after_step():
            force=base.scene.get_pairwise_contact_forces(target,destination)
            rows.append(dict(step=io.step_id,stage=policy.task_stage,phase=policy.phase,
                target_xyz=array(target.pose.p)[0].tolist(),target_velocity=array(target.get_linear_velocity())[0].tolist(),
                target_quaternion=array(target.pose.q)[0].tolist(),
                bilateral_contact=bool(io.robot.is_grasping(target)[0]),tcp_xyz=array(io.robot.tcp_pose.p)[0].tolist(),
                receptacle_contact_force_n=float(np.linalg.norm(array(force))),
                inside_receptacle=placement_geometry(target,destination,pickup,receptacle,instruction.placement_relation)))
            if video:
                video.capture(io,policy.occupancy,"Manipulation calibration: "+policy.phase,
                              memory=policy.memory,query=policy.query,cached=policy.cached)
        io.after_step=after_step
        policy.phase="Initial settling"
        for _ in range(100):
            io.command()
        detection=policy.observe()
        if detection is None:
            # A failed single camera pose is not a class-availability test.
            # Acquire real alternate head views; keep arm and base targets.
            for pitch in (.9,.45):
                io.body[1]=pitch
                detection=policy.scan_sweep()
                if detection is not None:
                    break
        observed=(policy.memory.observations[detection.observation_id]
                  if detection is not None else policy.last_obs)
        cv2.imwrite(str(args.output/"initial_head.png"),cv2.cvtColor(observed.rgb,cv2.COLOR_RGB2BGR))
        report=dict(scope="Learned near-table manipulation calibration, not a complete search or dynamic episode",
            pickup_detection=asdict(detection) if detection else None,
            placement_detections=[asdict(d) for d in perception.detect(observed,instruction.placement_query)],
            support_detections=[asdict(d) for d in perception.detect(observed,instruction.support_query)]
                if instruction.support_query else [],grasp_requested=args.grasp,place_requested=args.place,
            complete_instruction_task_success=False)
        if detection is None:
            raise RuntimeError("No learned pickup detection in calibration view")
        proposal,fit=tabletop_grasp(observed,detection)
        report["grasp_proposal"]=dict(center=proposal.tolist(),**fit)
        print(json.dumps(report),flush=True)
        if args.grasp:
            policy.cached=np.asarray(detection.point_world)
            policy.cached_detection=detection
            report["grasp_proprioception_retained"]=policy.attempt_grasp()
            report["independent_current_grasp_contact"]=bool(io.robot.is_grasping(target)[0])
            if args.place and report["grasp_proprioception_retained"]:
                policy.prepare_carry()
                policy.start_placement_search()
                policy.scan_sweep()
                report["local_observed_placement_approach"]=policy.navigate(budget=20)
                report["placement_motion_completed"]=(policy.place_observed()
                    if report["local_observed_placement_approach"] else False)
        report["target_final_xyz"]=array(target.pose.p)[0].tolist()
        report["target_final_speed_m_s"]=float(np.linalg.norm(array(target.get_linear_velocity())[0]))
        # Calibration contact and lift only. The future task evaluator must
        # additionally verify identity, relocation, room transitions and region.
        from learned_evaluation import longest_true_run
        lifted=[r["bilateral_contact"] and r["target_xyz"][2]>table_z-pickup.lower[2]+.06 for r in rows]
        report["sustained_bilateral_lift_s"]=longest_true_run(lifted)/base.control_freq
        report["calibration_grasp_passed"]=report["sustained_bilateral_lift_s"]>=1.
        report["wall_time_s"]=time.monotonic()-started
    except Exception as error:
        report["error"]=repr(error)
        (args.output/"failure.json").write_text(json.dumps(dict(error=repr(error)),indent=2)+"\n")
        raise
    finally:
        if video: video.close()
        if rows:
            from learned_evaluation import longest_true_run
            reference=float(np.median([r["target_xyz"][2] for r in rows[:min(100,len(rows))]]))
            held=[r["bilateral_contact"] and r["target_xyz"][2]>reference+.06 for r in rows]
            stable=[not r["bilateral_contact"] and r["inside_receptacle"] and r["receptacle_contact_force_n"]>.02
                    and np.linalg.norm(r["target_velocity"])<.05 and r["stage"] in ("place","finished") for r in rows]
            report.update(sustained_bilateral_lift_s=longest_true_run(held)/base.control_freq,
                calibration_grasp_passed=longest_true_run(held)>=base.control_freq,
                stable_released_placement_s=longest_true_run(stable)/base.control_freq,
                calibration_placement_passed=bool(len(stable)>=40 and all(stable[-40:])),
                complete_instruction_task_success=False,target_final_xyz=rows[-1]["target_xyz"],
                target_final_speed_m_s=float(np.linalg.norm(rows[-1]["target_velocity"])))
        if io:
            (args.output/"actions.json").write_text(json.dumps(io.trace)+"\n")
        (args.output/"evaluator_trajectory.json").write_text(json.dumps(rows)+"\n")
        if policy:
            (args.output/"memory_updates.json").write_text(json.dumps(policy.memory.updates,indent=2)+"\n")
        after={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
        report["source_files_unchanged_during_execution"]=after==source_hashes
        (args.output/"source_hashes_after.json").write_text(json.dumps(after,indent=2)+"\n")
        (args.output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
        if env: env.close()


if __name__=="__main__":
    main()
