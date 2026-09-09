#!/usr/bin/env python3
"""Replay one actual grasp, then physically test compact joint-space carrying.

This is an embodiment calibration, not an additional independent learned task.
Every candidate is reported; no target attachment or recorded-state playback.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from maniskill_learned_probe import (make_env,SimulatorIO,array,build_asset_target,
                                    build_delivery_bin,exclude_existing_apples,initialize_compact_arm)
from maniskill_learned_dynamic import ExternalMove
from mani_skill.utils.structs.pose import Pose
from inspect_fetch_envelope import measure
from dream_fetch_navigation import FetchObservedMap
from learned_video import EvidenceVideo


def run(source,output,joints,prefix_event):
    config=json.loads((source/"configuration.json").read_text())
    task=json.loads((source/"environment_task.json").read_text())
    actions=json.loads((source/"actions.json").read_text())
    truth=json.loads((source/"evaluator_trajectory.json").read_text())
    events=[json.loads(s) for s in (source/"events.jsonl").read_text().splitlines()]
    stop=next(event["step"] for event in events if event["event"]==prefix_event)
    env=make_env(config["scene"],width=640,height=480)
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    video=None
    maxima={"base":0.,"target":0.}
    try:
        env.reset(seed=config["seed"])
        base=env.unwrapped
        base.agent.robot.set_pose(Pose.create_from_pq(p=[[*task["spawn_xy"],.02]]))
        exclude_existing_apples(base)
        support,target=build_asset_target(base,np.asarray(config["target_xy"]),"Apple_1")
        build_delivery_bin(base,np.asarray(task["bin_xy"]))
        if config.get("initial_compact_arm",False):
            initialize_compact_arm(base)
        mover=ExternalMove(support,config["endpoint"],config["disturbance_step"],task.get("disturbance_waypoints"))
        io=SimulatorIO(env)
        io.ceiling_safe_overview=True
        for action,original in zip(actions,truth):
            if action["step"]>stop:
                break
            if base.agent.control_mode!=action["control_mode"]:
                base.agent.set_control_mode(action["control_mode"])
                base.agent.controller.reset()
            mover.before_step(action["step"]-1)
            value=np.asarray(action["action"],np.float32)
            env.step(value[None] if action["control_mode"]=="pd_ee_delta_pose" else value)
            maxima["base"]=max(maxima["base"],float(np.linalg.norm(io.pose()[:2]-np.asarray(action["base_xyyaw"])[:2])))
            maxima["target"]=max(maxima["target"],float(np.linalg.norm(array(target.pose.p)[0]-original["target_xyz"])))
            if max(maxima.values())>1e-4:
                raise RuntimeError(f"Calibration source replay diverged: {maxima}")
        io.step_id=stop
        io.stabilize_stationary_base=config.get("initial_compact_arm",False)
        # The prefix can end immediately before the policy's unrecorded mode
        # switch. Joint targets are ignored by SimulatorIO in EE-delta mode.
        if base.agent.control_mode != "pd_joint_pos":
            base.agent.set_control_mode("pd_joint_pos")
            base.agent.controller.reset()
        q=array(io.robot.robot.get_qpos())[0]
        io.arm=q[[5,7,8,9,10,11,12]].copy()
        io.body=q[[4,6,3]].copy()
        io.grip=-1.
        initial_height=float(array(target.pose.p)[0,2])
        phase="Raise held object"
        video=EvidenceVideo(output)
        occupancy=FetchObservedMap()
        def after():
            rows.append(dict(step=io.step_id,phase=phase,target_xyz=array(target.pose.p)[0].tolist(),
                             bilateral_contact=bool(io.robot.is_grasping(target)[0]),
                             finger_gap=float(array(io.robot.robot.get_qpos())[0,-2:].sum())))
            video.capture(io,occupancy,phase)
        io.after_step=after
        io.body[2]=.385
        for _ in range(140):
            io.command()
        phase="Fold held arm"
        initial=io.arm.copy()
        for alpha in np.linspace(0,1,260):
            io.arm=initial*(1-alpha)+np.asarray(joints)*alpha
            io.command()
        for _ in range(60):
            io.command()
        envelope=measure(io)
        measured_arm=array(io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]]
        phase="Turn with compact carry"
        yaw=io.pose()[2]
        for offset in (np.pi/2,np.pi,3*np.pi/2,2*np.pi):
            io.turn(yaw+offset)
        final=array(target.pose.p)[0]
        report=dict(source_run=str(source.resolve()),source_prefix_steps=stop,replay_maximum_errors=maxima,
            joints=joints,measured_arm=measured_arm.tolist(),
            maximum_joint_error_rad=float(np.max(np.abs(measured_arm-np.asarray(joints)))),
            control_mode=base.agent.control_mode,
            torso=.385,initial_height=initial_height,final_target_height=float(final[2]),
            final_bilateral_contact=bool(io.robot.is_grasping(target)[0]),envelope=envelope,
            min_target_height_during_fold=min(row["target_xyz"][2] for row in rows),
            final_tcp=array(io.robot.tcp_pose.p)[0].tolist(),final_base=io.pose().tolist(),
            boundary="Physical controller calibration after exact source action replay, not a new learned-policy trial")
        (output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
        return report
    finally:
        if video:
            video.close()
        (output/"calibration_trajectory.json").write_text(json.dumps(rows)+"\n")
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prefix-event",default="compact_carry_posture")
    parser.add_argument("--side-only",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    results=[]
    for name,joints in (("side_fold",[-1.2,1.3,0.,1.8,0.,1.2,0.]),
                        ("front_fold",[-.3,1.3,0.,1.8,0.,1.2,0.]),
                        ("front_high",[-.3,1.2,0.,1.9,0.,1.0,0.])):
        if args.side_only and name!="side_fold":
            continue
        try:
            results.append(dict(name=name,**run(args.source_run,args.output/name,joints,args.prefix_event)))
        except Exception as error:
            results.append(dict(name=name,error=repr(error)))
            print(json.dumps(results[-1]),flush=True)
    (args.output/"all_candidates.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
