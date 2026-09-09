#!/usr/bin/env python3
"""Re-execute saved controls in a fresh simulator and compare physical traces.

This is a reproducibility check, not a new learned-policy episode. It performs
every env.step; recorded states are never set onto robot/target actors.
"""
import argparse
import json
from pathlib import Path
import time

from maniskill_learned_probe import make_env,array,build_asset_target,build_delivery_bin,exclude_existing_apples,initialize_compact_arm
from maniskill_learned_dynamic import ExternalMove
from mani_skill.utils.structs.pose import Pose
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tolerance",type=float,default=1e-4)
    parser.add_argument("--contact-audit",action="store_true")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config=json.loads((args.source_run/"configuration.json").read_text())
    task=json.loads((args.source_run/"environment_task.json").read_text())
    actions=json.loads((args.source_run/"actions.json").read_text())
    evaluator=json.loads((args.source_run/"evaluator_trajectory.json").read_text())
    if len(actions)!=len(evaluator):
        raise ValueError("Action/evaluator traces are not aligned")
    torch.set_num_threads(2)
    env=make_env(config["scene"],width=config["sensor_width"],height=3*config["sensor_width"]//4)
    errors=[]
    started=time.monotonic()
    try:
        env.reset(seed=config["seed"])
        base=env.unwrapped
        if "spawn_xy" in task:
            base.agent.robot.set_pose(Pose.create_from_pq(p=[[*task["spawn_xy"],.02]]))
        if config["unique_target"]:
            exclude_existing_apples(base)
        support,target=build_asset_target(base,np.asarray(config["target_xy"]),"Apple_1")
        mover=ExternalMove(support,config["endpoint"],start_step=config.get("disturbance_step",480),
                           waypoints=task.get("disturbance_waypoints"))
        origin=array(base.agent.robot.pose.p)[0,:2].copy()
        if config.get("deliver",False):
            build_delivery_bin(base,np.asarray(task.get("bin_xy",origin+[-.7,0.])))
        if config.get("initial_compact_arm",False):
            initialize_compact_arm(base)
        contact_audit=None
        if args.contact_audit:
            from physics_contact_audit import ContactAudit
            contact_audit=ContactAudit(base)
        for control,truth in zip(actions,evaluator):
            if base.agent.control_mode!=control["control_mode"]:
                base.agent.set_control_mode(control["control_mode"])
                base.agent.controller.reset()
            mover.before_step(control["step"]-1)
            if contact_audit:
                contact_audit.current_step=control["step"]
                contact_audit.phase=truth["phase"]
            action=np.asarray(control["action"],dtype=np.float32)
            env.step(action[None] if control["control_mode"]=="pd_ee_delta_pose" else action)
            q=array(base.agent.robot.get_qpos())[0]
            robot=np.r_[origin+q[:2],q[2]]
            errors.append(dict(step=control["step"],
                base_xy_m=float(np.linalg.norm(robot[:2]-np.asarray(control["base_xyyaw"])[:2])),
                yaw_rad=float(abs(np.arctan2(np.sin(robot[2]-control["base_xyyaw"][2]),np.cos(robot[2]-control["base_xyyaw"][2])))),
                target_xyz_m=float(np.linalg.norm(array(target.pose.p)[0]-truth["target_xyz"])),
                tcp_xyz_m=float(np.linalg.norm(array(base.agent.tcp_pose.p)[0]-control["tcp_xyz"])) ))
            if control["step"]%1000==0:
                print(json.dumps(errors[-1]),flush=True)
        maxima={key:max(row[key] for row in errors) for key in ("base_xy_m","yaw_rad","target_xyz_m","tcp_xyz_m")}
        report=dict(source_run=str(args.source_run.resolve()),steps=len(actions),maximum_error=maxima,
            tolerance=args.tolerance,passed=all(v<=args.tolerance for v in maxima.values()),
            wall_time_s=time.monotonic()-started,
            scope="Open-loop re-execution of saved closed-loop controls, not an additional algorithm trial; no post-initialization actor/robot pose setters.")
        if contact_audit:
            report["contact_audit"]=contact_audit.report()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    finally:
        env.close()


if __name__=="__main__":
    main()
