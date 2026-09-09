#!/usr/bin/env python3
"""Physical delivery-controller calibration after exact grasp-prefix replay.

No learned searching is rerun: this is NOT another independent task result.
The suffix uses real depth and base/arm PD controls; no actor state is a policy
input. Evaluator actor state is used only for replay checking and final scoring.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from maniskill_learned_probe import (make_env,SimulatorIO,array,build_asset_target,
    build_delivery_bin,exclude_existing_apples,initialize_compact_arm)
from maniskill_learned_dynamic import ExternalMove
from maniskill_crossroom_policy import CrossRoomSearchPilot
from dream_learned_core import RGBDObservation
from learned_video import EvidenceVideo
from learned_evaluation import evaluate_episode
from crossroom_evaluation import evaluate_room_transitions
from mani_skill.utils.structs.pose import Pose


class DepthOnlyDelivery(CrossRoomSearchPilot):
    def observe(self,sensor_name="fetch_head"):
        self.io.display_sensor=sensor_name
        self.geometry_observe()
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source=args.source_run
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((source/"configuration.json").read_text())
    task=json.loads((source/"environment_task.json").read_text())
    actions=json.loads((source/"actions.json").read_text())
    truth=json.loads((source/"evaluator_trajectory.json").read_text())
    events=[json.loads(s) for s in (source/"events.jsonl").read_text().splitlines()]
    stop=next(e["step"] for e in events if e["event"]=="grip_retention_proprioception")
    by_step=defaultdict(list)
    for event in events:
        if event["event"] in ("observation","navigation_depth_update") and event["step"]<=stop:
            prefix="observation" if event["event"]=="observation" else "navigation_depth"
            by_step[event["step"]].append(source/f"{prefix}_{event['frame_id']:05d}.npz")
    torch.set_num_threads(1)
    env=make_env(config["scene"],width=config["sensor_width"],height=3*config["sensor_width"]//4)
    video=None
    io=None
    evaluation=[]
    maxima=dict(base=0.,target=0.)
    try:
        env.reset(seed=config["seed"])
        base=env.unwrapped
        base.agent.robot.set_pose(Pose.create_from_pq(p=[[*task["spawn_xy"],.02]]))
        exclude_existing_apples(base)
        support,target=build_asset_target(base,np.asarray(config["target_xy"]),"Apple_1")
        build_delivery_bin(base,np.asarray(task["bin_xy"]))
        io=SimulatorIO(env)
        if config.get("initial_compact_arm",False):
            initialize_compact_arm(base)
        io.ceiling_safe_overview=True
        policy=DepthOnlyDelivery(io,None,output,"apple",True)
        mover=ExternalMove(support,config["endpoint"],config["disturbance_step"],task.get("disturbance_waypoints"))
        for action,original in zip(actions,truth):
            if action["step"]>stop:
                break
            if base.agent.control_mode!=action["control_mode"]:
                base.agent.set_control_mode(action["control_mode"])
                base.agent.controller.reset()
            mover.before_step(action["step"]-1)
            value=np.asarray(action["action"],np.float32)
            env.step(value[None] if action["control_mode"]=="pd_ee_delta_pose" else value)
            io.step_id=action["step"]
            maxima["base"]=max(maxima["base"],float(np.linalg.norm(io.pose()[:2]-np.asarray(action["base_xyyaw"])[:2])))
            maxima["target"]=max(maxima["target"],float(np.linalg.norm(array(target.pose.p)[0]-original["target_xyz"])))
            if max(maxima.values())>1e-4:
                raise RuntimeError(f"Source replay diverged: {maxima}")
            io.trace.append(action)
            evaluation.append(original)
            for path in by_step[action["step"]]:
                with np.load(path) as saved:
                    obs=RGBDObservation(**{key:saved[key].item() if saved[key].shape==() else saved[key] for key in saved.files})
                io.frame_id=obs.frame_id
                policy.occupancy.integrate(obs,self_spheres=policy.robot_filter_spheres())
        q=array(io.robot.robot.get_qpos())[0]
        io.arm=q[[5,7,8,9,10,11,12]].copy()
        io.body=q[[4,6,3]].copy()
        io.grip=-1.
        io.hold_measured_arm()
        io.stabilize_stationary_base=True
        policy.perceived_payload_radius=next(e["radius_m"] for e in events if e["event"]=="depth_grasp_center")
        policy.events=[e for e in events if e["step"]<=stop]
        video=EvidenceVideo(output)
        def after():
            evaluation.append(dict(step=io.step_id,phase=policy.phase,target_xyz=array(target.pose.p)[0].tolist(),
                target_velocity=array(target.get_linear_velocity())[0].tolist(),
                bilateral_contact=bool(io.robot.is_grasping(target)[0]) if io.grip<0 else False))
            video.capture(io,policy.occupancy,policy.phase,planned_path=policy.planned_path,
                waypoint=policy.waypoint,navigation_goal=policy.navigation_goal)
        io.after_step=after
        completed=policy.carry_and_place(task["spawn_xy"],task["bin_xy"])
        for _ in range(20):
            io.command()
        result=evaluate_episode(evaluation,io.trace,policy.events,np.asarray(task["bin_xy"]))
        with np.load(source/"evaluator_room_map.npz") as room:
            result.update(evaluate_room_transitions(io.trace,policy.events,task,{k:room[k] for k in room.files},config["query_step"]))
        result.update(boundary="Controller suffix calibration after exact source action replay; not an independent learned-policy trial.",
            source_run=str(source.resolve()),source_prefix_steps=stop,replay_maximum_errors=maxima,
            suffix_steps=io.step_id-stop,placement_motion_completed=bool(completed))
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2),flush=True)
    finally:
        if video:
            video.close()
        if io:
            (output/"actions.json").write_text(json.dumps(io.trace)+"\n")
        (output/"evaluator_trajectory.json").write_text(json.dumps(evaluation)+"\n")
        env.close()


if __name__=="__main__":
    main()
