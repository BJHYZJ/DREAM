#!/usr/bin/env python3
"""Physics-only fixture check; no learned task success is inferred here."""
import argparse
import json
import math
from pathlib import Path

from maniskill_learned_probe import make_env,SimulatorIO,array,build_asset_target,build_delivery_bin,exclude_existing_apples
from maniskill_learned_dynamic import ExternalMove
from dream.dynamic_memory import initial_scan_yaws
from mani_skill.utils.structs.pose import Pose
import numpy as np
import torch


def check(task):
    env=make_env(task["scene"],width=320,height=240)
    io=None
    try:
        env.reset(seed=task["seed"])
        base=env.unwrapped
        base.agent.robot.set_pose(Pose.create_from_pq(p=[[*task["spawn_xy"],.02]]))
        exclude_existing_apples(base)
        support,target=build_asset_target(base,np.array(task["target_xy"]),"Apple_1")
        build_delivery_bin(base,np.array(task["bin_xy"]))
        io=SimulatorIO(env)
        mover=ExternalMove(support,task["endpoint"],task["disturbance_step"],task["disturbance_waypoints"])
        io.before_step=lambda:mover.before_step(io.step_id)
        io.prepare_scan()
        io.body[1]=.30
        for yaw in initial_scan_yaws(io.pose()[2]):
            io.turn(yaw)
            for _ in range(5):
                io.command()
        io.turn(math.pi)
        while io.step_id<task["query_step"]:
            io.command()
        xyz=array(target.pose.p)[0]
        support_xy=array(support.pose.p)[0,:2]
        endpoint_error=float(np.linalg.norm(support_xy-task["endpoint"]))
        payload_error=float(np.linalg.norm(xyz[:2]-support_xy))
        return dict(scene=task["scene"],valid=bool(endpoint_error<.10 and payload_error<.13 and .93<xyz[2]<1.02),
            steps=io.step_id,endpoint_error_m=endpoint_error,payload_offset_m=payload_error,
            target_height_m=float(xyz[2]),cart_done_step=mover.done_step,
            scope="Physics-only initial scan/relocation check, not detection/search/grasp evidence")
    except Exception as error:
        return dict(scene=task["scene"],valid=False,error=repr(error),steps=io.step_id if io else None)
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    rows=[]
    for path in sorted(args.tasks.glob("[0-9][0-9]_*.json")):
        row=check(json.loads(path.read_text()))
        rows.append(row)
        print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    main()
