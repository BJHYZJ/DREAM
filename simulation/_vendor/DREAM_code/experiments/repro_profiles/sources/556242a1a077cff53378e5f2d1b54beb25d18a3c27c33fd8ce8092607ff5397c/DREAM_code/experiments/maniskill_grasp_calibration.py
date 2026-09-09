#!/usr/bin/env python3
"""Isolated RGB-D grasp calibration; never count as a search/effectiveness trial."""
import argparse
import json
from pathlib import Path
import shutil

from maniskill_learned_probe import make_env, SimulatorIO, build_asset_target, exclude_existing_apples, array
from maniskill_learned_dynamic import LearnedSearchPilot
from dream_learned_core import LearnedPerception
from learned_video import EvidenceVideo
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--video",action="store_true")
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    snapshot=args.output/"source_snapshot"
    snapshot.mkdir()
    for name in ("maniskill_grasp_calibration.py","maniskill_learned_dynamic.py",
                 "maniskill_learned_probe.py","dream_learned_core.py","depth_grasp.py","learned_video.py"):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    torch.set_num_threads(2)
    perception=LearnedPerception(model_set="base",threshold=.25)
    env=make_env("ArchitecTHOR-Test-03",640,480)
    recorder=None
    io=None
    evaluator=[]
    try:
        env.reset(seed=17)
        exclude_existing_apples(env.unwrapped)
        support,target=build_asset_target(env.unwrapped,np.array([-2.40,0.]),"Apple_1")
        support.set_locked_motion_axes([True]*6)
        io=SimulatorIO(env)
        policy=LearnedSearchPilot(io,perception,args.output,"apple",True)
        if args.video:
            recorder=EvidenceVideo(args.output)
        def after_step():
            evaluator.append(dict(step=io.step_id,target_xyz=array(target.pose.p)[0].tolist(),
                bilateral_contact=bool(io.robot.is_grasping(target)[0])))
            if recorder:
                recorder.capture(io,policy.occupancy,policy.phase,cached=policy.cached)
        io.after_step=after_step
        io.prepare_scan()
        io.body[1]=.85
        for _ in range(20):
            io.command()
        policy.observe()
        policy.retrieve(use_live=True)
        if policy.cached is None:
            raise RuntimeError("Calibration target not visually detected")
        attempted=policy.attempt_grasp()
        result=dict(boundary="Isolated calibration, not a DREAM task result",attempted=attempted,
                    bilateral_contact=bool(io.robot.is_grasping(target)[0]),
                    target_height_m=float(array(target.pose.p)[0,2]),steps=io.step_id)
        (args.output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)
    except Exception as error:
        (args.output/"failure.json").write_text(json.dumps(dict(error=repr(error)))+"\n")
        raise
    finally:
        if recorder:
            recorder.close()
        if io:
            (args.output/"actions.json").write_text(json.dumps(io.trace)+"\n")
        (args.output/"evaluator_trajectory.json").write_text(json.dumps(evaluator)+"\n")
        env.close()


if __name__=="__main__":
    main()
