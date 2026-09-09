#!/usr/bin/env python3
"""Physical head-scan calibration, not a learned navigation/task experiment."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from maniskill_learned_probe import make_env,SimulatorIO,array,initialize_compact_arm
from maniskill_crossroom_policy import CrossRoomSearchPilot
from dream_fetch_navigation import FetchObservedMap
from learned_video import EvidenceVideo


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scene",default="ArchitecTHOR-Test-02")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    env=make_env(args.scene,width=640,height=480)
    video=None
    io=None
    rows=[]
    try:
        env.reset(seed=91)
        initialize_compact_arm(env.unwrapped)  # initial condition, before physics
        io=SimulatorIO(env)
        io.stabilize_stationary_base=True
        io.ceiling_safe_overview=True
        io.display_sensor="fetch_head"
        io.body[1]=.25
        for _ in range(100):
            io.command()
        # New recording begins after settling, with an explicit local step
        # clock. This is a calibration segment, not an uninterrupted task video.
        io.trace=[]
        io.step_id=0
        pol=object.__new__(CrossRoomSearchPilot)
        pol.io=io
        pol.output=args.output
        pol.events=[]
        pol.phase="Head-camera calibration"
        pol.occupancy=FetchObservedMap(origin=io.pose()[:2]-24.,size=1200,resolution=.04,radius=.40)
        pol.current_detection=None
        def observe(sensor="fetch_head"):
            obs=io.capture(sensor)
            obs.save(args.output/f"observation_{obs.frame_id:04d}.npz")
            pol.occupancy.integrate(obs,self_spheres=pol.robot_filter_spheres())
            cv2.imwrite(str(args.output/f"head_view_{obs.frame_id:02d}.png"),cv2.cvtColor(obs.rgb,cv2.COLOR_RGB2BGR))
            pol.event("calibration_head_observation",sensor=sensor,frame_id=obs.frame_id)
            return None  # No learned detector is run in this control calibration.
        pol.observe=observe
        arm_indices=[5,7,8,9,10,11,12]
        initial_arm=array(io.robot.robot.get_qpos())[0,arm_indices].copy()
        initial_base=io.pose()[:2].copy()
        initial_command=io.arm.copy()
        video=EvidenceVideo(args.output)
        def after():
            q=array(io.robot.robot.get_qpos())[0]
            rows.append(dict(step=io.step_id,measured_arm=q[arm_indices].tolist(),
                             commanded_arm=io.arm.tolist(),head_pan_rad=float(q[4]),
                             base_xy=io.pose()[:2].tolist()))
            video.capture(io,pol.occupancy,pol.phase)
        io.after_step=after
        for _ in range(3):
            pol.scan_sweep()
            for _ in range(20):
                io.command()
        arm_error=float(np.max(np.abs(np.array([r["measured_arm"] for r in rows])-initial_arm)))
        base_error=float(np.max(np.linalg.norm(np.array([r["base_xy"] for r in rows])-initial_base,axis=1)))
        head_range=float(np.ptp([r["head_pan_rad"] for r in rows]))
        checks=dict(arm_commands_constant=all(np.array_equal(r["commanded_arm"],initial_command) for r in rows),
                    arm_held_within_003_rad=arm_error<.03,base_held_within_001_m=base_error<.01,
                    head_actually_panned=head_range>.5,
                    recorded_sensor_is_head_only=all(r["sensor"]=="fetch_head" for r in video.rows))
        report=dict(scope="Physical head/arm control calibration; no learned search, grasp or placement trial.",
            scene=args.scene,recorded_control_steps=io.step_id,initial_settling_steps=100,
            maximum_arm_deviation_rad=arm_error,maximum_base_displacement_m=base_error,
            measured_head_pan_range_rad=head_range,checks=checks,passed=all(checks.values()))
        (args.output/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
        if not report["passed"]:
            raise RuntimeError("Physical head-scan calibration did not satisfy its checks")
    finally:
        if video:
            video.close()
        if io:
            (args.output/"actions.json").write_text(json.dumps(io.trace)+"\n")
        (args.output/"joint_trace.json").write_text(json.dumps(rows)+"\n")
        env.close()


if __name__=="__main__":
    main()
