#!/usr/bin/env python3
"""Re-execute controls exactly to render a ceiling-safe display-only camera.

The original head/wrist observations, memory panel, phase and time are retained.
No recorded actor/robot states are set. This is not another learned-policy trial.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def run_one(run,output,preview=False):
    from maniskill_learned_probe import make_env,array,build_asset_target,build_delivery_bin,exclude_existing_apples,initialize_compact_arm
    from maniskill_learned_dynamic import ExternalMove
    from mani_skill.utils.structs.pose import Pose
    from mani_skill.utils import sapien_utils
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch

    output.mkdir(parents=True,exist_ok=False)
    config=json.loads((run/"configuration.json").read_text())
    task=json.loads((run/"environment_task.json").read_text())
    actions=json.loads((run/"actions.json").read_text())
    evaluator=json.loads((run/"evaluator_trajectory.json").read_text())
    frames=json.loads((run/"video_frames.json").read_text())
    if len(actions)!=len(evaluator):
        raise ValueError("Unaligned source controls and evaluation")
    torch.set_num_threads(1)
    env=make_env(config["scene"],width=config["sensor_width"],height=3*config["sensor_width"]//4)
    video=cv2.VideoCapture(str(run/"reviewer_view.mp4"))
    writers={}
    maxima={key:0. for key in ("base_xy_m","yaw_rad","target_xyz_m","tcp_xyz_m")}
    count=0
    last_step=0
    try:
        env.reset(seed=config["seed"])
        base=env.unwrapped
        if "spawn_xy" in task:
            base.agent.robot.set_pose(Pose.create_from_pq(p=[[*task["spawn_xy"],.02]]))
        if config["unique_target"]:
            exclude_existing_apples(base)
        support,target=build_asset_target(base,np.asarray(config["target_xy"]),"Apple_1")
        mover=ExternalMove(support,config["endpoint"],config["disturbance_step"],task.get("disturbance_waypoints"))
        origin=array(base.agent.robot.pose.p)[0,:2].copy()
        if config["deliver"]:
            build_delivery_bin(base,np.asarray(task.get("bin_xy",origin+[-.7,0.])))
        if config.get("initial_compact_arm",False):
            initialize_compact_arm(base)
        if not preview:
            writers={name:imageio.get_writer(output/f"{name}.mp4",fps=5,codec="libx264",quality=8,
                macro_block_size=2,ffmpeg_log_level="error") for name in ("overview_raw","reviewer_view")}
        for control,truth in zip(actions,evaluator):
            if preview and control["step"]>360:
                break
            if base.agent.control_mode!=control["control_mode"]:
                base.agent.set_control_mode(control["control_mode"])
                base.agent.controller.reset()
            mover.before_step(control["step"]-1)
            action=np.asarray(control["action"],np.float32)
            env.step(action[None] if control["control_mode"]=="pd_ee_delta_pose" else action)
            q=array(base.agent.robot.get_qpos())[0]
            robot=np.r_[origin+q[:2],q[2]]
            errors=dict(base_xy_m=float(np.linalg.norm(robot[:2]-np.asarray(control["base_xyyaw"])[:2])),
                yaw_rad=float(abs(np.arctan2(np.sin(robot[2]-control["base_xyyaw"][2]),np.cos(robot[2]-control["base_xyyaw"][2])))),
                target_xyz_m=float(np.linalg.norm(array(target.pose.p)[0]-truth["target_xyz"])),
                tcp_xyz_m=float(np.linalg.norm(array(base.agent.tcp_pose.p)[0]-control["tcp_xyz"])))
            maxima={key:max(maxima[key],value) for key,value in errors.items()}
            if max(errors.values())>1e-4:
                raise RuntimeError(f"Physical re-execution diverged at step {control['step']}: {errors}")
            last_step=control["step"]
            if last_step%4:
                continue
            row=frames[count]
            if row["sim_step"]!=last_step:
                raise RuntimeError("Source video and physical controls do not align")
            ok,original=video.read()
            if not ok:
                raise RuntimeError("Cannot decode original observation/map video frame")
            camera=base._human_render_cameras["render_camera"].camera
            # Close to the base footprint, below low ceilings. Fixed
            # world heading; no target pose or privileged map controls this view.
            eye=[*(robot[:2]+[-.25,-.45]),1.95]
            look=[*(robot[:2]+[.20,0.]),.65]
            camera.set_fovy(1.25)
            camera.set_local_pose(sapien_utils.look_at(eye,look).sp)
            overview=array(base.render_rgb_array("render_camera"))[0].astype(np.uint8)
            composite=cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
            composite[36:576,:960]=overview
            if last_step==360:
                cv2.imwrite(str(output/"scene_step_00360.png"),cv2.cvtColor(overview,cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output/"preview.png"),cv2.cvtColor(composite,cv2.COLOR_RGB2BGR))
            for name,writer in writers.items():
                writer.append_data(overview if name=="overview_raw" else composite)
            count+=1
        if not preview and count!=len(frames):
            raise RuntimeError("Full re-render is missing source frames")
    finally:
        for writer in writers.values():
            writer.close()
        video.release()
        env.close()
    report=dict(source_run=str(run.resolve()),source_steps=len(actions),executed_steps=last_step,
        rendered_frames=count,preview_only=preview,maximum_error=maxima,passed=True,
        fps=5,sim_steps_per_frame=4,display_camera="base xy + (-0.25,-0.45), z=1.95 m, fovy=1.25 rad, fixed world heading",
        observation_and_map_source="Original recorded right-hand panel; no new learned decisions",
        scope="Exact physical action re-execution for display-only re-rendering, not a new algorithm trial; no post-initialization actor/robot state setters")
    for name in ("actions.json","evaluator_trajectory.json","reviewer_view.mp4"):
        hasher=hashlib.sha256()
        with (run/name).open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""):
                hasher.update(block)
        report.setdefault("source_sha256",{})[name]=hasher.hexdigest()
    (output/"reexecution_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run",type=Path)
    source.add_argument("--suite",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preview-only",action="store_true")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.run:
        run_one(args.run,args.output,args.preview_only)
        return
    suite=json.loads((args.suite/"suite_result.json").read_text())
    attempts=[x for x in suite["attempts"] if x["variant"]=="dynamic"]
    args.output.mkdir(parents=True,exist_ok=False)
    def worker(attempt):
        command=[sys.executable,str(Path(__file__).resolve()),"--run",str(args.suite/attempt["name"]),"--output",str(args.output/attempt["name"])]
        if args.preview_only:
            command.append("--preview-only")
        environment=os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        with (args.output/f"{attempt['name']}.log").open("w") as stream:
            subprocess.run(command,check=True,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        return json.loads((args.output/attempt["name"]/"reexecution_verification.json").read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports=list(pool.map(worker,attempts))
    (args.output/"all_reexecutions.json").write_text(json.dumps(reports,indent=2)+"\n")
    print(json.dumps(dict(runs=len(reports),passed=all(r["passed"] for r in reports))),flush=True)


if __name__=="__main__":
    main()
