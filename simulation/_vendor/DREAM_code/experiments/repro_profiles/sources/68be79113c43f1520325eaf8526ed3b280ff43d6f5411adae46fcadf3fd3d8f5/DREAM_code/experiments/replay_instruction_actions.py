#!/usr/bin/env python3
"""Independent physical re-execution for the instruction-first task.

Every control and external force is executed in a newly initialized simulator.
Recorded robot/target states are comparisons only, never assigned to actors.
This is a reproducibility/contact check, not another learned-policy episode.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from maniskill_learned_probe import ROOT,make_env,array,SimulatorIO,initialize_compact_arm,sha
from instruction_environment import create_fixture,exclude_initial_categories,sample_start
from instruction_evaluation import evaluate_instruction,placement_geometry
from instruction_task import parse_instruction
from physics_contact_audit import ContactAudit
from mani_skill.utils.structs.pose import Pose
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tolerance",type=float,default=1e-4)
    parser.add_argument("--render-review",type=Path,help="Optional version-2 record audit for an alternate display-only spectator video")
    args=parser.parse_args()
    source=args.source_run
    config=json.loads((source/"configuration.json").read_text())
    if config.get("entrypoint")!="run_instruction_task.py":
        raise ValueError("Not an instruction-task recording")
    task=json.loads((source/"environment_task.json").read_text())
    original=json.loads((source/"result.json").read_text())
    controls=json.loads((source/"actions.json").read_text())
    original_rows=json.loads((source/"evaluator_trajectory.json").read_text())
    forces={r["step"]:r for r in json.loads((source/"disturbance_forces.json").read_text())}
    events=[json.loads(line) for line in (source/"events.jsonl").read_text().splitlines()]
    if not controls or len(controls)!=len(original_rows):
        raise ValueError("Incomplete source control/evaluator trace; do not certify it")
    if any(c["step"]!=r["step"] or c["step"]!=i+1 for i,(c,r) in enumerate(zip(controls,original_rows))):
        raise ValueError("Control/evaluator step IDs are not contiguous and aligned")
    args.output.mkdir(parents=True,exist_ok=False)
    sources=list(Path(__file__).parent.glob("*.py"))+list((ROOT/"DREAM_code/src/dream").rglob("*.py"))
    hashes={str(p.relative_to(ROOT/"DREAM_code")):sha(p) for p in sources}
    recorded_hashes=json.loads((source/"source_hashes_before.json").read_text())
    (args.output/"replay_source_hashes.json").write_text(json.dumps(hashes,indent=2)+"\n")
    with np.load(source/"evaluator_room_map.npz") as data:
        room_map={key:data[key] for key in data.files}
    instruction=parse_instruction(task["instruction"])
    torch.set_num_threads(2)
    started=time.monotonic()
    env=make_env(task["scene"],width=640,height=480)
    rows=[]; errors=[];video=None
    try:
        env.reset(seed=task["seed"])
        base=env.unwrapped
        spawn,yaw,_=sample_start(room_map,task["initial_room"],task["seed"],
            [(task["placement_table_xy"],task["placement_table_collision_radius_m"])],
            target_xy=task["target_xy"],docking_xy=task["endpoint"])
        base.agent.robot.set_pose(Pose.create_from_pq(p=[[*spawn,.02]]))
        q=base.agent.robot.get_qpos().clone();q[:,2]=yaw
        base.agent.robot.set_qpos(q);base.agent.controller.reset()
        io=SimulatorIO(env)
        initialize_compact_arm(base)
        categories={task["recipe"]["environment_assets"][role].rsplit("_",1)[0]
                    for role in ("pickup","placement")}
        exclude_initial_categories(base,categories)
        directory=ROOT/".maniskill_assets/data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects"
        cart,target,destination,pickup,receptacle,_=create_fixture(base,directory,task)
        if args.render_review:
            from instruction_replay_video import InstructionReplayVideo
            video=InstructionReplayVideo(source,args.render_review,args.output)
        fixtures={cart.name,destination.name,"instruction_placement_"+task["placement_table_asset"]}
        audit=ContactAudit(base,target_name=target.name,fixture_names=fixtures)
        for control,truth in zip(controls,original_rows):
            step=control["step"]
            if base.agent.control_mode!=control["control_mode"]:
                base.agent.set_control_mode(control["control_mode"]);base.agent.controller.reset()
            if step-1 in forces:
                cart.apply_force(np.r_[forces[step-1]["force"],0.].astype(np.float32))
                if step-1==original.get("relocation_done_step"):
                    cart.set_locked_motion_axes([True]*6)
            audit.current_step=step;audit.phase=truth["phase"]
            action=np.asarray(control["action"],dtype=np.float32)
            env.step(action[None] if control["control_mode"]=="pd_ee_delta_pose" else action)
            pose=io.pose();tcp=array(base.agent.tcp_pose.p)[0];xyz=array(target.pose.p)[0]
            errors.append(dict(step=step,base_xy_m=float(np.linalg.norm(pose[:2]-control["base_xyyaw"][:2])),
                yaw_rad=float(abs(np.arctan2(np.sin(pose[2]-control["base_xyyaw"][2]),
                                            np.cos(pose[2]-control["base_xyyaw"][2])))),
                tcp_xyz_m=float(np.linalg.norm(tcp-control["tcp_xyz"])),
                target_xyz_m=float(np.linalg.norm(xyz-truth["target_xyz"]))))
            force=base.scene.get_pairwise_contact_forces(target,destination)
            rows.append(dict(step=step,task_stage=truth["task_stage"],phase=truth["phase"],
                base_xyyaw=pose.tolist(),target_xyz=xyz.tolist(),
                target_velocity=array(target.get_linear_velocity())[0].tolist(),
                bilateral_contact=bool(base.agent.is_grasping(target)[0]),
                receptacle_contact_force_n=float(np.linalg.norm(array(force))),
                inside_receptacle=placement_geometry(target,destination,pickup,receptacle,instruction.placement_relation)))
            if video:video.frame(io,step,truth["phase"])
            if step%1000==0:print(json.dumps(errors[-1]),flush=True)
        move=SimpleNamespace(observation_checks=json.loads((source/"evaluator_discoveries.json").read_text()),
            gate=SimpleNamespace(started_step=original.get("relocation_start_step")),
            done_step=original.get("relocation_done_step"),trace=list(forces.values()),
            initial=np.asarray(json.loads((source/"initial_conditions.json").read_text())["fixtures"]["target"]["initial_origin_xyz"]))
        evaluation=evaluate_instruction(rows,events,controls,move,room_map,hz=base.control_freq)
        maxima={k:max(e[k] for e in errors) for k in ("base_xy_m","yaw_rad","tcp_xyz_m","target_xyz_m")}
        report=dict(source_run=str(source.resolve()),steps=len(rows),maximum_error=maxima,
            tolerance=args.tolerance,physical_reexecution_passed=all(v<=args.tolerance for v in maxima.values()),
            evaluation=evaluation,contact_audit=audit.report(),
            source_recording_unchanged=bool(original.get("source_files_unchanged")),
            replay_environment_source_matches_recording=all(hashes.get(k)==recorded_hashes.get(k) for k in
                ("experiments/instruction_environment.py","experiments/instruction_assets.py","experiments/maniskill_learned_probe.py")),
            replay_source_unchanged=all(sha(p)==hashes[str(p.relative_to(ROOT/"DREAM_code"))] for p in sources),
            wall_time_s=time.monotonic()-started,release_ready=False,
            boundary="Exact saved-control/force re-execution. Not a new learned-policy trial; video/visibility/source review remains separate.")
        if video:report["alternate_spectator_video"]=video.close()
        (args.output/"evaluator_trajectory.json").write_text(json.dumps(rows)+"\n")
        (args.output/"control_errors.json").write_text(json.dumps(errors)+"\n")
        (args.output/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k!="contact_audit"},indent=2),flush=True)
    finally:
        if video and not video.closed:video.close()
        env.close()


if __name__=="__main__":
    main()
