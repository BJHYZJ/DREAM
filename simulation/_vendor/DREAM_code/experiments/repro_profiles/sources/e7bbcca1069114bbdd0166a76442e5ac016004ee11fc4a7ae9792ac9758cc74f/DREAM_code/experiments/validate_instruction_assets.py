#!/usr/bin/env python3
"""Render real candidate assets and test open-receptacle physics.

This is an asset/fixture calibration, NOT a robot search, grasp or place trial.
Objects are initially constructed on a table; a small sphere is gravity-dropped
into each bowl to catch accidentally convex/closed collision geometry.
"""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault("MS_ASSET_DIR",str(ROOT/".maniskill_assets"))
os.environ.setdefault("VK_ICD_FILENAMES","/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT","1")

import cv2
import gymnasium as gym
import mani_skill  # noqa: F401
import numpy as np
import sapien
import torch
from mani_skill.utils import sapien_utils
from mani_skill.envs.tasks.empty_env import EmptyEnv

from instruction_assets import inspect_asset,build_household_asset
from instruction_task import parse_instruction


class AssetCalibrationEnv(EmptyEnv):
    def get_state_dict(self):
        # ManiSkill 3.0.1 supports robot_uids='none' in loading/stepping but
        # its default state serializer still dereferences agent.controller.
        # This calibration has no robot controller; preserve its real actors.
        return self.scene.get_sim_state()


def array(value):
    return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)


def check(recipe,asset_directory,output):
    parsed=parse_instruction(recipe["instruction"])
    pickup=inspect_asset(asset_directory,recipe["environment_assets"]["pickup"],recipe.get("pickup_scale",1.))
    placement=inspect_asset(asset_directory,recipe["environment_assets"]["placement"])
    camera=sapien_utils.look_at([.85,-.85,1.35],[0,0,.77])
    env=AssetCalibrationEnv(robot_uids="none",num_envs=1,obs_mode="none",reward_mode="none",
        render_mode="rgb_array",sim_backend="physx_cpu",render_backend="cpu",
        human_render_camera_configs=dict(width=960,height=640,pose=camera,fov=.85))
    try:
        env.reset(seed=91)
        base=env.unwrapped
        builder=base.scene.create_actor_builder()
        builder.add_box_collision(half_size=[.5,.35,.35])
        color=[.80,.80,.78,1.] if "white" in (parsed.support_query or "") else [.44,.25,.12,1.]
        builder.add_box_visual(half_size=[.5,.35,.35],material=color)
        builder.set_initial_pose(sapien.Pose([0,0,.35]))
        builder.build_static(name="calibration_table_not_search_target")
        target,target_metadata=build_household_asset(base,pickup,xy=(-.19,0),support_height=.70,role="pickup")
        receptacle,destination_metadata=build_household_asset(base,placement,xy=(.20,0),support_height=.70,role="placement")
        for _ in range(100):
            env.step(None)
        base.scene.update_render(update_sensors=True,update_human_render_cameras=True)
        pixels=array(base.render_rgb_array("render_camera"))[0]
        cv2.imwrite(str(output/(recipe["id"]+"_assets.png")),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
        report=dict(recipe_id=recipe["id"],instruction=parsed.policy_payload(),pickup=target_metadata,
            placement=destination_metadata,settled_pickup_xyz=array(target.pose.p)[0].tolist(),
            settled_pickup_speed_m_s=float(np.linalg.norm(array(target.get_linear_velocity())[0])),
            pickup_horizontal_min_extent_m=float(min((pickup.upper-pickup.lower)[:2])),
            boundary="Asset/fixture calibration only; no robot, learned search or grasp success is claimed.")
        if parsed.placement_relation=="in":
            builder=base.scene.create_actor_builder()
            builder.add_sphere_collision(radius=.01,density=500)
            builder.add_sphere_visual(radius=.01,material=[.9,.25,.05,1.])
            rim=.702+(placement.upper[2]-placement.lower[2])
            builder.set_initial_pose(sapien.Pose([.20,0,rim+.08]))
            probe=builder.build(name="calibration_probe_not_task_target")
            for _ in range(100):
                env.step(None)
            xyz=array(probe.pose.p)[0]
            speed=float(np.linalg.norm(array(probe.get_linear_velocity())[0]))
            # Below the rim but supported above the underlying table, with
            # a one-centimetre probe. This does not prove object-sized fit.
            enters=bool(np.linalg.norm(xyz[:2]-[.20,0])<.04 and .711<xyz[2]<rim-.012 and speed<.01)
            report["bowl_opening_probe"]=dict(radius_m=.01,final_xyz=xyz.tolist(),rim_height_m=float(rim),
                speed_m_s=speed,entered_and_supported_below_rim=enters)
            base.scene.update_render(update_sensors=True,update_human_render_cameras=True)
            pixels=array(base.render_rgb_array("render_camera"))[0]
            cv2.imwrite(str(output/(recipe["id"]+"_probe.png")),cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
        return report
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipes",type=Path,default=Path(__file__).with_name("configs")/"instruction_pickplace_v1/recipes.json")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    directory=Path(os.environ["MS_ASSET_DIR"])/"data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects"
    reports=[]
    for recipe in json.loads(args.recipes.read_text())["recipes"]:
        try:
            row=check(recipe,directory,args.output)
        except Exception as error:
            (args.output/"failure.json").write_text(json.dumps(dict(recipe_id=recipe["id"],error=repr(error)),indent=2)+"\n")
            raise
        reports.append(row)
        (args.output/"asset_checks.json").write_text(json.dumps(reports,indent=2)+"\n")
        print(json.dumps(row),flush=True)
    page="<!doctype html><meta charset='utf-8'><title>Candidate assets — not robot experiments</title><h1>Candidate assets: not robot experiments</h1><p>Actual ManiSkill renders. No learned search/grasp/placement trial is represented here.</p>"
    for row in reports:
        page+=f"<h2>{row['recipe_id']}</h2><p>{row['instruction']['text']}</p><img style='max-width:960px;width:100%' src='{row['recipe_id']}_assets.png'>"
    (args.output/"index.html").write_text(page)


if __name__=="__main__":
    main()
