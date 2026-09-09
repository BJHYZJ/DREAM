#!/usr/bin/env python3
"""Development-only physical scan with production DREAM learned perception.

This is a perception/mapping integration test, NOT a successful search/grasp
episode. It intentionally saves every detection, including false positives,
with separate evaluator-only ground truth for offline checking.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MS_ASSET_DIR", str(ROOT / ".maniskill_assets"))
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".dream_model_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")

import cv2
import gymnasium as gym
import numpy as np
import torch
import mani_skill  # noqa: F401
import sapien
from mani_skill.utils import sapien_utils

from architecthor_navigation import SCENE_IDS
from maniskill_dynamic_dream import build_moving_target
from dream_learned_core import LearnedPerception, SemanticMemory, ObservedOccupancy, RGBDObservation


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def initialize_compact_arm(base):
    """Initial condition only. Replay uses the identical pre-step posture."""
    q = base.agent.robot.get_qpos().clone()
    q[:, [5, 7, 8, 9, 10, 11, 12]] = torch.tensor(
        [-1.2, 1.3, 0., 1.8, 0., 1.2, 0.], device=q.device, dtype=q.dtype)
    base.agent.robot.set_qpos(q)
    base.agent.controller.reset()


class SimulatorIO:
    """Robot-only interface; the task actor and evaluator are owned elsewhere."""
    def __init__(self, env):
        self.env = env
        self.base = env.unwrapped
        self.robot = self.base.agent
        self.origin = array(self.robot.robot.pose.p)[0, :2].copy()
        self.step_id = 0
        self.frame_id = 0
        self.trace = []
        self.before_step = None
        self.after_step = None
        q = array(self.robot.robot.get_qpos())[0]
        self.arm = q[[5, 7, 8, 9, 10, 11, 12]].copy()
        self.grasp_seed_arm = self.arm.copy()
        self.body = q[[4, 6, 3]].copy()
        self.grip = 1.
        self.display_sensor = "fetch_head"
        self.overview_heading = float(self.pose()[2])
        self.stabilize_stationary_base = False
        self._stationary_anchor = None

    def pose(self):
        q = array(self.robot.robot.get_qpos())[0]
        return np.r_[self.origin + q[:2], q[2]].astype(float)

    def command(self, forward=0., yaw_rate=0., grip=None):
        if self.robot.control_mode == "pd_ee_delta_pose":
            return self.command_ee(grip=grip,forward=forward,yaw_rate=yaw_rate)
        forward,yaw_rate=self.stationary_feedback(forward,yaw_rate)
        if grip is not None:
            self.grip = grip
        q = array(self.robot.robot.get_qpos())[0]
        action = np.zeros(13, dtype=np.float32)
        action[:7] = self.arm
        action[7] = self.grip
        action[8:11] = np.clip((self.body - q[[4, 6, 3]]) / .1, -1., 1.)
        action[11:] = (forward, yaw_rate / 3.14)
        if self.before_step is not None:
            self.before_step()
        self.env.step(action)
        self.step_id += 1
        self.trace.append(dict(step=self.step_id, base_xyyaw=self.pose().tolist(), action=action.tolist(),
            control_mode=self.robot.control_mode,tcp_xyz=array(self.robot.tcp_pose.p)[0].tolist()))
        if self.after_step is not None:
            self.after_step()

    def command_ee(self, delta=None, grip=None, forward=0.,yaw_rate=0.):
        forward,yaw_rate=self.stationary_feedback(forward,yaw_rate)
        if grip is not None:
            self.grip = grip
        q = array(self.robot.robot.get_qpos())[0]
        action = np.zeros(12,dtype=np.float32)
        if delta is not None:
            action[:6] = delta
        action[6] = self.grip
        action[7:10] = np.clip((self.body-q[[4,6,3]])/.1,-1.,1.)
        action[10:] = (forward,yaw_rate/3.14)
        if self.before_step:
            self.before_step()
        self.env.step(action[None,:])
        self.step_id += 1
        self.trace.append(dict(step=self.step_id,base_xyyaw=self.pose().tolist(),action=action.tolist(),
            control_mode=self.robot.control_mode,tcp_xyz=array(self.robot.tcp_pose.p)[0].tolist()))
        if self.after_step:
            self.after_step()

    def stationary_feedback(self, forward, yaw_rate):
        """Odometry feedback counters slow PhysX creep under a zero-velocity
        command. It applies real base controls, never state correction."""
        if not self.stabilize_stationary_base:
            return forward,yaw_rate
        if abs(forward)>1e-8 or abs(yaw_rate)>1e-8:
            self._stationary_anchor=None
            return forward,yaw_rate
        pose=self.pose()
        if self._stationary_anchor is None:
            self._stationary_anchor=pose.copy()
        delta=self._stationary_anchor[:2]-pose[:2]
        heading=np.array([math.cos(pose[2]),math.sin(pose[2])])
        angle=math.atan2(math.sin(self._stationary_anchor[2]-pose[2]),
                         math.cos(self._stationary_anchor[2]-pose[2]))
        return float(np.clip(2*np.dot(delta,heading),-.05,.05)),float(np.clip(2*angle,-.10,.10))

    def move_tcp(self, goal, tolerance=.015, steps=240, target_rotation=None):
        """Measured TCP servo. No target actor pose enters this controller."""
        from scipy.spatial.transform import Rotation

        for _ in range(steps):
            error=np.asarray(goal)-array(self.robot.tcp_pose.p)[0]
            residual=float(np.linalg.norm(error))
            root_rotation=array(self.robot.torso_lift_link.pose.to_transformation_matrix())[0,:3,:3]
            rotvec=np.zeros(3)
            if target_rotation is not None:
                current=array(self.robot.tcp_pose.to_transformation_matrix())[0,:3,:3]
                relative=target_rotation@current.T
                rotvec=root_rotation.T@Rotation.from_matrix(relative).as_rotvec()
            if residual<tolerance and np.linalg.norm(rotvec)<.06:
                return residual
            local=root_rotation.T@error
            # ManiSkill 3.0.1 scales normalized rotational commands by
            # config.rot_lower (-0.1), not +rot_upper. Respect the installed
            # controller's actual convention; a positive gain here otherwise
            # produces rotational positive feedback and wrist-limit failure.
            rotation_scale=float(self.robot.controller.controllers["arm"].config.rot_lower)
            rotation_action=np.clip(.2*rotvec/rotation_scale,-.12,.12)
            self.command_ee(np.r_[np.clip(local*3.,-.08,.08),rotation_action])
        return float(np.linalg.norm(np.asarray(goal)-array(self.robot.tcp_pose.p)[0]))

    def restore_grasp_seed(self, grip=1.):
        if self.robot.control_mode != "pd_joint_pos":
            self.robot.set_control_mode("pd_joint_pos")
            self.robot.controller.reset()
        initial=array(self.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]]
        if grip<0:
            # Lift the folded upper arm before extending it. The fingers stay
            # closed throughout this physical joint-space transition.
            raised=initial.copy()
            raised[1]=min(initial[1],-.55)
            for alpha in np.linspace(0,1,160):
                self.arm=initial*(1-alpha)+raised*alpha
                self.command(grip=grip)
            initial=array(self.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]]
        for alpha in np.linspace(0,1,220 if grip<0 else 100):
            self.arm=initial*(1-alpha)+self.grasp_seed_arm*alpha
            self.command(grip=grip)

    def hold_measured_arm(self):
        self.arm=array(self.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].copy()
        self.robot.set_control_mode("pd_joint_pos")
        self.robot.controller.reset()

    def position_torso_for_grasp(self, target_z):
        # Use the available torso degree of freedom before Cartesian descent.
        # The raised rest seed alone cannot reach every table surface.
        q=array(self.robot.robot.get_qpos())[0]
        tcp_z=float(array(self.robot.tcp_pose.p)[0,2])
        desired=q[3]+(target_z+.10-tcp_z)
        limits=array(self.robot.robot.get_qlimits())[0,3]
        self.body[2]=float(np.clip(desired,limits[0]+.01,limits[1]-.01))
        for _ in range(90):
            self.command()

    def turn(self, yaw):
        for _ in range(450):
            error = math.atan2(math.sin(yaw-self.pose()[2]), math.cos(yaw-self.pose()[2]))
            if abs(error) < .025:
                return
            self.command(yaw_rate=float(np.clip(1.8*error, -.6, .6)))
        raise RuntimeError("Physical yaw control timed out")

    def prepare_scan(self):
        # Fetch's official raised rest arm occludes much of the head image.
        # Move the shoulder to the side through joint PD, never qpos setters.
        initial = self.arm.copy()
        tucked = initial.copy()
        tucked[:] = [-1.30, .80, 0., 1.70, 0., 1.40, 0.]
        for alpha in np.linspace(0., 1., 60):
            self.arm = initial*(1-alpha) + tucked*alpha
            self.command()

    def capture(self, sensor_name="fetch_head", *, keyframe=True):
        self.base.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.base.capture_sensor_data()
        sensor = self.base._sensors[sensor_name]
        # Crucially, neither segmentation nor ground-truth world-position
        # images are requested or transmitted to the learned agent.
        data = sensor.get_obs(rgb=True, depth=True, position=False, segmentation=False)
        params = sensor.get_params()
        rgb = array(data["rgb"])[0, ..., :3].astype(np.uint8)
        depth = array(data["depth"])[0, ..., 0].astype(np.float32) / 1000.
        extrinsic = array(params["extrinsic_cv"])[0]
        homogeneous = np.eye(4, dtype=np.float32)
        homogeneous[:3, :4] = extrinsic[:3, :4]
        # Visualization is a read-only observer, not part of the policy's
        # observation-ID clock. Turning recording on must not change recency
        # scores, rejection timestamps, or which stored observation is chosen.
        if keyframe:
            self.frame_id += 1
        return RGBDObservation(
            self.frame_id, self.step_id, sensor_name, rgb, depth,
            array(params["intrinsic_cv"])[0], np.linalg.inv(homogeneous), self.pose(),
        )

    def overview(self):
        # Keep the world viewing direction fixed as the robot turns, instead
        # of making the whole room spin with the base yaw.
        xy, yaw = self.pose()[:2], self.overview_heading
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        side = np.array([-heading[1], heading[0]])
        # Keep the overview close to the actual base footprint; the previous
        # 1.55 m side offset could put this display-only camera behind a wall.
        # This camera never supplies observations or decisions to the policy.
        eye = [*(xy - .25*heading - .45*side), 1.95 if getattr(self,"ceiling_safe_overview",False) else 2.45]
        look = [*(xy + .20*heading), .75]
        camera = self.base._human_render_cameras["render_camera"].camera
        if getattr(self,"ceiling_safe_overview",False):
            camera.set_fovy(1.25)
        camera.set_local_pose(sapien_utils.look_at(eye, look).sp)
        rgb = array(self.base.render_rgb_array("render_camera"))[0]
        return rgb.astype(np.uint8)


def make_env(scene, width=320, height=240):
    if scene.startswith("ProcTHOR-"):
        from procthor_fetch_scene import selected_procthor_builder
        builder, index = selected_procthor_builder(scene), 0
    else:
        builder, index = "ArchitecTHOR", SCENE_IDS.index(scene)
    return gym.make(
        "SceneManipulation-v1", scene_builder_cls=builder, robot_uids="fetch",
        build_config_idxs=[index], num_envs=1, obs_mode="state",
        reward_mode="none", render_mode="rgb_array", control_mode="pd_joint_pos",
        sim_backend="physx_cpu", render_backend="cpu", max_episode_steps=40000,
        sensor_configs=dict(width=width, height=height,
                            fetch_head=dict(fov=1.10), fetch_hand=dict(fov=1.50)),
        human_render_camera_configs=dict(width=960, height=540, fov=1.05, near=.05, far=100),
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_asset_target(base, xy, asset_name):
    """Textured official household asset on a separate dynamic support.

    The visual and convex collision mesh are the same official asset. Scale is
    fixed at 0.8 for the apple (about 8 cm wide), without a grasp-assist shoulder.
    All poses are set once at construction; later motion is under PhysX force.
    """
    asset = Path(os.environ["MS_ASSET_DIR"]) / "data/scene_datasets/ai2thor/ai2thorhab-uncompressed/assets/objects" / f"{asset_name}.glb"
    if not asset.is_file():
        raise FileNotFoundError(asset)
    builder = base.scene.create_actor_builder()
    floor = sapien.physx.PhysxMaterial(.02, .01, 0.)
    table = sapien.physx.PhysxMaterial(1., .8, 0.)
    for z, half, material in ((.02, [.08,.08,.02], floor),
                             (.455, [.025,.025,.435], table),
                             (.90, [.15,.15,.02], table)):
        builder.add_box_collision(pose=sapien.Pose([0,0,z]), half_size=half, material=material, density=150)
        builder.add_box_visual(pose=sapien.Pose([0,0,z]), half_size=half, material=[.12,.35,.82,1.])
    builder.set_initial_pose(sapien.Pose([*xy,.001]))
    support = builder.build(name="learned_dynamic_support")
    support.set_locked_motion_axes([False,False,False,True,True,True])
    support.linear_damping = 3.
    object_builder = base.scene.create_actor_builder()
    scale = [.8,.8,.8]
    rotation = sapien.Pose(q=[math.sqrt(.5),math.sqrt(.5),0,0])
    object_builder.add_visual_from_file(str(asset), pose=rotation, scale=scale)
    object_builder.add_convex_collision_from_file(str(asset), pose=rotation, scale=scale,
                                                  material=table, density=700)
    object_builder.set_initial_pose(sapien.Pose([*xy,.966]))
    target = object_builder.build(name="learned_household_target")
    target.linear_damping = 2.
    target.angular_damping = 2.
    return support, target


def exclude_existing_apples(base):
    """Initialization-only task construction, explicitly recorded in manifests.

    An 'apple' command is ambiguous if another genuine apple is in the official
    room. The single-target protocol excludes only exact Apple asset instances;
    source asset files remain untouched, and a fresh environment restores them.
    """
    import re

    removed=[]
    for name,actor in list(base.scene.actors.items()):
        if re.search(r"(?:^|/)Apple_[0-9]+_[0-9]+$", name):
            removed.append(dict(name=name,initial_pose=array(actor.pose.raw_pose).tolist()))
            # ManiSkill 3.0.1's remove_actor helper assumes component-backed
            # _objs, while these official scene actors hold Entities directly.
            # Resolve exactly that recorded actor; never touch asset files.
            for entity in actor._objs:
                base.scene.sub_scenes[0].remove_entity(getattr(entity,"entity",entity))
            base.scene.actors.pop(name)
    return removed


def build_delivery_bin(base, xy, floor_height=.68):
    builder=base.scene.create_actor_builder()
    material=sapien.physx.PhysxMaterial(1.,.8,0.)
    parts=[([0,0,floor_height/2],[.25,.25,floor_height/2])]
    for axis in (0,1):
        for sign in (-1,1):
            position=[0.,0.,floor_height+.055]
            position[axis]=sign*.24
            size=[.25,.25,.055]
            size[axis]=.01
            parts.append((position,size))
    for position,size in parts:
        builder.add_box_collision(pose=sapien.Pose(position),half_size=size,material=material)
        builder.add_box_visual(pose=sapien.Pose(position),half_size=size,material=[.12,.50,.28,1.])
    builder.set_initial_pose(sapien.Pose([*xy,0.]))
    return builder.build_static(name="known_delivery_bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENE_IDS, default=SCENE_IDS[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-xy", type=float, nargs=2, default=(-1.6, 0.0))
    parser.add_argument("--queries", nargs="+", default=["orange box", "bottle", "apple"])
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=.15)
    parser.add_argument("--object-asset", default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve prior attempts; choose a new output: {args.output}")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    started = time.monotonic()
    perception = LearnedPerception(threshold=args.threshold)
    memories = {name: SemanticMemory(perception, dynamic=dynamic)
                for name, dynamic in (("dynamic", True), ("static", False))}
    occupancy = ObservedOccupancy()
    env = make_env(args.scene)
    io = None
    reports = []
    try:
        env.reset(seed=args.seed)
        if args.object_asset:
            support, target = build_asset_target(env.unwrapped, np.asarray(args.target_xy), args.object_asset)
        else:
            support, target = build_moving_target(env.unwrapped, np.asarray(args.target_xy))
        io = SimulatorIO(env)
        io.body[1] = .30
        io.prepare_scan()
        for _ in range(20):
            io.command()
        initial_pose = io.pose().copy()
        for i in range(args.views):
            io.turn(initial_pose[2] + i*2*math.pi/args.views)
            for _ in range(5):
                io.command()
            obs = io.capture()
            stem = f"view_{i:02d}"
            obs.save(args.output / f"{stem}.npz")
            cv2.imwrite(str(args.output / f"{stem}_rgb.png"), cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR))
            overview = io.overview()
            cv2.imwrite(str(args.output / f"{stem}_overview.png"), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
            dense = perception.dense(obs)
            updates = {name: memory.integrate(obs, dense) for name, memory in memories.items()}
            occupancy.integrate(obs)
            detections = {query: [asdict(d) for d in perception.detect(obs, query)] for query in args.queries}
            # Evaluation is performed only AFTER the agent has returned results.
            evaluator = dict(target_xyz=array(target.pose.p)[0].tolist(), support_xyz=array(support.pose.p)[0].tolist())
            report = dict(view=i, frame_id=obs.frame_id, sim_step=io.step_id, updates=updates,
                          detections=detections, evaluator=evaluator,
                          depth_min=float(obs.depth_m.min()), depth_max=float(obs.depth_m.max()),
                          observed_free=int((occupancy.known == 1).sum()),
                          observed_occupied=int((occupancy.known == -1).sum()))
            reports.append(report)
            print(json.dumps(report), flush=True)
        retrieval = {}
        for name, memory in memories.items():
            retrieval[name] = {}
            for query in args.queries:
                detection, candidate = memory.retrieve(query)
                retrieval[name][query] = dict(candidate=candidate,
                    detection=asdict(detection) if detection else None,
                    ranked_observations=memory.ranked_observations(query))
        np.savez_compressed(args.output / "occupancy.npz", known=occupancy.known,
                            origin=occupancy.origin, resolution=occupancy.resolution)
        result = dict(status="completed_development_probe", scene=args.scene,
                      object_asset=args.object_asset, queries=args.queries, detector_threshold=args.threshold,
                      boundary="Learned perception/memory scan only; no end-to-end task result",
                      ground_truth_not_agent_input=True, segmentation_requested=False,
                      views=reports, retrieval=retrieval, env_steps=io.step_id,
                      initial_pose=initial_pose.tolist(), final_pose=io.pose().tolist(),
                      base_drift_m=float(np.linalg.norm(io.pose()[:2]-initial_pose[:2])),
                      wall_time_s=time.monotonic()-started,
                      model_lock=json.loads((ROOT / ".dream_model_cache/dream_models.lock.json").read_text()))
        (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps(dict(error=repr(error), views=reports), indent=2)+"\n")
        raise
    finally:
        if io is not None:
            (args.output / "actions.json").write_text(json.dumps(io.trace)+"\n")
        env.close()
        hashes = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "sha256.json"}
        hashes["source:dream_learned_core.py"] = sha(Path(__file__).with_name("dream_learned_core.py"))
        hashes["source:maniskill_learned_probe.py"] = sha(Path(__file__))
        (args.output / "sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")


if __name__ == "__main__":
    main()
