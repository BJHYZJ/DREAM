#!/usr/bin/env python3
"""Development pilot: learned memory -> physical relocation -> live focus.

This executes observation-grounded retrieval, navigation and optional physical
grasp/carry/place. Success is independently scored, never assumed. The task
scheduler is separate from the policy; it cannot pass the endpoint to the agent.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

# Configure simulator/model environment before importing heavy dependencies.
from maniskill_learned_probe import (
    ROOT, SCENE_IDS, SimulatorIO, make_env, build_asset_target, exclude_existing_apples,
    build_delivery_bin, array, sha,
)
import cv2
import numpy as np
import torch

from dream_learned_core import (
    LearnedPerception, SemanticMemory, ObservedOccupancy, select_live_or_memory_target,
)
from learned_video import EvidenceVideo, map_image
from houseexpo_cross_room import weighted_distances, _reconstruct_path
from depth_grasp import grasp_center_from_detection


class ExternalMove:
    """Environment-only force controller, triggered by absolute simulation step."""
    def __init__(self, support, endpoint, start_step=480, waypoints=None):
        self.support = support
        self.endpoint = np.asarray(endpoint, dtype=float)
        self.start_step = start_step
        self.done_step = None
        self.initial = array(support.pose.p)[0].copy()
        self.trace = []
        self.waypoints=[np.asarray(p,dtype=float) for p in (waypoints or [])]+[self.endpoint]
        self.waypoint_index=0

    def before_step(self, step):
        if step < self.start_step or self.done_step is not None:
            return
        position = array(self.support.pose.p)[0, :2]
        if self.waypoint_index<len(self.waypoints)-1 and np.linalg.norm(self.waypoints[self.waypoint_index]-position)<.05:
            self.waypoint_index+=1
        delta = self.waypoints[self.waypoint_index]-position
        distance = np.linalg.norm(delta)
        velocity = array(self.support.get_linear_velocity())[0, :2]
        desired = delta / max(distance, 1e-9)*min(.05, distance*.8)
        force = np.clip(300*(desired-velocity),-30,30)
        self.support.apply_force(np.r_[force,0.].astype(np.float32))
        self.trace.append(dict(step=step,position=position.tolist(),force=force.tolist(),waypoint=self.waypoint_index))
        if self.waypoint_index==len(self.waypoints)-1 and distance < .02 and np.linalg.norm(velocity) < .01:
            self.done_step = step
            # The cart parks; its separate payload remains fully dynamic.
            self.support.set_locked_motion_axes([True]*6)


def motion_detection_changed(tracked_point, detection, distance=.5):
    """Only discovery/displacement interrupts an observed route chunk."""
    return detection is not None and (tracked_point is None or
        np.linalg.norm(np.asarray(detection.point_world)[:2]-np.asarray(tracked_point)[:2])>distance)


class LearnedSearchPilot:
    """Policy receives only the robot interface and observation-owned memories."""
    def __init__(self, io, perception, output, query, dynamic):
        self.io, self.perception, self.output, self.query = io, perception, output, query
        self.memory = SemanticMemory(perception, dynamic=dynamic)
        self.occupancy = ObservedOccupancy()
        self.phase = "Build memory"
        self.cached = None
        self.cached_detection=None
        self.events = []
        self.current_detection = None
        self.last_obs = None
        self.grasp_motion_attempted = False
        self.placement_motion_attempted = False

    def event(self, name, **kwargs):
        row = dict(step=self.io.step_id,event=name,**kwargs)
        self.events.append(row)
        print(json.dumps(row),flush=True)
        # Flush incrementally, including failures interrupted during long runs.
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")

    def observe(self, sensor_name="fetch_head"):
        self.io.display_sensor=sensor_name
        obs = self.io.capture(sensor_name)
        obs.save(self.output / f"observation_{obs.frame_id:05d}.npz")
        update = self.memory.integrate(obs)
        spheres=[]
        for link in self.io.robot.robot.get_links():
            if any(part in link.name for part in ("arm", "forearm", "wrist", "gripper", "shoulder", "elbow")):
                spheres.append((array(link.pose.p)[0],.10))
        if self.io.grip<0:
            spheres.append((array(self.io.robot.tcp_pose.p)[0],.13))
        if hasattr(self,"robot_filter_spheres"):
            spheres.extend(self.robot_filter_spheres())
        self.occupancy.integrate(obs,self_spheres=spheres)
        detections = self.perception.detect(obs,self.query)
        self.current_detection = detections[0] if detections else None
        self.last_obs = obs
        self.event("observation",frame_id=obs.frame_id,memory=update,
                   detections=[asdict(d) for d in detections])
        return self.current_detection

    def turn_to(self,yaw):
        """Retry a blocked rotation only after a short observed-free retreat."""
        for attempt in range(3):
            try:
                self.io.turn(yaw)
                return
            except RuntimeError as error:
                if "yaw control timed out" not in str(error) or attempt==2:
                    raise
                self.event("blocked_turn_recovery",attempt=attempt)
                self.observe()
                pose=self.io.pose()
                direction=np.array([math.cos(pose[2]),math.sin(pose[2])])
                safe=self.occupancy.traversable()
                sign=None
                for candidate in (-1.,1.):
                    points=pose[:2]+candidate*np.linspace(0,.20,10)[:,None]*direction
                    cells=self.occupancy.cells(points)
                    if self.occupancy.inside(cells).all() and safe[cells[:,0],cells[:,1]].all():
                        sign=candidate
                        break
                if sign is None:
                    raise
                self.phase="Recover clearance"
                for _ in range(100):
                    if np.linalg.norm(self.io.pose()[:2]-pose[:2])>=.20:
                        break
                    self.io.command(forward=sign*.05)
                self.observe()

    def survey(self):
        from dream.dynamic_memory import initial_scan_yaws

        self.io.prepare_scan()
        self.io.body[1] = .30
        yaw = self.io.pose()[2]
        for target_yaw in initial_scan_yaws(yaw):
            self.turn_to(target_yaw)
            for _ in range(5):
                self.io.command()
            self.observe()

    def retrieve(self, use_live=False):
        result,candidate = self.memory.retrieve(self.query)
        live = self.current_detection if use_live else None
        result = select_live_or_memory_target(live,result)
        source="live" if live is not None else "memory"
        if result is None and self.cached_detection is not None:
            # A look away from the target is not evidence of target removal.
            # Keep the last verified track until an actual focused check fails.
            # Identical tracking behavior is available to the static baseline.
            result=self.cached_detection
            source="last_verified_track"
        self.cached_detection=result
        self.cached = None if result is None else np.asarray(result.point_world)
        self.event("memory_retrieval",candidate=candidate,detection=asdict(result) if result else None,
                   selected_from=source)
        return result

    def missing_location_observed(self,point,observation):
        return True  # Historical adapter behavior; new protocol overrides it.

    def focus_camera_height(self):
        return 1.45  # Historical fixed-height calibration.

    def focus(self):
        if self.cached is None:
            return None
        self.phase = "Verify remembered location"
        old = self.cached.copy()
        old_detection=self.cached_detection
        old_max_id = max(self.memory.observations)
        base = self.io.pose()
        delta = old[:2]-base[:2]
        bearing=math.atan2(delta[1],delta[0])
        turned=self.turn_to(bearing)
        # Pitch from measured camera height and retrieved point, not actor pose.
        self.io.body[0] = float(np.clip(math.atan2(math.sin(bearing-self.io.pose()[2]),
            math.cos(bearing-self.io.pose()[2])),-1.3,1.3)) if turned is False else 0.
        self.io.body[1] = float(np.clip(math.atan2(self.focus_camera_height()-old[2],np.linalg.norm(delta)), -.3,.9))
        for _ in range(12):
            self.io.command()
        current = self.observe()
        displaced = current is not None and np.linalg.norm(np.asarray(current.point_world)[:2]-old[:2]) > .5
        missing = current is None and self.missing_location_observed(old,self.last_obs)
        if missing or displaced:
            self.memory.reject(old[:2],self.last_obs.frame_id if missing else old_max_id,
                               observation_id=old_detection.observation_id if old_detection else None,
                               query=self.query)
            self.cached_detection=None
            self.cached=None
        self.event("focused_verification",old_point=old.tolist(),
                   fresh_detection=asdict(current) if current else None,
                   missing=missing,displaced=bool(displaced),
                   dynamic_updates_enabled=self.memory.dynamic)
        return current

    def scan_sweep(self):
        """The same active observation recovery is available to both variants."""
        self.phase="Active observation"
        found=[]
        for pan in (-.45,0.,.45):
            self.io.body[0]=pan
            for _ in range(12):
                self.io.command()
            result=self.observe()
            if result is not None:
                found.append(result)
        self.io.body[0]=0.
        for _ in range(12):
            self.io.command()
        # The selected detection is fresh within this stationary sweep, not an
        # evaluator answer or a remembered pre-relocation coordinate.
        self.current_detection=max(found,key=lambda d:d.score) if found else None
        self.event("head_camera_scan_complete",views=3,sensor="fetch_head")
        return self.current_detection

    def wrist_sweep(self):
        """Physically change the arm camera view and fuse its calibrated RGB-D.

        These are modest, joint-limited moves around the tucked posture, not
        camera pose setters. Both memory variants receive the same scan action.
        """
        if self.io.grip<0:
            return
        self.phase="Arm-camera scan"
        saved=self.io.arm.copy()
        found=[]
        for shoulder in (-1.50,-1.05):
            initial=self.io.arm.copy()
            desired=saved.copy()
            desired[0]=shoulder
            desired[5]=.90
            self.io.display_sensor="fetch_hand"
            for alpha in np.linspace(0,1,24):
                self.io.arm=initial*(1-alpha)+desired*alpha
                self.io.command()
            result=self.observe("fetch_hand")
            if result is not None:
                found.append(result)
        initial=self.io.arm.copy()
        for alpha in np.linspace(0,1,24):
            self.io.arm=initial*(1-alpha)+saved*alpha
            self.io.command()
        result=self.observe()
        if result is not None:
            found.append(result)
        self.current_detection=max(found,key=lambda d:d.score) if found else None
        self.event("arm_camera_scan_complete",views=2)

    def move_chunk(self, path, max_cells=12, stop_on_detection=True, speed=.14):
        """Closed-loop rotation, then a short straight line; no coordinate hints."""
        # Walk only to the next line-of-sight point within the observed route.
        # Keep all intervening raster samples safe rather than skipping corners.
        if not path:
            return False
        tracked_point = None if self.cached is None else self.cached.copy()
        safe = self.occupancy.traversable()
        chosen = path[0]
        base = self.io.pose()[:2]
        for cell in path[1:max_cells+1]:
            point = self.occupancy.world(cell)
            samples = self.occupancy.cells(np.linspace(base,point,50))
            if not (self.occupancy.inside(samples).all() and safe[samples[:,0],samples[:,1]].all()):
                break
            chosen = cell
        goal = self.occupancy.world(chosen)
        if np.linalg.norm(goal-base) < .10:
            return False
        self.turn_to(math.atan2(*(goal-base)[::-1]))
        start = self.io.pose()[:2].copy()
        best_distance = float(np.linalg.norm(goal-start))
        stagnant = 0
        max_steps=max(240,int(np.linalg.norm(goal-start)/max(.02,speed)*self.io.base.control_freq*1.5))
        for n in range(max_steps):
            delta = goal-self.io.pose()[:2]
            if np.linalg.norm(delta) < .08:
                break
            yaw = self.io.pose()[2]
            error = math.atan2(math.sin(math.atan2(delta[1],delta[0])-yaw),
                               math.cos(math.atan2(delta[1],delta[0])-yaw))
            self.io.command(forward=0. if abs(error)>.18 else min(speed,np.linalg.norm(delta)),
                            yaw_rate=float(np.clip(1.8*error,-.4,.4)))
            if np.linalg.norm(delta) < best_distance-.01:
                best_distance=float(np.linalg.norm(delta))
                stagnant=0
            else:
                stagnant+=1
            if n and n%40==0:
                self.observe()
                # Seeing the same tracked target is not a new event. Stopping
                # on every positive frame consumed the search budget without
                # progressing and could hide a physical stall. Both variants
                # still interrupt immediately on discovery/displacement.
                if stop_on_detection and motion_detection_changed(tracked_point,self.current_detection):
                    self.event("live_detection_during_motion",detection=asdict(self.current_detection))
                    return True
                if stagnant > 100:
                    self.event("controller_stall",goal=goal.tolist())
                    return False
        self.observe()
        self.event("executed_observed_route_chunk",goal=goal.tolist(),base=self.io.pose().tolist())
        return True

    def navigate(self, budget=20, wrist_scans=False):
        self.phase = "Search with observed memory"
        for iteration in range(budget):
            result = self.retrieve(use_live=True)
            if result is not None:
                distance = np.linalg.norm(self.cached[:2]-self.io.pose()[:2])
                if distance < 2.0:
                    self.focus()
                    result = self.retrieve(use_live=True)
                if result is not None and self.current_detection is not None:
                    xyz = np.asarray(self.current_detection.point_world)
                    if np.linalg.norm(xyz[:2]-self.io.pose()[:2]) < .67:
                        self.event("visual_approach_complete",detection=asdict(self.current_detection))
                        return True
            safe = self.occupancy.traversable()
            start = tuple(self.occupancy.cells(self.io.pose()[:2]))
            if not safe[start]:
                self.event("base_cell_not_depth_navigable")
                return False
            distances,parents = weighted_distances(safe,start)
            goal = None
            if self.cached is not None:
                cells = np.argwhere(np.isfinite(distances) & safe)
                offsets = np.linalg.norm(self.occupancy.world(cells)-self.cached[:2],axis=1)
                candidates = (offsets>=.52)&(offsets<=.60)
                if candidates.any():
                    cells = cells[candidates]
                    goal = tuple(cells[np.argmin(distances[cells[:,0],cells[:,1]])])
                else:
                    # A target beyond the currently observed docking ring is
                    # approached through known free space, not replaced by an
                    # unrelated exploration frontier. Unknown cells remain
                    # forbidden and new depth observations extend the route.
                    current_distance=np.linalg.norm(self.cached[:2]-self.io.pose()[:2])
                    partial=(offsets>=.52)&(offsets<current_distance-.15)
                    if partial.any():
                        candidates=cells[partial]
                        cost=offsets[partial]+.02*distances[candidates[:,0],candidates[:,1]]
                        goal=tuple(candidates[np.argmin(cost)])
            if goal is None:
                goal,parents = self.occupancy.frontier(self.io.pose()[:2],
                    self.occupancy.semantic_field(self.memory,self.query))
            if goal is None:
                self.event("no_observed_reachable_goal",iteration=iteration)
                return False
            path = _reconstruct_path(parents,start,goal)
            executed=len(path)>=2 and self.move_chunk(path)
            if executed and wrist_scans:
                self.wrist_sweep()
            elif executed:
                self.scan_sweep()
            if not executed:
                self.event("route_execution_incomplete",iteration=iteration)
                if self.scan_sweep() is not None:
                    continue
                # A route ending at the current cell is a request to look
                # around, not an instant failure assigned to a stale baseline.
                frontier,frontier_parents=self.occupancy.frontier(self.io.pose()[:2],
                    self.occupancy.semantic_field(self.memory,self.query))
                if frontier is not None:
                    # move_chunk/recovery may have physically moved the base.
                    # The new parent tree is rooted at its CURRENT cell.
                    recovery_start=tuple(self.occupancy.cells(self.io.pose()[:2]))
                    self.move_chunk(_reconstruct_path(frontier_parents,recovery_start,frontier))
        return False

    def propose_grasp(self, observation, detection):
        return grasp_center_from_detection(observation,detection)

    def attempt_grasp(self):
        """Geometry-adapted Fetch grasp using only learned RGB-D localization.

        This executes an attempt; bilateral contact and lift success are scored
        independently by the evaluator. No grasp attachment or actor setter.
        """
        self.focus()
        detection=self.current_detection
        if detection is None:
            self.event("grasp_aborted_no_fresh_detection")
            return False
        target,fit=self.propose_grasp(self.memory.observations[detection.observation_id],detection)
        self.perceived_payload_radius=float(fit["radius_m"])
        self.perceived_payload_height=float(fit.get("height_m",2*self.perceived_payload_radius))
        self.event("depth_grasp_center",center_world=target.tolist(),**fit)
        delta=target[:2]-self.io.pose()[:2]
        yaw=math.atan2(delta[1],delta[0])
        self.turn_to(yaw)
        self.phase="Prepare grasp"
        self.io.restore_grasp_seed()
        self.io.position_torso_for_grasp(target[2])
        self.event("torso_workspace_alignment",torso_target_m=float(self.io.body[2]),
                   achieved_tcp_xyz=array(self.io.robot.tcp_pose.p)[0].tolist())
        self.io.robot.set_control_mode("pd_ee_delta_pose")
        self.io.robot.controller.reset()
        # Fetch.tcp_pose is the midpoint of finger-link origins. The old
        # cuboid's elevated shoulder offset is not applicable to this mesh.
        goal=target
        self.phase="Grasp"
        self.grasp_motion_attempted = True
        self.io.display_sensor="fetch_hand"
        world_yaw=np.array([[math.cos(yaw),-math.sin(yaw),0],
                           [math.sin(yaw),math.cos(yaw),0],[0,0,1]])
        vertical=world_yaw@np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        if "closing_axis_xy" in fit:
            closing=np.r_[fit["closing_axis_xy"],0.]
            closing/=np.linalg.norm(closing)
            downward=np.array([0.,0.,-1.])
            vertical=np.column_stack((downward,closing,np.cross(downward,closing)))
        pre=self.io.move_tcp(goal+[0,0,.14],target_rotation=vertical)
        contact=self.io.move_tcp(goal,tolerance=.007,target_rotation=vertical)
        self.event("grasp_tcp_approach",perceived_target=target.tolist(),
                   pregrasp_residual_m=pre,grasp_residual_m=contact,goal_tcp=goal.tolist(),
                   qpos=array(self.io.robot.robot.get_qpos())[0].tolist(),
                   tcp_matrix=array(self.io.robot.tcp_pose.to_transformation_matrix())[0].tolist())
        for grip in np.linspace(1.,-1.,40):
            self.io.command_ee(grip=float(grip))
        for _ in range(40):
            self.io.command_ee()
        self.event("gripper_closed",finger_positions=array(self.io.robot.robot.get_qpos())[0,-2:].tolist())
        self.phase="Lift"
        lift_goal=array(self.io.robot.tcp_pose.p)[0]+[0,0,.12]
        residual=self.io.move_tcp(lift_goal)
        for _ in range(20):
            self.io.command_ee()
        self.io.hold_measured_arm()
        for _ in range(20):
            self.io.command()
        self.event("lift_attempt_complete",tcp_residual_m=residual)
        gap=float(array(self.io.robot.robot.get_qpos())[0,-2:].sum())
        self.event("grip_retention_proprioception",finger_gap_m=gap)
        return gap > .015

    def prepare_carry(self):
        """Physically fold a held payload; no destination is needed here."""
        self.phase="Carry"
        self.io.display_sensor="fetch_head"
        if hasattr(self,"retreat"):
            if not self.retreat(.30) and not self.retreat(.15):
                self.event("carry_retreat_not_observed_safe_fold_in_place")
        else:
            retreat=self.io.pose()[:2].copy()
            for _ in range(160):
                if np.linalg.norm(self.io.pose()[:2]-retreat)>.30:
                    break
                self.io.command(forward=-.05)
        # Carry with a shorter measured reach so the extended grasp posture
        # does not sweep into furniture during base turns. This is a PD/IK
        # motion with the fingers still closed, not attachment or pose reset.
        current_tcp=array(self.io.robot.tcp_pose.p)[0]
        delta=current_tcp[:2]-self.io.pose()[:2]
        reach=np.linalg.norm(delta)
        carry_xy=self.io.pose()[:2]+delta/max(reach,1e-6)*min(reach,.38)
        self.io.robot.set_control_mode("pd_ee_delta_pose")
        self.io.robot.controller.reset()
        carry_residual=self.io.move_tcp(np.r_[carry_xy,max(current_tcp[2],1.18)])
        self.io.hold_measured_arm()
        self.event("compact_carry_posture",tcp_residual_m=carry_residual)
        if hasattr(self,"planner"):
            # Whole-arm fold, calibrated with a genuinely grasped payload and
            # full turns. Shortening TCP reach alone left the forearm at 0.63 m.
            # Fingers remain closed; all motion is executed by joint PD.
            self.io.body[2]=.385
            for _ in range(140):
                self.io.command()
            initial=self.io.arm.copy()
            desired=np.array([-1.2,1.3,0.,1.8,0.,1.2,0.])
            for alpha in np.linspace(0,1,260):
                self.io.arm=initial*(1-alpha)+desired*alpha
                self.io.command()
            for _ in range(60):
                self.io.command()
            self.event("held_arm_joint_fold",command=desired.tolist(),
                measured_arm=array(self.io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].tolist(),
                finger_gap_m=float(array(self.io.robot.robot.get_qpos())[0,-2:].sum()))
            from inspect_fetch_envelope import measure
            envelope=measure(self.io)
            measured_tcp=array(self.io.robot.tcp_pose.p)[0]
            payload_envelope=float(np.linalg.norm(measured_tcp[:2]-self.io.pose()[:2]))+self.perceived_payload_radius
            self.occupancy.radius=max(.40,envelope["max_radius"]+.04,payload_envelope+.04)
            self.event("measured_carry_envelope",robot=envelope,payload_radius_from_depth_m=self.perceived_payload_radius,
                       navigation_radius_m=self.occupancy.radius)
        self.observe()

    def carry_and_place(self, home_xy, bin_xy, bin_floor=.68):
        """Historical known-bin protocol; not the instruction-driven task."""
        self.prepare_carry()
        reached=False
        route_retries=0
        for iteration in range(80 if hasattr(self,"planner") else 24):
            current=self.io.pose()[:2]
            if np.linalg.norm(current-np.asarray(home_xy))<(.025 if hasattr(self,"planner") else .20):
                reached=True
                break
            safe=self.occupancy.traversable()
            start=tuple(self.occupancy.cells(current))
            goal=tuple(self.occupancy.cells(home_xy))
            distances,parents=weighted_distances(safe,start)
            if hasattr(self,"planner") and safe[start] and (not safe[goal] or not np.isfinite(distances[goal])):
                cells=np.argwhere(safe&np.isfinite(distances))
                error=np.linalg.norm(self.occupancy.world(cells)-np.asarray(home_xy),axis=1)
                nearby=error<.03
                if nearby.any():
                    choices=cells[nearby]
                    goal=tuple(choices[np.argmin(error[nearby])])
            if not safe[start] or not safe[goal] or not np.isfinite(distances[goal]):
                self.event("delivery_route_not_observed",iteration=iteration,
                    start_safe=bool(safe[start]),goal_safe=bool(safe[goal]),
                    reachable=bool(np.isfinite(distances[goal])))
                if hasattr(self,"planner") and route_retries<3:
                    route_retries+=1
                    self.look_around()
                    continue
                return False
            path=(self.planner.path(current,self.occupancy.world(goal)) if hasattr(self,"planner")
                  else _reconstruct_path(parents,start,goal))
            terminal=dict(terminal_xy=home_xy) if hasattr(self,"planner") else {}
            if not self.move_chunk(path,stop_on_detection=False,speed=.055,**terminal):
                self.event("delivery_controller_stall",iteration=iteration)
                if hasattr(self,"planner") and route_retries<8:
                    route_retries+=1
                    self.geometry_observe()
                    continue
                return False
            route_retries=0
            if float(array(self.io.robot.robot.get_qpos())[0,-2:].sum())<.015:
                self.event("grip_lost_by_proprioception")
                return False
        if not reached:
            return False
        self.phase="Place"
        self.placement_motion_attempted=True
        yaw=math.atan2(*(np.asarray(bin_xy)-self.io.pose()[:2])[::-1])
        self.turn_to(yaw)
        if hasattr(self,"planner"):
            self.io.restore_grasp_seed(grip=-1.)
            self.event("held_arm_prepared_for_place",finger_gap_m=float(array(self.io.robot.robot.get_qpos())[0,-2:].sum()))
        # Lower the torso while preserving the held joint configuration and
        # gripper command; do not reset to an open-hand navigation posture.
        current_tcp=array(self.io.robot.tcp_pose.p)[0]
        q=array(self.io.robot.robot.get_qpos())[0]
        self.io.body[2]=float(max(.01,q[3]+(bin_floor+.20-current_tcp[2])))
        for _ in range(100):
            self.io.command()
        self.io.robot.set_control_mode("pd_ee_delta_pose")
        self.io.robot.controller.reset()
        self.io.display_sensor="fetch_hand"
        # A task-defined opening, approached from above. Round-object radius
        # was inferred from depth earlier; the final release has clearance.
        release=np.r_[bin_xy,bin_floor+.11]
        vertical=None
        if hasattr(self,"planner"):
            # Use a point safely inside the known opening, slightly towards
            # the robot; no hidden target coordinate enters this choice.
            release[:2]-=.10*np.array([math.cos(yaw),math.sin(yaw)])
            world_yaw=np.array([[math.cos(yaw),-math.sin(yaw),0],
                               [math.sin(yaw),math.cos(yaw),0],[0,0,1]])
            vertical=world_yaw@np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        pre=self.io.move_tcp(release+[0,0,.13],target_rotation=vertical)
        residual=self.io.move_tcp(release,target_rotation=vertical)
        self.event("placement_tcp_approach",known_bin_xy=np.asarray(bin_xy).tolist(),
                   commanded_release_xyz=release.tolist(),pre_residual_m=pre,release_residual_m=residual)
        if hasattr(self,"planner"):
            # A receptacle is a region, not a single exact TCP point. Check
            # measured TCP against the task-known opening and RGB-D payload
            # radius, rather than releasing outside it after an IK timeout.
            tcp=array(self.io.robot.tcp_pose.p)[0]
            inside_xy=float(np.max(np.abs(tcp[:2]-bin_xy)))+self.perceived_payload_radius+.02<.23
            inside_z=bin_floor+.04<float(tcp[2])<bin_floor+.24
            self.event("release_opening_check",measured_tcp=tcp.tolist(),
                       horizontal_clearance_pass=bool(inside_xy),height_pass=bool(inside_z))
            if not (inside_xy and inside_z):
                self.event("placement_aborted_unreachable_opening",tcp_residual_m=residual)
                return False
        for grip in np.linspace(-1.,1.,40):
            self.io.command_ee(grip=float(grip))
        self.io.move_tcp(array(self.io.robot.tcp_pose.p)[0]+[0,0,.15])
        self.phase="Check placement"
        for _ in range(100):
            self.io.command_ee()
        self.event("placement_attempt_complete")
        return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scene",default=SCENE_IDS[0])
    parser.add_argument("--variant",choices=("dynamic","static"),default="dynamic")
    parser.add_argument("--target-xy",type=float,nargs=2,default=(-1.6,0.))
    parser.add_argument("--endpoint",type=float,nargs=2,default=(-.6,0.))
    parser.add_argument("--query",default="apple")
    parser.add_argument("--query-step",type=int,default=1100)
    parser.add_argument("--seed",type=int,default=17)
    parser.add_argument("--video",action="store_true")
    parser.add_argument("--navigate",action="store_true")
    parser.add_argument("--sensor-width",type=int,default=640)
    parser.add_argument("--threshold",type=float,default=.25)
    parser.add_argument("--grasp",action="store_true")
    parser.add_argument("--unique-target",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--model-set",choices=("base","production"),default="production")
    parser.add_argument("--deliver",action="store_true")
    parser.add_argument("--wrist-scans",action=argparse.BooleanOptionalAction,default=False,
                        help="Legacy optional arm-camera scans; disabled for head-camera navigation")
    parser.add_argument("--navigation-budget",type=int,default=20)
    parser.add_argument("--navigation-version",choices=("v2","astar"),default="v2")
    parser.add_argument("--disturbance-step",type=int,default=480)
    parser.add_argument("--task-json",type=Path)
    args=parser.parse_args()
    args.initial_compact_arm = args.navigation_version == "astar"
    task=json.loads(args.task_json.read_text()) if args.task_json else {}
    for key in ("scene","target_xy","endpoint","seed","disturbance_step","query_step"):
        if key in task:
            setattr(args,key,task[key])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    # Preserve the actual code before starting, so concurrent later edits
    # cannot silently change provenance of a long-running experiment.
    import shutil
    snapshot=args.output / "source_snapshot"
    snapshot.mkdir()
    for name in ("dream_learned_core.py","maniskill_learned_probe.py","maniskill_learned_dynamic.py","learned_video.py","depth_grasp.py","learned_evaluation.py",
                 "dream_fetch_navigation.py","maniskill_crossroom_policy.py","crossroom_evaluation.py","inspect_fetch_envelope.py","procthor_fetch_scene.py"):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    repo=Path(__file__).resolve().parents[1]
    for source in (repo/"src/dream").rglob("*.py"):
        destination=snapshot / source.relative_to(repo)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    (args.output / "configuration.json").write_text(json.dumps(
        {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2)+"\n")
    torch.set_num_threads(2)
    started=time.monotonic()
    perception=LearnedPerception(threshold=args.threshold,model_set=args.model_set)
    env=make_env(args.scene,width=args.sensor_width,height=3*args.sensor_width//4)
    recorder=None
    io=None
    policy=None
    evaluation=[]
    try:
        env.reset(seed=args.seed)
        if "spawn_xy" in task:
            # Environment construction before the first recorded physical step.
            # This cannot be called by the policy or during an episode.
            from mani_skill.utils.structs.pose import Pose
            env.unwrapped.agent.robot.set_pose(Pose.create_from_pq(p=[[*task["spawn_xy"],.02]]))
        excluded=exclude_existing_apples(env.unwrapped) if args.unique_target else []
        (args.output / "scene_task_edits.json").write_text(json.dumps(dict(excluded_initial_actors=excluded,
            reason="Unique language-category target, initialization only; original assets unchanged"),indent=2)+"\n")
        support,target=build_asset_target(env.unwrapped,np.asarray(args.target_xy),"Apple_1")
        disturbance=ExternalMove(support,args.endpoint,start_step=args.disturbance_step,
                                 waypoints=task.get("disturbance_waypoints"))
        io=SimulatorIO(env)
        if args.initial_compact_arm:
            from maniskill_learned_probe import initialize_compact_arm
            initialize_compact_arm(env.unwrapped)
            # Keep the native rest seed for later grasp IK, but initialize
            # navigation with a documented folded posture before step one.
            io.arm=array(io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].copy()
            io.stabilize_stationary_base=True
        home_xy=io.pose()[:2].copy()
        bin_xy=np.asarray(task.get("bin_xy",home_xy+[-.70,0.]),dtype=float)
        (args.output / "environment_task.json").write_text(json.dumps(task,indent=2)+"\n")
        room_map=None
        if "room_map_file" in task:
            source=args.task_json.parent/task["room_map_file"]
            shutil.copy2(source,args.output/"evaluator_room_map.npz")
            with np.load(source) as data:
                room_map={key:data[key] for key in data.files}
        if args.deliver:
            build_delivery_bin(env.unwrapped,bin_xy)
        io.before_step=lambda: disturbance.before_step(io.step_id)
        policy_class=LearnedSearchPilot
        if args.navigation_version=="astar":
            from maniskill_crossroom_policy import CrossRoomSearchPilot
            policy_class=CrossRoomSearchPilot
            io.ceiling_safe_overview=True
        policy=policy_class(io,perception,args.output,args.query,args.variant=="dynamic")
        if args.video:
            recorder=EvidenceVideo(args.output,hz=env.unwrapped.control_freq)
        def after_step():
            # Evaluator trajectory is not returned to the policy.
            evaluation.append(dict(step=io.step_id,phase=policy.phase,target_xyz=array(target.pose.p)[0].tolist(),
                                   support_xyz=array(support.pose.p)[0].tolist(),
                                   target_velocity=array(target.get_linear_velocity())[0].tolist(),
                                   bilateral_contact=bool(io.robot.is_grasping(target)[0]) if io.grip<0 else False))
            if recorder:
                disturbance_label=None
                if args.navigation_version=="astar":
                    if disturbance.start_step<=io.step_id and disturbance.done_step is None:
                        disturbance_label="External event: target is being moved"
                    elif disturbance.done_step is not None and io.step_id<args.query_step:
                        disturbance_label="Target moved; robot has not searched yet"
                recorder.capture(io,policy.occupancy,policy.phase,cached=policy.cached,
                                 memory=policy.memory,query=policy.query,
                                 planned_path=getattr(policy,"planned_path",()),
                                 waypoint=getattr(policy,"waypoint",None),
                                 navigation_goal=getattr(policy,"navigation_goal",None),
                                 disturbance_label=disturbance_label)
        io.after_step=after_step
        policy.survey()
        policy.phase="Idle before retrieval request"
        policy.turn_to(math.pi)
        while io.step_id<args.query_step:
            io.command()
        policy.phase="Retrieve from memory"
        before=policy.retrieve()
        policy.focus()
        after=policy.retrieve()
        navigation_result=policy.navigate(budget=args.navigation_budget,wrist_scans=args.wrist_scans) if args.navigate else None
        grip_retained=policy.attempt_grasp() if args.grasp and navigation_result else False
        grasp_attempted=policy.grasp_motion_attempted
        placement_completed=policy.carry_and_place(home_xy,bin_xy) if args.deliver and grip_retained else False
        placement_attempted=policy.placement_motion_attempted
        # Finish one second of physical observation, not duplicated freeze frames.
        for _ in range(20):
            io.command()
        cv2.imwrite(str(args.output / "memory_map.png"), cv2.cvtColor(
            map_image(policy.occupancy,policy.memory,policy.query,io.pose()[:2],cached=policy.cached),cv2.COLOR_RGB2BGR))
        payload=dict(status="completed_development_pilot",variant=args.variant,
            boundary="Development configuration; a completed script is not a successful task or an unbiased benchmark",
            before=asdict(before) if before else None,after=asdict(after) if after else None,
            visual_approach_complete=navigation_result,env_steps=io.step_id,
            grasp_attempted=grasp_attempted,
            evaluator_bilateral_grasp=bool(env.unwrapped.agent.is_grasping(target)[0]) if grasp_attempted else False,
            evaluator_target_height_m=float(array(target.pose.p)[0,2]),
            placement_attempted=placement_attempted,
            placement_motion_completed=placement_completed,
            evaluator_placed_in_bin=bool(placement_attempted
                and np.max(np.abs(array(target.pose.p)[0,:2]-bin_xy))<.205
                and .69<float(array(target.pose.p)[0,2])<.81
                and np.linalg.norm(array(target.get_linear_velocity())[0])<.05),
            task_known_bin_xy=bin_xy.tolist() if args.deliver else None,
            disturbance_start_step=disturbance.start_step,disturbance_done_step=disturbance.done_step,
            evaluator_final_target_xyz=array(target.pose.p)[0].tolist(),
            evaluator_relocation_m=float(np.linalg.norm(array(target.pose.p)[0,:2]-np.asarray(args.target_xy))),
            model_lock=json.loads((Path(os.environ["HF_HUB_CACHE"]) / "dream_models.lock.json").read_text()),
            wall_time_s=time.monotonic()-started)
        from learned_evaluation import evaluate_episode,evaluate_dynamic_protocol
        payload.update(evaluate_episode(evaluation,io.trace,policy.events,bin_xy,
                                      hz=io.base.control_freq,require_delivery=args.deliver))
        payload.update(evaluate_dynamic_protocol(evaluation,policy.events,
            disturbance_step=args.disturbance_step,query_step=args.query_step,endpoint=np.asarray(args.endpoint)))
        payload["evaluator_protocol_success"]=bool(payload["evaluator_task_success"]
            and payload["evaluator_initial_target_observed"] and payload["evaluator_disturbance_valid"])
        if room_map is not None:
            from crossroom_evaluation import evaluate_room_transitions
            payload.update(evaluate_room_transitions(io.trace,policy.events,task,room_map,args.query_step))
            payload["evaluator_protocol_success"] &= payload["evaluator_cross_room_success"]
        payload["evaluator_support_relocation_m"]=float(np.linalg.norm(array(support.pose.p)[0,:2]-disturbance.initial[:2]))
        (args.output / "disturbance_forces.json").write_text(json.dumps(disturbance.trace)+"\n")
        (args.output / "result.json").write_text(json.dumps(payload,indent=2)+"\n")
        print(json.dumps(payload,indent=2),flush=True)
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps(dict(error=repr(error),
            env_steps=io.step_id if io else None),indent=2)+"\n")
        raise
    finally:
        if recorder:
            recorder.close()
        if io:
            (args.output / "actions.json").write_text(json.dumps(io.trace)+"\n")
        (args.output / "evaluator_trajectory.json").write_text(json.dumps(evaluation)+"\n")
        if policy:
            (args.output / "memory_updates.json").write_text(json.dumps(policy.memory.updates,indent=2)+"\n")
            np.savez_compressed(args.output / "observed_occupancy.npz",known=policy.occupancy.known,
                last_seen=policy.occupancy.last_seen,origin=policy.occupancy.origin,
                resolution=policy.occupancy.resolution,radius=policy.occupancy.radius)
        env.close()
        hashes={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!="sha256.json"}
        for source in snapshot.rglob("*.py"):
            hashes[f"source:{source.relative_to(snapshot)}"]=sha(source)
        (args.output / "sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")


if __name__=="__main__":
    main()
