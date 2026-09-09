"""Instruction-driven learned search and observed household manipulation.

The policy never receives task fixtures, actor handles or known destination
coordinates. The runner/environment and independent evaluator own those data.
"""
from dataclasses import asdict
import math

import numpy as np

from maniskill_crossroom_policy import CrossRoomSearchPilot
from maniskill_learned_probe import array
from instruction_geometry import tabletop_grasp,receptacle_region,support_relation_matches,point_observed_empty


class InstructionSearchPilot(CrossRoomSearchPilot):
    def __init__(self,io,perception,output,instruction,dynamic=True):
        self.instruction=instruction
        self.task_stage="pickup_search"
        self.observation_listener=None
        self.latest_support_detection=None
        self.last_memory_change=None
        self.placement_sightings=[]
        self.route_turn_observation=False
        self.regional_frontier_limit=48
        super().__init__(io,perception,output,instruction.pickup_query,dynamic)
        self.event("instruction_received",instruction=instruction.policy_payload())

    def event(self,name,**data):
        super().event(name,task_stage=self.task_stage,**data)

    def _support_for(self,observation,detection):
        if not self.instruction.support_query:
            return True,None
        supports=self.perception.detect(observation,self.instruction.support_query)
        found=next((s for s in supports if support_relation_matches(observation,detection,s)),None)
        return found is not None,found

    def observe(self,sensor_name="fetch_head"):
        if self.task_stage=="placement_search":
            self.require_retained_payload()
        previous=None if self.cached is None else self.cached.copy()
        previous_detection=self.cached_detection
        detection=super().observe(sensor_name)
        if self.task_stage=="pickup_search" and previous is not None:
            # A changed live detection can precede the next explicit focus.
            # Require measured free depth at the old point; a new box alone
            # must not be mistaken for proof that the old object disappeared.
            if point_observed_empty(self.last_obs,previous):
                old_id=previous_detection.observation_id if previous_detection else None
                self.memory.reject(previous[:2],self.last_obs.frame_id-1,
                                   observation_id=old_id,query=self.query)
                self.last_memory_change=dict(step=self.io.step_id,old_point=previous.tolist(),
                    new_point=detection.point_world if detection else None)
                self.event("observed_memory_update",old_point=previous.tolist(),
                    new_point=detection.point_world if detection else None,
                    dynamic_updates_enabled=self.memory.dynamic,
                    evidence_observation_id=self.last_obs.frame_id,evidence="observed_free_depth_at_old_location")
                self.cached_detection=detection
                self.cached=None if detection is None else np.asarray(detection.point_world)
                if detection is None:
                    self.search_anchor=previous[:2].copy()
                    self.search_region_center=previous[:2].copy()
                    self.regional_frontier_queries=0
        if self.task_stage=="pickup_search":
            # Both nouns are already in the initial instruction. Remember a
            # genuinely observed destination without acting on it prematurely.
            # This record remains subject to focused verification after pickup.
            for candidate in self.perception.detect(self.last_obs,self.instruction.placement_query):
                matched,support=self._support_for(self.last_obs,candidate)
                if matched:
                    self.placement_sightings.append(candidate)
                    self.event("placement_observed_during_initial_search",detection=asdict(candidate),
                               support=asdict(support) if support else None)
        if detection is not None and self.task_stage in ("placement_search","place"):
            matched,support=self._support_for(self.last_obs,detection)
            self.latest_support_detection=support
            self.event("placement_support_verification",observation_id=self.last_obs.frame_id,
                       candidate=asdict(detection),support=asdict(support) if support else None,
                       relation_grounded=matched)
            if not matched:
                self.current_detection=None
                detection=None
        if self.observation_listener is not None:
            # One-way observation event; the listener cannot return an answer.
            self.observation_listener(self.last_obs,detection,self.task_stage)
        return detection

    def missing_location_observed(self,point,observation):
        return point_observed_empty(observation,point)

    def focus_camera_height(self):
        # Use measured robot camera calibration, especially after torso motion.
        return float(self.io.capture("fetch_head",keyframe=False).camera_to_world_cv[2,3])

    def turn_to(self,yaw):
        super().turn_to(yaw)
        if self.route_turn_observation:
            self.observe()  # Fresh semantic view before starting translation.

    def move_chunk(self,*args,**kwargs):
        self.route_turn_observation=True
        try:
            return super().move_chunk(*args,**kwargs)
        finally:
            self.route_turn_observation=False

    def navigate(self,budget=90,wrist_scans=False):
        if super().navigate(budget=budget,wrist_scans=False):
            return True
        # A remembered bowl viewed nearly from above may not be text-grounded.
        # Try a lower actual camera viewpoint while stopped, with the arm's
        # joint targets and closed gripper unchanged. Never assume placement.
        if (self.task_stage=="placement_search" and self.cached is not None
                and np.linalg.norm(self.cached[:2]-self.io.pose()[:2])<.90):
            initial_arm=self.io.arm.copy()
            self.io.body[2]=min(float(self.io.body[2]),.18)
            self.phase="Lower head viewpoint for receptacle"
            for _ in range(100): self.io.command()
            self.event("head_viewpoint_height_recovery",torso_target_m=float(self.io.body[2]),
                       arm_joint_targets_unchanged=bool(np.array_equal(initial_arm,self.io.arm)))
            self.focus()
            return super().navigate(budget=min(budget,20),wrist_scans=False)
        return False

    def pick_frontier(self,start,distances):
        goal=super().pick_frontier(start,distances)
        if goal is not None:return goal
        from instruction_observation_frontier import observation_standoffs
        from dream_learned_core import select_frontier_goal
        mask=observation_standoffs(self.occupancy.known,self.occupancy.traversable(),distances,
            self.frontier_visits,self.occupancy.resolution,self.occupancy.radius)
        if not mask.any():return None
        if self.search_region_center is not None:
            cells=np.argwhere(mask)
            local=np.linalg.norm(self.occupancy.world(cells)-self.search_region_center,axis=1)<=4.
            if local.any():
                mask[:]=False;selected=cells[local];mask[selected[:,0],selected[:,1]]=True
        age=self.occupancy.last_seen.max()-self.occupancy.last_seen.astype(float)
        age/=max(1.,age.max());age-=np.minimum(self.frontier_visits,3)*.35
        goal=select_frontier_goal(age,self.occupancy.semantic_field(self.memory,self.query),
                                 mask,np.array(start),semantic_rate=.1).index
        if goal is not None:
            self.event("frontier_observation_standoff",point_xy=self.occupancy.world(goal).tolist(),
                source="reachable observed-free cells near an occlusion frontier",radius_m=self.occupancy.radius)
        return goal

    def retrieve(self,use_live=False):
        detection=super().retrieve(use_live)
        if detection is not None and self.task_stage in ("placement_search","place"):
            observation=self.memory.observations[detection.observation_id]
            matched,_=self._support_for(observation,detection)
            if not matched:
                self.memory.reject(detection.point_world[:2],observation.frame_id,
                                   observation_id=observation.frame_id,query=self.query)
                self.cached=None
                self.cached_detection=None
                return None
        return detection

    def focus(self):
        previous=None if self.cached is None else self.cached.copy()
        current=super().focus()
        if previous is not None and self.events[-1]["event"] in ("focused_verification","last_seen_search_anchor"):
            event=next(e for e in reversed(self.events) if e["event"]=="focused_verification")
            if event["missing"] or event["displaced"]:
                self.last_memory_change=dict(step=self.io.step_id,old_point=previous.tolist(),
                    new_point=current.point_world if current else None)
                self.event("observed_memory_update",old_point=previous.tolist(),
                           new_point=current.point_world if current else None,
                           dynamic_updates_enabled=self.memory.dynamic,
                           evidence_observation_id=self.last_obs.frame_id)
        if (current is None and self.task_stage in ("placement_search","place")
                and self.cached is not None and previous is not None
                and np.linalg.norm(previous[:2]-self.io.pose()[:2])<1.2):
            current=self.offset_head_view(previous)
        return current

    def offset_head_view(self,remembered_point):
        """Move the held arm out of the sightline by turning the base/head.

        Only a remembered visual point aims the camera. The arm's joint
        targets and closed gripper are fixed; no destination is inferred from
        a failed detector frame. Both memory variants use this recovery.
        """
        original_arm=self.io.arm.copy()
        for offset in (-.55,.55):
            self.require_retained_payload()
            delta=np.asarray(remembered_point)[:2]-self.io.pose()[:2]
            bearing=math.atan2(delta[1],delta[0])
            self.phase="Observe beside held object"
            self.turn_to(bearing+offset)
            self.io.body[0]=-offset
            self.io.body[1]=float(np.clip(math.atan2(self.focus_camera_height()-remembered_point[2],
                                                     np.linalg.norm(delta)),-.3,.9))
            for _ in range(20):self.io.command()
            result=self.observe()
            self.event("offset_head_view",body_bearing_offset_rad=offset,
                arm_joint_targets_unchanged=bool(np.array_equal(original_arm,self.io.arm)),
                observation_id=self.last_obs.frame_id,detected=result is not None)
            if result is not None:
                return result
        return None

    def propose_grasp(self,observation,detection):
        try:
            return tabletop_grasp(observation,detection)
        except ValueError as error:
            self.event("grasp_geometry_rejected",observation_id=observation.frame_id,error=str(error))
        # A bad foreground fit (e.g. cart rim in the learned box) is a request
        # for actual additional sensing, not an excuse to widen the gripper's
        # limits or substitute an actor pose. Rotate the body/head, keeping the
        # arm targets fixed, and require a nearby fresh text-grounded detection.
        remembered=np.asarray(detection.point_world)
        original_arm=self.io.arm.copy()
        for offset in (-.40,.40):
            delta=remembered[:2]-self.io.pose()[:2]
            bearing=math.atan2(delta[1],delta[0])
            self.phase="Observe again before grasp"
            self.turn_to(bearing+offset)
            self.io.body[0]=-offset
            self.io.body[1]=float(np.clip(math.atan2(self.focus_camera_height()-remembered[2],
                                                     np.linalg.norm(delta)),-.3,.9))
            for _ in range(20):self.io.command()
            fresh=self.observe()
            self.event("grasp_geometry_view_retry",observation_id=self.last_obs.frame_id,
                body_bearing_offset_rad=offset,arm_joint_targets_unchanged=bool(np.array_equal(original_arm,self.io.arm)),
                detected=fresh is not None)
            if fresh is None or np.linalg.norm(np.asarray(fresh.point_world)-remembered)>.15:
                continue
            try:
                return tabletop_grasp(self.last_obs,fresh)
            except ValueError as error:
                self.event("grasp_geometry_rejected",observation_id=self.last_obs.frame_id,error=str(error))
        raise ValueError("No valid observed grasp geometry after actual complementary head views")

    def attempt_grasp(self):
        self.task_stage="grasp"
        return super().attempt_grasp()

    def require_retained_payload(self):
        # Robot proprioception only, not the evaluator's actor contact query.
        gap=float(array(self.io.robot.robot.get_qpos())[0,-2:].sum())
        if gap<.015:
            self.event("payload_lost_by_proprioception",finger_gap_m=gap)
            raise RuntimeError("Closed gripper is empty; abort delivery rather than perform an empty placement")

    def prepare_carry(self):
        self.require_retained_payload()
        # A fixed whole-arm joint fold dropped some small as well as large
        # payloads. Use the same observation/proprioception-based Cartesian
        # retraction for every class; never select a controller from actor data.
        self.phase="Carry: preserve grasp orientation"
        self.io.display_sensor="fetch_head"
        if not self.retreat(.30):self.retreat(.15)
        self.io.body[2]=.385
        for _ in range(140):self.io.command()
        self.require_retained_payload()
        transform=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
        rotation=transform[:3,:3].copy()
        delta=transform[:2,3]-self.io.pose()[:2]
        direction=delta/max(np.linalg.norm(delta),1e-9)
        # Keep the held object below the head's forward view. A high
        # front carry preserved the grasp but occluded the destination.
        carry_height=float(np.clip(self.focus_camera_height()-.38,.95,1.10))
        self.io.robot.set_control_mode("pd_ee_delta_pose");self.io.robot.controller.reset()
        for reach in (.55,.40,.28):
            goal=np.r_[self.io.pose()[:2]+direction*reach,carry_height]
            residual=self.io.move_tcp(goal,tolerance=.02,steps=360,target_rotation=rotation)
            self.event("orientation_preserving_carry_retraction",requested_reach_m=reach,
                       goal_world=goal.tolist(),residual_m=residual)
            self.require_retained_payload()
            if residual>.06:break
        self.io.hold_measured_arm()
        for _ in range(40):self.io.command()
        from inspect_fetch_envelope import measure
        envelope=measure(self.io)
        tcp=array(self.io.robot.tcp_pose.p)[0]
        payload_radius=np.linalg.norm(tcp[:2]-self.io.pose()[:2])+self.perceived_payload_radius
        self.occupancy.radius=max(.40,envelope["max_radius"]+.04,float(payload_radius)+.04)
        self.event("measured_carry_envelope",robot=envelope,
            payload_radius_from_depth_m=self.perceived_payload_radius,navigation_radius_m=self.occupancy.radius,
            posture="orientation_preserving_observed_payload")
        self.observe()
        self.require_retained_payload()
        return True

    def start_placement_search(self):
        self.require_retained_payload()
        # Keep the scene memory; reset only target-specific navigation state.
        # Query-specific rejection prevents a missing pickup from hiding its
        # still-valid support/receptacle in the same historical image.
        self.task_stage="placement_search"
        self.query=self.instruction.placement_query
        self.cached=None
        self.cached_detection=None
        self.current_detection=None
        self.search_anchor=None
        self.search_region_center=None
        self.exploration_goal=None
        self.regional_frontier_queries=0
        self.frontier_visits[:]=0
        self.planned_path=[]
        self.navigation_goal=None
        self.waypoint=None
        self.io.display_sensor="fetch_head"
        sightings=[d for d in self.placement_sightings if not self.memory.detection_rejected(d)]
        if sightings:
            self.cached_detection=sightings[-1]
            self.cached=np.asarray(sightings[-1].point_world)
        self.event("placement_search_started",query=self.query,
                   support_query=self.instruction.support_query,memory_preserved=True,
                   memory_observations=len(self.memory.observations),
                   prior_visual_sighting=asdict(self.cached_detection) if self.cached_detection else None)

    def place_observed(self):
        self.require_retained_payload()
        self.task_stage="place"
        self.focus()
        detection=self.current_detection
        if detection is None and self.cached is not None:
            # One failed detector frame is not a task failure or a valid
            # release target: acquire another real stationary head sweep.
            self.scan_sweep()
            detection=self.current_detection
        if detection is None:
            self.event("placement_aborted_no_fresh_receptacle")
            return False
        observation=self.memory.observations[detection.observation_id]
        matched,support=self._support_for(observation,detection)
        if not matched:
            self.event("placement_aborted_support_not_grounded")
            return False
        release,geometry=receptacle_region(observation,detection,
            self.instruction.placement_relation,self.perceived_payload_radius,self.perceived_payload_height)
        self.event("observed_placement_proposal",release_world=release.tolist(),**geometry)
        yaw=math.atan2(*(release[:2]-self.io.pose()[:2])[::-1])
        self.turn_to(yaw)
        self.io.restore_grasp_seed(grip=-1.)
        self.io.position_torso_for_grasp(release[2])
        self.io.robot.set_control_mode("pd_ee_delta_pose")
        self.io.robot.controller.reset()
        self.io.display_sensor="fetch_head"
        world_yaw=np.array([[math.cos(yaw),-math.sin(yaw),0],
                           [math.sin(yaw),math.cos(yaw),0],[0,0,1]])
        vertical=world_yaw@np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        self.phase="Place in observed receptacle"
        self.placement_motion_attempted=True
        pre=self.io.move_tcp(release+[0,0,.15],target_rotation=vertical)
        residual=self.io.move_tcp(release,tolerance=.008,target_rotation=vertical)
        tcp=array(self.io.robot.tcp_pose.p)[0]
        clearance=geometry["observed_safe_radius_m"]-self.perceived_payload_radius-.004
        inside=np.linalg.norm(tcp[:2]-release[:2])<clearance
        height=abs(tcp[2]-release[2])<.025
        self.event("observed_release_check",measured_tcp=tcp.tolist(),release_world=release.tolist(),
                   pre_residual_m=pre,residual_m=residual,horizontal_pass=bool(inside),height_pass=bool(height))
        if not (inside and height):
            return False
        self.require_retained_payload()
        for grip in np.linspace(-1.,1.,40):
            self.io.command_ee(grip=float(grip))
        for _ in range(40):self.io.command_ee(grip=1.)
        self.event("gripper_release_proprioception",
            measured_finger_gap_m=float(array(self.io.robot.robot.get_qpos())[0,-2:].sum()),
            maximum_finger_gap_m=float(array(self.io.robot.robot.get_qlimits())[0,-2:,1].sum()),
            physical_release_requires_independent_evaluation=True)
        self.io.move_tcp(array(self.io.robot.tcp_pose.p)[0]+[0,0,.15])
        self.phase="Check released placement"
        for _ in range(60):
            self.io.command_ee()
        self.task_stage="finished"
        self.event("placement_motion_complete",success_requires_independent_evaluation=True)
        return True
