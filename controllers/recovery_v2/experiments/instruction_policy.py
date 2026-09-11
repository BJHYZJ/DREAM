"""Instruction-driven learned search and observed household manipulation.

The policy never receives task fixtures, actor handles or known destination
coordinates. The runner/environment and independent evaluator own those data.
"""
from dataclasses import asdict,replace
import math

import numpy as np

from maniskill_crossroom_policy import CrossRoomSearchPilot
from maniskill_learned_probe import array
from instruction_geometry import tabletop_grasp,receptacle_region,support_relation_matches,point_observed_empty,align_symmetric_grasp_axis,observed_payload_template,observed_payload_offset,focused_rgbd_crop
from instruction_heading_navigation import InstructionHeadingNavigation
from maniskill_learned_dynamic import motion_detection_changed


class InstructionSearchPilot(InstructionHeadingNavigation,CrossRoomSearchPilot):
    def __init__(self,io,perception,output,instruction,dynamic=True,heading_navigation=False):
        self.instruction=instruction
        self.heading_navigation=heading_navigation
        self.initial_survey_complete=False
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

    def settle_head_view(self,maximum_steps=160,tolerance=.025):
        """Wait for measured pan/tilt, not an assumed fixed head-turn duration.

        Only ordinary zero-base commands are issued; arm targets stay fixed.
        A timeout is recorded and the subsequent image still uses its actual
        calibrated pose. No target geometry is used to fake camera alignment.
        """
        if maximum_steps<=0 or tolerance<=0:
            raise ValueError("Head settling needs a positive finite budget/tolerance")
        arm=self.io.arm.copy();target=np.asarray(self.io.body[:2],dtype=float).copy()
        stable=0;commands=0;settled=False
        for step in range(maximum_steps+1):
            measured=np.asarray(array(self.io.robot.robot.get_qpos())[0,[4,6]],dtype=float)
            if not np.isfinite(measured).all() or not np.isfinite(target).all():
                raise ValueError("Non-finite head proprioception/target")
            stable=stable+1 if np.max(np.abs(measured-target))<=tolerance else 0
            if stable>=3:
                settled=True;break
            if step==maximum_steps:break
            if self.task_stage in ("placement_search","place") and step%8==0:
                self.require_retained_payload()
            self.io.command();commands+=1
        unchanged=bool(np.array_equal(arm,self.io.arm))
        self.event("head_view_settled",settled=settled,additional_controls=commands,
                   target_pan_tilt_rad=target.tolist(),measured_pan_tilt_rad=measured.tolist(),
                   tolerance_rad=tolerance,arm_joint_targets_unchanged=unchanged)
        if not unchanged:raise RuntimeError("Arm targets changed while settling head view")
        return settled

    def observe(self,sensor_name="fetch_head"):
        if self.task_stage=="placement_search":
            self.require_retained_payload()
        if sensor_name=="fetch_head":
            self.settle_head_view()
        previous=None if self.cached is None else self.cached.copy()
        previous_detection=self.cached_detection
        detection=super().observe(sensor_name)
        prior=previous if previous is not None else getattr(self,"receptacle_view_prior",None)
        if (detection is None and self.task_stage in ("placement_search","place")
                and prior is not None and np.linalg.norm(prior[:2]-self.io.pose()[:2])<1.2):
            detection=self.focused_receptacle_detection(self.last_obs,prior)
            self.current_detection=detection
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
        if detection is not None:
            # A route-end observation can detect the object immediately before
            # a stopped head sweep looks elsewhere. Record that verified track
            # now, not only at the next planning/retrieval call. Otherwise a
            # failed top-ranked semantic view can discard the genuine detection.
            # This is only a visual search prior: free-depth/focus invalidation
            # above still clears it, and grasp/place still require fresh views.
            # Apply identically to both memory variants, after support grounding.
            if motion_detection_changed(previous,detection):
                # A newly seen or substantially displaced target supersedes a detour.
                # Repeated views of the same target keep an unfinished detour active.
                if self.exploration_goal is not None:
                    self.event("detour_superseded_by_visual_target",
                               previous_target=None if previous is None else previous.tolist(),
                               observed_target=detection.point_world)
                self.exploration_goal=None
            self.cached_detection=detection
            self.cached=np.asarray(detection.point_world,dtype=float)
            self.event("verified_visual_track",detection=asdict(detection),
                       source="actual_observation_after_support_check")
        if self.observation_listener is not None:
            # One-way observation event; the listener cannot return an answer.
            self.observation_listener(self.last_obs,detection,self.task_stage)
        return detection

    def missing_location_observed(self,point,observation):
        return point_observed_empty(observation,point)

    def focused_receptacle_detection(self,observation,prior):
        cropped=focused_rgbd_crop(observation,prior)
        if cropped is None:return None
        crop,bounds=cropped;x,y,_,_=bounds
        detections=[replace(d,box_xyxy=(np.asarray(d.box_xyxy)+[x,y,x,y]).tolist())
                    for d in self.perception.detect(crop,self.instruction.placement_query)]
        self.event("focused_receptacle_detection",observation_id=observation.frame_id,
            crop_box_xyxy=list(bounds),detections=[asdict(d) for d in detections],
            source="calibrated_crop_of_saved_camera_observation")
        return detections[0] if detections else None

    def focus_camera_height(self):
        # Use measured robot camera calibration, especially after torso motion.
        return float(self.io.capture("fetch_head",keyframe=False).camera_to_world_cv[2,3])

    def turn_to(self,yaw):
        if (getattr(self,"heading_navigation",False) and self.initial_survey_complete
                and self.task_stage in ("pickup_search","placement_search")):
            # Search/focus rotations must fit the current robot and held
            # envelope. Manipulation over table surfaces remains separately
            # evaluated in 3-D; it is not constrained by this navigation slab.
            footprint=self.measured_navigation_footprint()
            # In genuinely wide free space preserve the existing turning
            # controller/timing. The all-yaw disk must contain the measured
            # footprint plus padding and both raster-center uncertainty radii.
            # No occupancy radius is lowered and no map obstacle is removed.
            disk=footprint["maximum_combined_radius_m"]+footprint["padding_m"]+math.sqrt(2)*self.occupancy.resolution
            xy=self.io.pose()[:2]
            if not self.occupancy.segment_safe(xy,xy,radius=disk):
                completed=self.heading_executor().turn(yaw)
                if self.route_turn_observation:self.observe()
                return completed
        super().turn_to(yaw)
        if self.route_turn_observation:
            self.observe()  # Fresh semantic view before starting translation.

    def survey(self):
        super().survey()
        self.initial_survey_complete=True

    def move_chunk(self,*args,**kwargs):
        self.route_turn_observation=True
        start=self.io.pose()[:2].copy()
        try:
            moved=super().move_chunk(*args,**kwargs)
            if moved:self.note_forward_progress(float(np.linalg.norm(self.io.pose()[:2]-start)))
            return moved
        finally:
            self.route_turn_observation=False

    def docking_limits(self):
        if getattr(self,"heading_navigation",False) and self.task_stage=="placement_search":
            # Held Fetch geometry can make a .58-.64 m facing pose infeasible
            # beside a table. Hand off a little farther away; fresh language/
            # support grounding, measured TCP reach and actual released-object
            # evaluation remain mandatory. This never declares placement success.
            # The wider .70-.74 ring passed navigation but exceeded the tight
            # observed bread/plate release margin. .69-.705 passed actual
            # retained navigation and released placement with the same guards.
            return .69,.705,.725
        return super().docking_limits()

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

    def pick_frontier(self,start,distances,safe_override=None):
        local_prior_active=(self.search_region_center is not None
                            and self.regional_frontier_queries<self.regional_frontier_limit)
        goal=super().pick_frontier(start,distances,safe_override=safe_override)
        if goal is not None:return goal
        from instruction_observation_frontier import observation_standoffs
        from dream_learned_core import select_frontier_goal
        safe=self.occupancy.traversable() if safe_override is None else safe_override
        mask=observation_standoffs(self.occupancy.known,safe,distances,
            self.frontier_visits,self.occupancy.resolution,self.occupancy.radius)
        if not mask.any():return None
        if local_prior_active:
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
        previous=self.cached_detection
        detection=super().retrieve(use_live)
        if detection is not None and self.task_stage in ("placement_search","place"):
            observation=self.memory.observations[detection.observation_id]
            matched,_=self._support_for(observation,detection)
            if not matched:
                # This image cannot establish the support relation. It does
                # not establish that the receptacle disappeared from nearby
                # locations seen in other images. Exclude only this view.
                self.memory.reject(detection.point_world[:2],observation.frame_id,
                                   radius=None,observation_id=observation.frame_id,query=self.query)
                fallback=None
                if (previous is not None and previous.observation_id!=detection.observation_id
                        and not self.memory.detection_rejected(previous)):
                    old_observation=self.memory.observations.get(previous.observation_id)
                    if old_observation is not None and self._support_for(old_observation,previous)[0]:
                        fallback=previous
                self.cached_detection=fallback
                self.cached=None if fallback is None else np.asarray(fallback.point_world)
                self.event("placement_memory_view_rejected",observation_id=observation.frame_id,
                    invalidation_scope="candidate_observation",preserved_track=asdict(fallback) if fallback else None)
                return fallback
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
                and previous is not None
                and np.linalg.norm(previous[:2]-self.io.pose()[:2])<1.2):
            # Invalidation may just have cleared self.cached. The old visual
            # point can still aim an actual recovery observation; it is NOT
            # reinstated as a verified destination. Require new grounding.
            current=self.offset_head_view(previous)
            if (current is None and self.task_stage=="placement_search"
                    and np.linalg.norm(previous[:2]-self.io.pose()[:2])<.90):
                current=self.alternate_receptacle_views(previous)
        return current

    def alternate_receptacle_views(self,remembered_point):
        """Unfold the carried arm slightly when it blocks a nearby receptacle view."""
        if getattr(self,"near_receptacle_views_tried",False):return None
        self.near_receptacle_views_tried=True
        self.require_retained_payload()
        transform=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
        direction=transform[:2,3]-self.io.pose()[:2]
        initial_reach=float(np.linalg.norm(direction))
        if not .05<initial_reach<.50:return None
        direction=direction/initial_reach
        rotation=transform[:3,:3].copy();height=float(transform[2,3])
        self.receptacle_view_prior=np.asarray(remembered_point,dtype=float).copy()
        try:
            for extension in (.12,.24):
                reach=min(initial_reach+extension,.55)
                goal=np.r_[self.io.pose()[:2]+direction*reach,height]
                self.phase="Clear held-arm camera occlusion"
                self.io.robot.set_control_mode("pd_ee_delta_pose");self.io.robot.controller.reset()
                try:
                    for step in range(360):
                        if step%8==0:self.require_retained_payload()
                        self.hold_release_pose_step(goal,rotation)
                        measured=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
                        residual=float(np.linalg.norm(measured[:3,3]-goal))
                        angle=math.acos(float(np.clip((np.trace(rotation@measured[:3,:3].T)-1)/2,-1.,1.)))
                        if residual<.015 and angle<.06:break
                finally:
                    self.io.hold_measured_arm()
                self.require_retained_payload()
                from inspect_fetch_envelope import measure
                envelope=measure(self.io)
                tcp=array(self.io.robot.tcp_pose.p)[0]
                payload_radius=np.linalg.norm(tcp[:2]-self.io.pose()[:2])+self.perceived_payload_radius
                self.occupancy.radius=max(self.occupancy.radius,envelope["max_radius"]+.04,
                                          float(payload_radius)+.04)
                self.event("retained_payload_view_extension",goal_world=goal.tolist(),
                    requested_reach_m=reach,residual_m=residual,rotation_error_rad=angle,
                    navigation_radius_m=self.occupancy.radius,controls=step+1)
                if residual>.06 or angle>.15:return None
                delta=self.receptacle_view_prior[:2]-self.io.pose()[:2]
                bearing=math.atan2(delta[1],delta[0])
                self.io.body[0]=float(np.clip(math.atan2(math.sin(bearing-self.io.pose()[2]),
                    math.cos(bearing-self.io.pose()[2])),-1.3,1.3))
                self.io.body[1]=float(np.clip(math.atan2(self.focus_camera_height()-remembered_point[2],
                    np.linalg.norm(delta)),-.3,.9))
                result=self.observe()
                if result is not None:
                    try:
                        receptacle_region(self.last_obs,result,self.instruction.placement_relation,
                            self.perceived_payload_radius,self.perceived_payload_height)
                    except ValueError as error:
                        self.event("receptacle_extension_view_incomplete",observation_id=self.last_obs.frame_id,
                                   reason=str(error))
                        continue
                    return result
            return None
        finally:
            self.receptacle_view_prior=None

    def offset_head_view(self,remembered_point):
        """Move the held arm out of the sightline by turning the base/head.

        Only a remembered visual point aims the camera. The arm's joint
        targets and closed gripper are fixed; no destination is inferred from
        a failed detector frame. Both memory variants use this recovery.
        """
        original_arm=self.io.arm.copy()
        for offset in (-.55,.55,-1.05,1.05):
            self.require_retained_payload()
            delta=np.asarray(remembered_point)[:2]-self.io.pose()[:2]
            bearing=math.atan2(delta[1],delta[0])
            self.phase="Observe beside held object"
            turned=self.turn_to(bearing+offset)
            self.io.body[0]=float(np.clip(math.atan2(math.sin(bearing-self.io.pose()[2]),
                math.cos(bearing-self.io.pose()[2])),-1.3,1.3)) if turned is False else -offset
            self.io.body[1]=float(np.clip(math.atan2(self.focus_camera_height()-remembered_point[2],
                                                     np.linalg.norm(delta)),-.3,.9))
            for _ in range(20):self.io.command()
            result=self.observe()
            self.event("offset_head_view",body_bearing_offset_rad=offset,
                arm_joint_targets_unchanged=bool(np.array_equal(original_arm,self.io.arm)),
                observation_id=self.last_obs.frame_id,detected=result is not None)
            if result is not None:
                return result
        if float(self.io.body[2])>.20:
            # Side views can still see only a nearly top-down, ungrounded
            # receptacle. Use the existing lower-torso head-view recovery now,
            # rather than exhausting the entire navigation budget first.
            # No arm-joint sweep, prior-coordinate release or threshold change.
            self.require_retained_payload()
            self.io.body[2]=.18
            self.phase="Lower head viewpoint for receptacle"
            for _ in range(100):self.io.command()
            delta=np.asarray(remembered_point)[:2]-self.io.pose()[:2]
            bearing=math.atan2(delta[1],delta[0])
            turned=self.turn_to(bearing)
            self.io.body[0]=float(np.clip(math.atan2(math.sin(bearing-self.io.pose()[2]),
                math.cos(bearing-self.io.pose()[2])),-1.3,1.3)) if turned is False else 0.
            self.io.body[1]=float(np.clip(math.atan2(self.focus_camera_height()-remembered_point[2],
                                                     np.linalg.norm(delta)),-.3,.9))
            for _ in range(20):self.io.command()
            result=self.observe()
            self.event("head_viewpoint_height_recovery",torso_target_m=float(self.io.body[2]),
                arm_joint_targets_unchanged=bool(np.array_equal(original_arm,self.io.arm)),
                observation_id=self.last_obs.frame_id,detected=result is not None,
                during_local_focus=True)
            return result
        return None

    def propose_grasp(self,observation,detection):
        try:
            target,fit=tabletop_grasp(observation,detection)
            return target,align_symmetric_grasp_axis(target,fit,self.io.pose()[:2])
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
                target,fit=tabletop_grasp(self.last_obs,fresh)
                return target,align_symmetric_grasp_axis(target,fit,self.io.pose()[:2])
            except ValueError as error:
                self.event("grasp_geometry_rejected",observation_id=self.last_obs.frame_id,error=str(error))
        raise ValueError("No valid observed grasp geometry after actual complementary head views")

    def attempt_grasp(self):
        self.task_stage="grasp"
        self.payload_template=None
        retained=super().attempt_grasp()
        if retained and self.instruction.placement_relation=="in":
            fit=next(e for e in reversed(self.events) if e["event"]=="depth_grasp_center")
            approach=next(e for e in reversed(self.events) if e["event"]=="grasp_tcp_approach")
            observation=self.memory.observations[fit["observation_id"]]
            detections=self.perception.detect(observation,self.instruction.pickup_query)
            if detections:
                try:
                    self.payload_template=observed_payload_template(observation,detections[0],
                        fit["center_world"],np.asarray(approach["tcp_matrix"]))
                    np.savez_compressed(self.output/"observed_payload_template.npz",**self.payload_template)
                    self.event("observed_payload_template_created",observation_id=observation.frame_id,
                        points=len(self.payload_template["points_tcp"]))
                except ValueError as error:
                    self.event("observed_payload_template_unavailable",reason=str(error))
        return retained

    def hold_release_pose_step(self,goal,rotation):
        # Zero delta commands follow the measured tool and permit gravity drift.
        # Keep a fixed world pose with the existing bounded Cartesian feedback.
        before=self.io.step_id
        self.io.move_tcp(goal,tolerance=.0005,steps=1,target_rotation=rotation)
        if self.io.step_id==before:self.io.command_ee()

    def align_observed_release(self,release,rotation):
        template=getattr(self,"payload_template",None)
        if template is None or self.instruction.placement_relation!="in":
            return release.copy(),np.zeros(3)
        delta=release[:2]-self.io.pose()[:2]
        bearing=math.atan2(delta[1],delta[0])
        self.io.body[0]=float(np.clip(math.atan2(math.sin(bearing-self.io.pose()[2]),
            math.cos(bearing-self.io.pose()[2])),-1.3,1.3))
        camera_height=self.focus_camera_height()
        self.io.body[1]=float(np.clip(math.atan2(camera_height-release[2],np.linalg.norm(delta)),-.3,.9))
        stable=0
        for step in range(160):
            if step%8==0:self.require_retained_payload()
            measured=array(self.io.robot.robot.get_qpos())[0,[4,6]]
            stable=stable+1 if np.max(np.abs(measured-self.io.body[:2]))<.025 else 0
            if stable>=3:break
            self.hold_release_pose_step(release,rotation)
        if stable<3:
            self.event("payload_alignment_declined",reason="head_view_not_settled")
            return release.copy(),np.zeros(3)
        self.observe()
        observation=self.last_obs
        detections=self.perception.detect(observation,self.instruction.pickup_query)
        tcp=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
        try:
            offset,fit=observed_payload_offset(observation,detections,template,tcp)
        except ValueError as error:
            self.event("payload_alignment_declined",reason=str(error),observation_id=observation.frame_id)
            return release.copy(),np.zeros(3)
        world_offset=rotation@offset
        if np.linalg.norm(world_offset[:2])>.06:
            self.event("payload_alignment_declined",reason="horizontal_correction_exceeds_bound")
            return release.copy(),np.zeros(3)
        goal=release.copy();goal[:2]-=world_offset[:2]
        self.event("observed_payload_alignment",offset_tcp=offset.tolist(),
                   commanded_tcp_world=goal.tolist(),**fit)
        self.io.move_tcp(goal,tolerance=.006,target_rotation=rotation)
        return goal,offset

    def wait_for_release_stillness(self,goal,rotation,release=None,geometry=None,offset=None):
        stable=0;previous=array(self.io.robot.tcp_pose.p)[0].copy()
        frequency=float(self.io.base.control_freq)
        for step in range(int(round(2*frequency))):
            self.require_retained_payload()
            self.hold_release_pose_step(goal,rotation)
            current=array(self.io.robot.tcp_pose.p)[0].copy()
            speed=float(np.linalg.norm(current-previous)*frequency)
            positioned=True
            if release is not None:
                matrix=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
                center=current+matrix[:3,:3]@offset
                clearance=geometry["observed_safe_radius_m"]-self.perceived_payload_radius-.004
                positioned=(np.linalg.norm(center[:2]-release[:2])<clearance
                            and abs(current[2]-release[2])<.025)
            stable=stable+1 if speed<.01 and positioned else 0
            previous=current
            if stable>=3:
                self.event("release_pose_settled",speed_m_s=speed,controls=step+1)
                return True
        self.event("release_pose_unsettled",speed_m_s=speed,maximum_seconds=2.)
        return False

    def require_retained_payload(self):
        # Robot proprioception only, not the evaluator's actor contact query.
        gap=float(array(self.io.robot.robot.get_qpos())[0,-2:].sum())
        if gap<.015:
            self.event("payload_lost_by_proprioception",finger_gap_m=gap)
            raise RuntimeError("Closed gripper is empty; abort delivery rather than perform an empty placement")

    def carry_rotation(self,measured_rotation):
        # Default behavior remains orientation-preserving. A separately marked
        # near-table calibration may override this hook without changing an
        # ongoing full-task source snapshot or its policy configuration.
        return measured_rotation.copy()

    def prepare_carry(self):
        self.require_retained_payload()
        # A fixed whole-arm joint fold dropped some small as well as large
        # payloads. Use the same observation/proprioception-based Cartesian
        # retraction for every class; never select a controller from actor data.
        self.phase=getattr(self,"carry_phase_label","Carry: preserve grasp orientation")
        self.io.display_sensor="fetch_head"
        if not self.retreat(.30):self.retreat(.15)
        self.io.body[2]=.385
        for _ in range(140):self.io.command()
        self.require_retained_payload()
        transform=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
        rotation=self.carry_rotation(transform[:3,:3])
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
            posture=getattr(self,"carry_posture_description","orientation_preserving_observed_payload"))
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

    def propose_placement(self):
        """Reobserve an incomplete receptacle fit from a higher head viewpoint."""
        remembered=None if self.cached is None else self.cached.copy()
        for view,torso in enumerate((None,.30,.385)):
            self.require_retained_payload()
            if torso is not None:
                measured=float(array(self.io.robot.robot.get_qpos())[0,3])
                if torso<=measured+.015:
                    continue
                limits=array(self.io.robot.robot.get_qlimits())[0,3]
                self.io.body[2]=float(np.clip(torso,limits[0]+.01,limits[1]-.01))
                self.phase="Observe receptacle interior"
                for step in range(100):
                    if step%10==0:self.require_retained_payload()
                    self.io.command()
                if remembered is not None:
                    delta=remembered[:2]-self.io.pose()[:2]
                    bearing=math.atan2(delta[1],delta[0])
                    self.io.body[0]=float(np.clip(math.atan2(math.sin(bearing-self.io.pose()[2]),
                        math.cos(bearing-self.io.pose()[2])),-1.3,1.3))
                    self.io.body[1]=float(np.clip(math.atan2(self.focus_camera_height()-remembered[2],
                        np.linalg.norm(delta)),-.3,.9))
                self.observe()
            else:
                self.focus()
                if self.current_detection is None and self.cached is not None:
                    self.scan_sweep()
            detection=self.current_detection
            if detection is None:
                self.event("placement_view_rejected",view=view,reason="no_fresh_detection")
                continue
            observation=self.memory.observations[detection.observation_id]
            matched,support=self._support_for(observation,detection)
            if not matched:
                self.event("placement_view_rejected",view=view,reason="support_not_grounded",
                    observation_id=observation.frame_id)
                continue
            remembered=np.asarray(detection.point_world)
            try:
                release,geometry=receptacle_region(observation,detection,
                    self.instruction.placement_relation,self.perceived_payload_radius,self.perceived_payload_height)
            except ValueError as error:
                self.event("placement_geometry_view_rejected",view=view,
                    observation_id=observation.frame_id,error=str(error),
                    torso_height_m=float(array(self.io.robot.robot.get_qpos())[0,3]))
                continue
            self.event("placement_geometry_view_accepted",view=view,
                observation_id=observation.frame_id,
                torso_height_m=float(array(self.io.robot.robot.get_qpos())[0,3]))
            return release,geometry
        self.event("placement_aborted_no_valid_observed_geometry")
        return None

    def align_base_before_placement(self,release,geometry):
        """Approach a verified receptacle slightly before unfolding a broad payload."""
        if (not getattr(self,"heading_navigation",False)
                or self.instruction.placement_relation!="on"
                or self.perceived_payload_radius<=.08):
            return None
        clearance=geometry["observed_safe_radius_m"]-self.perceived_payload_radius-.004
        if not 0.<clearance<.035:return None
        base=self.io.pose().copy();delta=np.asarray(release[:2])-base[:2]
        distance=float(np.linalg.norm(delta))
        if not np.isfinite(distance) or distance<=.71:return None
        bearing=math.atan2(delta[1],delta[0])
        error=math.atan2(math.sin(bearing-base[2]),math.cos(bearing-base[2]))
        if abs(error)>.08:return None
        advance=min(.05,distance-.69)
        goal=base[:2]+advance*np.array([math.cos(base[2]),math.sin(base[2])])
        from dream_fetch_footprint import capture_robot_footprint
        from fetch_heading_control import HeadingRouteExecutor
        self.require_retained_payload()
        self.phase="Align base before arm extension"
        self.waypoint=goal.tolist();self.planned_path=[base[:2].tolist(),goal.tolist()]
        executor=HeadingRouteExecutor(self.io,self.occupancy,
            lambda:capture_robot_footprint(self.io,payload_radius=self.perceived_payload_radius,
                                          payload_height=self.perceived_payload_height),
            self.geometry_observe,self.event)
        self.event("observed_placement_base_alignment",planned_advance_m=advance,
                   observed_release_world=np.asarray(release).tolist(),
                   observed_xy_clearance_m=clearance,base=base.tolist())
        completed=executor.translate(goal,planned_heading=base[2])
        self.require_retained_payload()
        self.event("observed_placement_base_alignment_result",completed=completed,
                   measured_advance_m=float(np.linalg.norm(self.io.pose()[:2]-base[:2])),
                   last_failure=executor.last_failure)
        return completed

    def place_observed(self):
        self.require_retained_payload()
        self.task_stage="place"
        proposal=self.propose_placement()
        if proposal is None:
            return False
        release,geometry=proposal
        self.event("observed_placement_proposal",release_world=release.tolist(),**geometry)
        yaw=math.atan2(*(release[:2]-self.io.pose()[:2])[::-1])
        self.turn_to(yaw)
        self.align_base_before_placement(release,geometry)
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
        command_release,offset=self.align_observed_release(release,vertical)
        if not self.wait_for_release_stillness(command_release,vertical,
                release=release,geometry=geometry,offset=offset):return False
        tcp_pose=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
        tcp=tcp_pose[:3,3]
        estimated_center=tcp+tcp_pose[:3,:3]@offset
        clearance=geometry["observed_safe_radius_m"]-self.perceived_payload_radius-.004
        inside=np.linalg.norm(estimated_center[:2]-release[:2])<clearance
        height=abs(tcp[2]-release[2])<.025
        self.event("observed_release_check",measured_tcp=tcp.tolist(),release_world=release.tolist(),
                   commanded_tcp_world=command_release.tolist(),estimated_payload_center_world=estimated_center.tolist(),
                   offset_tcp=offset.tolist(),pre_residual_m=pre,residual_m=float(np.linalg.norm(tcp-command_release)),
                   horizontal_pass=bool(inside),height_pass=bool(height))
        if not (inside and height):
            return False
        self.require_retained_payload()
        for grip in np.linspace(-1.,1.,40):
            self.io.grip=float(grip)
            self.hold_release_pose_step(command_release,vertical)
        self.io.grip=1.
        for _ in range(40):self.hold_release_pose_step(command_release,vertical)
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
