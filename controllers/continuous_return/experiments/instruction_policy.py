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
        self.visual_search_observations=0
        # Recover nearby receptacle views with the existing guarded base/head
        # motions. The legacy arm extension has no observed full-arm sweep.
        self.receptacle_arm_views=False
        self.deferred_pickup_regions=[]
        # An old sighting remains a short local prior; after four choices,
        # allow all reachable frontiers so a missed room can be explored.
        self.regional_frontier_limit=4
        from instruction_search_progress import SearchProgressBudget
        # Under the whole-task deadline, four unproductive planning rounds
        # trigger another search direction. Measured progress still renews it.
        self.pickup_progress=SearchProgressBudget(patience=4)
        self.placement_progress=SearchProgressBudget(patience=4)
        from instruction_surface_search import SurfaceSearch
        self.surface_search=SurfaceSearch(inspection_budget=1)
        self.surface_view_prior=None
        self.navigation_deferrals=[]
        self.pickup_geometry_screen_cache={}
        super().__init__(io,perception,output,instruction.pickup_query,dynamic)
        self.event("instruction_received",instruction=instruction.policy_payload())

    def event(self,name,**data):
        super().event(name,task_stage=self.task_stage,**data)

    def deferred_pickup(self,detection):
        if detection is None:return False
        if detection.query==self.query and self.task_stage in ("pickup_search","placement_search"):
            if any(np.linalg.norm(np.asarray(detection.point_world)[:2]-row['point'])<.30
                   for row in self.navigation_deferrals):return True
        if detection.query!=self.instruction.pickup_query:return False
        if any(np.linalg.norm(np.asarray(detection.point_world)[:2]-point)<.30
               for point in self.deferred_pickup_regions):return True
        if self.task_stage!="pickup_search":return False
        key=(detection.observation_id,tuple(detection.box_xyxy))
        if key not in self.pickup_geometry_screen_cache:
            observation=self.memory.observations.get(detection.observation_id)
            if observation is None:return False
            rejected=False
            geometry_error=None
            try:
                tabletop_grasp(observation,detection)
            except ValueError as error:
                geometry_error=str(error)
                # Uncertain support/visibility is not a reason to discard a
                # distant detection. Screen only a measured jaw width well
                # beyond the gripper opening, with a 20 mm observation margin.
                width=getattr(error,"observed_grasp_width_m",None)
                rejected=width is not None and width>.12
                if rejected:
                    self.memory.reject(detection.point_world[:2],detection.observation_id,
                                       radius=.05,query=self.query)
                    self.event("pickup_view_width_rejected",observation_id=detection.observation_id,
                        point_world=detection.point_world,observed_width_m=width,
                        rejection_scope="this location in observations no newer than this view")
            if str(detection.query).strip().lower()=="candle":
                from instruction_grasp_geometry_progress import record_geometry_view
                history,evidence=record_geometry_view(getattr(self,"grasp_geometry_view_history",[]),
                    query=detection.query,point_world=detection.point_world,
                    observation_id=detection.observation_id,observation_step=observation.sim_step,
                    current_step=self.io.step_id,base_xy=observation.base_xyyaw[:2],error=geometry_error)
                self.grasp_geometry_view_history=history
                if evidence is not None:
                    self.navigation_deferrals.append(dict(point=evidence['point_xy'],
                        until_step=self.io.step_id+evidence['defer_controls']))
                    self.event("repeated_ungraspable_views_deferred",**evidence)
                    rejected=True
            self.pickup_geometry_screen_cache[key]=rejected
            if len(self.pickup_geometry_screen_cache)>1024:
                self.pickup_geometry_screen_cache.pop(next(iter(self.pickup_geometry_screen_cache)))
        return self.pickup_geometry_screen_cache[key]

    def check_search_progress(self):
        if self.task_stage not in ("pickup_search","placement_search"):return False
        # A brief detector dropout must get local visual recovery before a
        # global frontier can turn the robot away from a recently seen object.
        if (self.task_stage=="pickup_search" and self.cached is None
                and getattr(self,"recent_pickup_sighting",None) is not None):
            self.recover_recent_pickup()
        progress=self.pickup_progress if self.task_stage=="pickup_search" else self.placement_progress
        self.navigation_deferrals=[row for row in self.navigation_deferrals
                                   if row['until_step']>self.io.step_id]
        self.memory.retrieval_exclusions[self.query]=[
            (row['point'],.30) for row in self.navigation_deferrals]
        from instruction_search_progress import remaining_approach_route,executed_approach_progress
        route_distance=(None if self.cached is None else remaining_approach_route(
            getattr(self,"planned_path",[]),self.io.pose()[:2],self.cached[:2]))
        route_progress=(None if self.cached is None else executed_approach_progress(
            getattr(self,"planned_path",[]),self.io.pose()[:2],self.cached[:2]))
        if not progress.exhausted(self.cached,self.io.pose()[:2],route_distance=route_distance,
                                  route_progress=route_progress):return False
        point=self.cached[:2].copy()
        self.navigation_deferrals.append(dict(point=point.tolist(),until_step=self.io.step_id+4000))
        self.memory.retrieval_exclusions[self.query]=[
            (row['point'],.30) for row in self.navigation_deferrals]
        self.event("unproductive_target_approach_deferred",point_xy=point.tolist(),
            closest_range_m=progress.best_distance,
            iterations_without_improvement=progress.stale_calls,
            minimum_improvement_m=progress.improvement_m,
            defer_controls=4000,evidence="measured approach progress; identity remains unproven")
        # Keep measured pursuit history across a temporary deferral. Otherwise
        # two inaccessible detections can each erase the other's failed route
        # and receive another full pursuit budget when the exclusion expires.
        # Actual approach improvement or target displacement still renews it.
        progress.exhausted(None,self.io.pose()[:2])
        self.cached=None;self.cached_detection=None;self.current_detection=None
        self.search_anchor=None;self.search_region_center=None
        self.exploration_goal=None;self.heading_expansion_goal=None
        return True

    def recover_recent_pickup(self):
        sighting=getattr(self,"recent_pickup_sighting",None)
        if (sighting is None or self.io.step_id-sighting['step']>1200
                or getattr(self,"pickup_recovery_rounds",0)>=2):return None
        point=np.asarray(sighting['point_world'],dtype=float)
        delta=point[:2]-self.io.pose()[:2]
        if np.linalg.norm(delta)>3.5:return None
        self.pickup_recovery_rounds=getattr(self,"pickup_recovery_rounds",0)+1
        self.phase="Look again near the last seen object"
        self.aim_search_view(point)
        pan=float(self.io.body[0])
        tilt=math.atan2(self.focus_camera_height()-point[2],max(.1,np.linalg.norm(delta)))
        for offset in (0.,-.25,.25):
            self.io.body[0]=float(np.clip(pan+offset,-1.3,1.3))
            self.io.body[1]=float(np.clip(tilt,-.3,.9))
            fresh=self.observe()
            self.event("recent_pickup_local_recovery",prior_point_world=point.tolist(),
                observation_id=self.last_obs.frame_id,detected=fresh is not None,
                evidence="recent real RGBD sighting; prior is not a grasp target")
            if fresh is not None:return fresh
        return None

    def placement_track_matches(self,detection,previous):
        if (detection is None or previous is None
                or self.task_stage not in ("placement_search","place")):return True
        if self.memory.detection_rejected(previous):return True
        separation=float(np.linalg.norm(np.asarray(detection.point_world)[:2]-np.asarray(previous.point_world)[:2]))
        if separation<=.45:return True
        self.event("unassociated_placement_candidate",candidate=asdict(detection),
            preserved_track=asdict(previous),separation_m=separation,
            reason="A distant new box does not invalidate the observed destination")
        return False

    def pickup_track_matches(self,detection,previous):
        if self.task_stage!="pickup_search" or detection is None or previous is None:
            return True
        from instruction_target_association import pickup_track_matches
        prior=self.memory.observations.get(previous.observation_id)
        elapsed=(0. if prior is None else max(0.,
            (self.io.step_id-prior.sim_step)/float(self.io.base.control_freq)))
        matches=pickup_track_matches(detection,previous,
            previous_rejected=self.memory.detection_rejected(previous),
            elapsed_seconds=elapsed)
        if not matches:
            self.event("unassociated_pickup_candidate",candidate=asdict(detection),
                preserved_track=asdict(previous),association_radius_m=.45+.12*elapsed,
                observation_age_seconds=elapsed,
                reason="A distant same-noun detection does not establish target displacement")
        return matches

    def defer_ungraspable_pickup(self):
        if self.cached is None or getattr(self,"grasp_closure_attempted",self.grasp_motion_attempted):
            return False
        if self.grasp_motion_attempted:
            # An unreachable open-hand approach is not a completed pickup.
            # Physically recover the compact pose before resuming navigation.
            self.phase="Recover from unreachable grasp"
            self.io.hold_measured_arm()
            # Clear the observed support before folding an open hand after
            # an unreachable approach. A rejected clearance path stops here.
            folded=self.fold_arm_in_place(loaded=False)
            self.event("unreachable_grasp_arm_folded",**folded)
            self.grasp_motion_attempted=False
        point=self.cached[:2].copy()
        self.deferred_pickup_regions.append(point)
        self.memory.reject(point,10**12,query=self.instruction.pickup_query,radius=.30)
        self.event("pickup_candidate_deferred",point_xy=point.tolist(),
                   reason="no executable observed grasp; continue searching",radius_m=.30)
        self.cached=None
        self.cached_detection=None
        self.current_detection=None
        self.search_anchor=None
        self.search_region_center=None
        self.exploration_goal=None
        self.task_stage="pickup_search"
        return True

    def initial_placement_detections(self):
        candidates=self.perception.detect(self.last_obs,self.instruction.placement_query)
        # Reuse calibrated search tiles until a supported destination is remembered.
        # This is an observation prior; placement still requires fresh verification.
        if (not candidates and not self.placement_sightings
                and (self.visual_search_observations+getattr(self,"search_tile_phase_offset",0))%3==0):
            from instruction_visual_search import tiled_detections
            candidates=tiled_detections(self.perception,self.last_obs,
                                        self.instruction.placement_query)
            self.event("initial_placement_multiscale_search",
                observation_id=self.last_obs.frame_id,
                detections=[asdict(d) for d in candidates],
                source="overlapping_calibrated_tiles_of_saved_RGBD")
        return candidates

    def _support_for(self,observation,detection):
        if not self.instruction.support_query:
            return True,None
        supports=self.perception.detect(observation,self.instruction.support_query)
        found=next((s for s in supports if support_relation_matches(observation,detection,s)),None)
        query=str(getattr(detection,"query","")).strip().lower()
        if (found is not None and self.task_stage in ("placement_search","place")
                and self.instruction.placement_relation=="on" and query=="plate"):
            point=np.asarray(detection.point_world,dtype=float)
            conflicts=getattr(self,"receptacle_identity_conflicts",[])
            prior=next((r for r in conflicts if r["query"]==query
                and np.linalg.norm(point[:2]-np.asarray(r["point_xy"]))<=.30),None)
            geometry=None
            error=None
            try:
                _,geometry=receptacle_region(observation,detection,self.instruction.placement_relation,
                    self.perceived_payload_radius,self.perceived_payload_height)
            except ValueError as caught:
                error=str(caught)
            strong=error=="Observed deep receptacle geometry conflicts with a shallow plate"
            # Absence of a deep rim in one partial view does not overturn an
            # earlier positive bowl observation. A later complete shallow fit can.
            if (prior is not None and geometry is not None
                    and observation.frame_id>prior["observation_id"]
                    and 0.<=geometry["observed_surface_height_m"]-geometry["support_height_m"]<=.025
                    and 0.<=geometry["observed_rim_height_m"]-geometry["observed_surface_height_m"]<=.025):
                self.receptacle_identity_conflicts=[r for r in conflicts if r is not prior]
                self.event("receptacle_identity_geometry_revalidated",query=detection.query,
                    observation_id=observation.frame_id,previous_conflict_observation_id=prior["observation_id"],
                    point_world=point.tolist(),evidence="fresh supported shallow surface and rim")
                prior=None
            if strong or prior is not None:
                if prior is None:
                    prior=dict(query=query,point_xy=point[:2].tolist(),observation_id=observation.frame_id)
                    self.receptacle_identity_conflicts=[*conflicts,prior]
                elif strong:
                    prior["observation_id"]=max(prior["observation_id"],observation.frame_id)
                self.memory.reject(point[:2],observation.frame_id,query=detection.query,radius=.30)
                if self.cached is not None and np.linalg.norm(self.cached[:2]-point[:2])<=.30:
                    self.cached=None
                    self.cached_detection=None
                    self.current_detection=None
                    self.search_anchor=None
                    self.search_region_center=None
                    self.exploration_goal=None
                self.event("receptacle_identity_geometry_rejected",
                    observation_id=observation.frame_id,query=detection.query,point_world=point.tolist(),
                    reason=error if strong else "Prior observed deep receptacle requires fresh shallow plate evidence",
                    evidence="observed raised central surface and deep rim" if strong else "retained observed identity contradiction",
                    conflict_observation_id=prior["observation_id"],
                    rejection_max_observation_id=observation.frame_id)
                return False,found
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
        if self.task_stage=="pickup_search":
            from instruction_target_association import order_pickup_candidates
            detection=next((d for d in order_pickup_candidates(self.last_detections,previous_detection)
                if self.pickup_track_matches(d,previous_detection) and not self.deferred_pickup(d)),None)
            self.current_detection=detection
        if self.deferred_pickup(detection):
            detection=next((d for d in self.last_detections if not self.deferred_pickup(d)),None)
            self.current_detection=detection
        prior=previous if previous is not None else getattr(self,"receptacle_view_prior",None)
        if prior is None:prior=getattr(self,"surface_view_prior",None)
        if prior is None and self.task_stage=="pickup_search":
            recent=getattr(self,"recent_pickup_sighting",None)
            if recent is not None and self.io.step_id-recent['step']<=1200:
                prior=np.asarray(recent['point_world'],dtype=float)
        crop_range=3.5 if self.task_stage=="pickup_search" else 1.2
        if (detection is None and self.task_stage in ("pickup_search","grasp","placement_search","place")
                and prior is not None and np.linalg.norm(prior[:2]-self.io.pose()[:2])<crop_range):
            detection=self.focused_target_detection(self.last_obs,prior)
            self.current_detection=detection
        self.visual_search_observations+=1
        if (detection is None and self.task_stage in ("pickup_search","grasp","placement_search","place")
                and (self.task_stage in ("grasp","place") or (self.visual_search_observations+getattr(self,"search_tile_phase_offset",0))%3==0)):
            from instruction_visual_search import tiled_detections
            candidates=[d for d in tiled_detections(self.perception,self.last_obs,self.query)
                        if not self.deferred_pickup(d) and self.pickup_track_matches(d,previous_detection)]
            if (not candidates and self.task_stage=="pickup_search"
                    and previous_detection is None and getattr(self,"recent_pickup_sighting",None) is None
                    and str(self.query).strip().lower()=="egg"
                    and getattr(self,"egg_detail_attempts",0)<60):
                self.egg_detail_attempts=getattr(self,"egg_detail_attempts",0)+1
                from instruction_egg_detail_search import egg_detail_detections
                candidates=[d for d in egg_detail_detections(self.perception,self.last_obs,self.query)
                    if not self.deferred_pickup(d) and self.pickup_track_matches(d,previous_detection)]
                self.event("egg_detail_visual_search",observation_id=self.last_obs.frame_id,
                    attempt=self.egg_detail_attempts,maximum_attempts=60,
                    detections=[asdict(d) for d in candidates],
                    source="fixed_calibrated_224_pixel_crops_with_unchanged_detector_threshold")
            if self.task_stage=="pickup_search":
                candidates=order_pickup_candidates(candidates,previous_detection)
            self.event("multiscale_visual_search",observation_id=self.last_obs.frame_id,
                detections=[asdict(d) for d in candidates],
                source="overlapping_calibrated_tiles_of_saved_RGBD")
            detection=candidates[0] if candidates else None
            self.current_detection=detection
        if not self.pickup_track_matches(detection,previous_detection):
            detection=None
            self.current_detection=None
        if not self.placement_track_matches(detection,previous_detection):
            detection=next((d for d in self.last_detections
                if self.placement_track_matches(d,previous_detection)),None)
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
            for candidate in self.initial_placement_detections():
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
            if self.task_stage=="pickup_search":
                if detection.score>=self.perception.threshold:
                    self.last_confident_pickup_detection=detection
                self.recent_pickup_sighting=dict(point_world=list(detection.point_world),step=self.io.step_id)
                self.pickup_recovery_rounds=0
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
            if hasattr(self,"surface_search"):self.surface_search.active=None
        elif self.task_stage in ("pickup_search","placement_search") and hasattr(self,"surface_search"):
            active=self.surface_search.active
            if self.task_stage=="placement_search":
                from instruction_surface_search import named_support_surfaces
                supports=(self.perception.detect(self.last_obs,self.instruction.support_query)
                          if self.instruction.support_query else [])
                regions=named_support_surfaces(self.last_obs,supports)
                self.surface_search.update(self.last_obs,point_observed_empty,regions=regions)
            else:
                self.surface_search.update(self.last_obs,point_observed_empty)
            if (active is not None and self.surface_search.active is None
                    and self.exploration_goal is not None
                    and np.linalg.norm(self.exploration_goal-active['goal_xy'])<.05):
                self.exploration_goal=None
        if self.observation_listener is not None:
            # One-way observation event; the listener cannot return an answer.
            self.observation_listener(self.last_obs,detection,self.task_stage)
        return detection

    def missing_location_observed(self,point,observation):
        return point_observed_empty(observation,point)

    def focused_target_detection(self,observation,prior):
        cropped=focused_rgbd_crop(observation,prior)
        if cropped is None:return None
        crop,bounds=cropped;x,y,_,_=bounds
        detections=[replace(d,box_xyxy=(np.asarray(d.box_xyxy)+[x,y,x,y]).tolist())
                    for d in self.perception.detect(crop,self.query)]
        detections=[d for d in detections if not self.deferred_pickup(d)]
        if not detections and getattr(self,"task_stage",None) in ("pickup_search","grasp"):
            from instruction_detection_hysteresis import can_recheck_track,continuous_crop_candidates
            anchor=getattr(self,"cached_detection",None)
            if anchor is None or anchor.score<self.perception.threshold:
                anchor=getattr(self,"last_confident_pickup_detection",None)
            old=None if anchor is None else self.memory.observations.get(anchor.observation_id)
            elapsed=(float('inf') if old is None else
                     (self.io.step_id-old.sim_step)/float(self.io.base.control_freq))
            if (anchor is not None and anchor.query==self.query
                    and can_recheck_track(anchor,prior,self.io.pose()[:2],elapsed,self.perception.threshold)
                    and not self.memory.detection_rejected(anchor)
                    and not point_observed_empty(observation,prior)):
                candidates=[replace(d,box_xyxy=(np.asarray(d.box_xyxy)+[x,y,x,y]).tolist())
                            for d in self.perception.detect(crop,self.query,threshold=.10)]
                detections=continuous_crop_candidates([d for d in candidates if not self.deferred_pickup(d)],prior)
                self.event("tracked_crop_recheck",observation_id=observation.frame_id,
                    anchor_observation_id=anchor.observation_id,anchor_age_seconds=elapsed,
                    standard_threshold=self.perception.threshold,recheck_threshold=.10,
                    maximum_displacement_m=.15,detections=[asdict(d) for d in detections],
                    evidence="recent confident RGBD anchor and spatially continuous fresh detection")
        self.event("focused_target_detection",observation_id=observation.frame_id,
            crop_box_xyxy=list(bounds),detections=[asdict(d) for d in detections],
            source="calibrated_crop_of_saved_camera_observation")
        return detections[0] if detections else None

    def focus_camera_height(self):
        # Use measured robot camera calibration, especially after torso motion.
        return float(self.io.camera_pose_cv("fetch_head")[2,3])

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
        self.compact_navigation_arm()
        self.head_first_look_around()
        self.initial_survey_complete=True

    def aim_search_view(self,point):
        """Aim from an observed point, using head travel before base rotation."""
        point=np.asarray(point,dtype=float)
        delta=point[:2]-self.io.pose()[:2]
        bearing=math.atan2(delta[1],delta[0])
        relative=math.atan2(math.sin(bearing-self.io.pose()[2]),math.cos(bearing-self.io.pose()[2]))
        base_turn=False
        if abs(relative)>1.3:
            # Only the angle beyond a comfortable head pan needs base motion.
            yaw=bearing-float(np.clip(relative,-1.,1.))
            self.io.body[0]=0.
            self.settle_head_view()
            base_turn=bool(self.heading_executor().turn(yaw))
        relative=math.atan2(math.sin(bearing-self.io.pose()[2]),math.cos(bearing-self.io.pose()[2]))
        self.io.body[0]=float(np.clip(relative,-1.3,1.3))
        self.io.body[1]=float(np.clip(math.atan2(self.focus_camera_height()-point[2],
                                              np.linalg.norm(delta)),-.3,.9))
        self.event("head_first_target_view",point_world=point.tolist(),
                   head_pan_rad=float(self.io.body[0]),base_turn_completed=base_turn)

    def head_first_look_around(self):
        """Stationary head views first; at most one guarded turn for the rear."""
        self.phase="Observe search region with head"
        found=self.scan_sweep()
        if found is not None:return found
        # Head limits cannot cover directly behind the robot. Refresh depth
        # and use the measured swept-footprint guard, including at startup.
        # This observation path never invokes a retreat-to-turn fallback.
        self.io.body[0]=0.;self.io.body[1]=.25
        self.settle_head_view()
        start=self.io.pose().copy()
        turned=self.heading_executor().turn(float(start[2]+math.pi))
        self.event("head_scan_rear_view",base_turn_completed=bool(turned),
                   maximum_base_turns=1,head_scan_performed_first=True)
        if not turned:return None
        return self.scan_sweep()

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
        # Lower the camera while coordinating the arm to hold the payload pose.
        if (self.task_stage=="placement_search" and self.cached is not None
                and np.linalg.norm(self.cached[:2]-self.io.pose()[:2])<.90):
            self.io.body[2]=min(float(self.io.body[2]),.18)
            self.phase="Lower head viewpoint for receptacle"
            self.io.settle_torso(check=self.require_retained_payload, hold_tcp=False)
            self.event("head_viewpoint_height_recovery",torso_target_m=float(self.io.body[2]),
                       payload_pose_hold_used=False)
            self.focus()
            return super().navigate(budget=min(budget,20),wrist_scans=False)
        return False

    def pick_frontier(self,start,distances,safe_override=None):
        if (self.task_stage in ("pickup_search","placement_search") and self.cached is None
                and self.search_anchor is None and hasattr(self,"surface_search")):
            safe=self.occupancy.traversable() if safe_override is None else safe_override
            excluded=[r['point'] for r in self.navigation_deferrals]
            if self.task_stage=="pickup_search":excluded.extend(self.deferred_pickup_regions)
            region_center=(self.search_region_center
                if self.regional_frontier_queries<self.regional_frontier_limit else None)
            goal=self.surface_search.choose(self.occupancy,safe&(self.frontier_visits<3),
                                            distances,self.io.pose()[:2],excluded,region_center=region_center)
            if goal is not None:
                self.event("observed_surface_inspection_goal",**self.surface_search.active,
                    inspection_budget=self.surface_search.inspection_budget,
                    source="observed_support_surface_exploration_prior_not_an_object_detection")
                return goal
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
        if not self.pickup_track_matches(detection,previous):
            self.cached_detection=previous
            self.cached=np.asarray(previous.point_world,dtype=float)
            self.current_detection=None
            return previous
        if not self.placement_track_matches(detection,previous):
            self.cached_detection=previous
            self.cached=np.asarray(previous.point_world,dtype=float)
            self.current_detection=None
            return previous
        if self.deferred_pickup(detection):
            self.cached=None
            self.cached_detection=None
            self.current_detection=None
            return None
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
                    and getattr(self,"receptacle_arm_views",False)
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
        moved_arm=False
        arm_path=[array(self.io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].copy()]
        try:
            for extension in (.12,.24):
                reach=min(initial_reach+extension,.55)
                goal=np.r_[self.io.pose()[:2]+direction*reach,height]
                self.phase="Clear held-arm camera occlusion"
                moved_arm=True
                self.io.robot.set_control_mode("pd_ee_delta_pose");self.io.robot.controller.reset()
                try:
                    for step in range(360):
                        if step%8==0:self.require_retained_payload()
                        self.hold_release_pose_step(goal,rotation)
                        arm_path.append(array(self.io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].copy())
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
                self.update_carry_navigation_radius()
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
            if moved_arm:
                # A temporary camera-clearing extension must not become the
                # transport posture, including unsuccessful viewpoint trials.
                restored=self.restore_view_arm_path(arm_path)
                self.update_carry_navigation_radius()
                self.event("retained_payload_view_refolded",**restored,
                    navigation_radius_m=self.occupancy.radius)

    def restore_view_arm_path(self,arm_path):
        """Reverse a temporary camera-clearing motion using robot proprioception."""
        self.require_retained_payload()
        self.phase="Restore folded arm after camera view"
        self.io.hold_measured_arm()
        start_base=self.io.pose()[:2].copy()
        start_step=self.io.step_id
        original=np.asarray(arm_path[0],dtype=float)
        # Retain the observed joint path while merging only close samples.
        # No wider generic fold or torso elevation is introduced near a table.
        waypoints=[]
        for waypoint in reversed(arm_path[:-1]):
            point=np.asarray(waypoint,dtype=float)
            if not waypoints or np.max(np.abs(point-waypoints[-1]))>=.04:
                waypoints.append(point)
        if not waypoints or np.max(np.abs(waypoints[-1]-original))>1e-8:waypoints.append(original)
        maximum_error=0.
        for target in waypoints:
            initial=array(self.io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].copy()
            count=max(1,int(math.ceil(1.875*float(np.max(np.abs(target-initial)))/.6*self.io.base.control_freq)))
            for i in range(1,count+1):
                if i%4==1:self.require_retained_payload()
                t=i/count;blend=10*t**3-15*t**4+6*t**5
                self.io.arm=initial+(target-initial)*blend;self.io.command()
                measured=array(self.io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]]
                error=float(np.max(np.abs(measured-self.io.arm)));maximum_error=max(maximum_error,error)
                if error>.10:
                    self.io.hold_measured_arm()
                    raise RuntimeError("View-arm return tracking exceeded0.10rad")
        for _ in range(int(self.io.base.control_freq)):
            self.require_retained_payload();self.io.command()
        measured=array(self.io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]]
        error=float(np.max(np.abs(measured-original)))
        moved=float(np.linalg.norm(self.io.pose()[:2]-start_base))
        if error>.035 or moved>.02:raise RuntimeError("Temporary view motion did not restore stationary arm posture")
        return dict(restored=True,restored_joint_error_rad=error,maximum_tracking_error_rad=maximum_error,
            base_displacement_m=moved,requested_base_translation_m=0.,return_controls=self.io.step_id-start_step)

    def offset_head_view(self,remembered_point,require_geometry=False):
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
                if require_geometry:
                    try:
                        receptacle_region(self.last_obs,result,self.instruction.placement_relation,
                            self.perceived_payload_radius,self.perceived_payload_height)
                    except ValueError as error:
                        self.event("offset_receptacle_geometry_rejected",offset_rad=offset,
                            observation_id=self.last_obs.frame_id,reason=str(error))
                        continue
                return result
        if float(self.io.body[2])>.20:
            # Side views can still see only a nearly top-down, ungrounded
            # receptacle. Use the existing lower-torso head-view recovery now,
            # rather than exhausting the entire navigation budget first.
            # No arm-joint sweep, prior-coordinate release or threshold change.
            self.require_retained_payload()
            self.io.body[2]=.18
            self.phase="Lower head viewpoint for receptacle"
            self.io.settle_torso(check=self.require_retained_payload, hold_tcp=False)
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
                payload_pose_hold_used=False,
                observation_id=self.last_obs.frame_id,detected=result is not None,
                during_local_focus=True)
            return result
        return None

    def inspect_search_surface(self, finish=False):
        active=self.surface_search.active.copy()
        point=np.asarray(active['point_world'])+np.array([0.,0.,.10])
        self.phase="Inspect an observed surface"
        delta=point[:2]-self.io.pose()[:2]
        bearing=math.atan2(delta[1],delta[0])
        relative=math.atan2(math.sin(bearing-self.io.pose()[2]),math.cos(bearing-self.io.pose()[2]))
        self.aim_search_view(point)
        self.surface_view_prior=point
        try:
            result=self.observe()
            self.event("observed_surface_inspected",**active,completed=finish,
                       detected=result is not None,observation_id_used=self.last_obs.frame_id)
        finally:self.surface_view_prior=None
        if finish:self.surface_search.finish(active['region_id'])
        return result

    def fallback_observation(self,moved,stalled,iteration):
        result=super().fallback_observation(moved,stalled,iteration)
        if (not moved and self.exploration_goal is None
                and getattr(getattr(self,"surface_search",None),"active",None) is not None):
            self.surface_search.active=None
        return result

    def look_around(self):
        active=getattr(getattr(self,"surface_search",None),"active",None)
        if (self.task_stage in ("pickup_search","placement_search") and active is not None
                and np.linalg.norm(self.io.pose()[:2]-active['goal_xy'])<.35):
            return self.inspect_search_surface(finish=True)
        return self.head_first_look_around()

    def travel_observation(self):
        """Reuse only a recent complete negative scan after successful travel."""
        previous=getattr(self,'last_negative_travel_scan',None)
        active=getattr(getattr(self,'surface_search',None),'active',None)
        local_prior=(getattr(self,'search_region_center',None) is not None
            and self.regional_frontier_queries<self.regional_frontier_limit)
        if (previous is not None and self.cached is None and self.current_detection is None and self.search_anchor is None
                and active is None and not local_prior and previous['query']==self.query):
            pose=self.io.pose();distance=float(np.linalg.norm(pose[:2]-np.asarray(previous['pose'])[:2]))
            heading=abs(math.atan2(math.sin(pose[2]-previous['pose'][2]),math.cos(pose[2]-previous['pose'][2])))
            elapsed=(self.io.step_id-previous['step'])/float(self.io.base.control_freq)
            if distance<1.2 and heading<math.pi/4 and 0<=elapsed<30.:
                self.event('recent_negative_travel_scan_reused',distance_m=distance,
                    heading_change_rad=heading,elapsed_robot_seconds=elapsed,
                    original_scan_step=previous['step'],query=self.query)
                return None
        return self.scan_sweep()

    def scan_sweep(self):
        self.last_negative_travel_scan=None
        # Preserve a currently visible target before sweeping away from it.
        if self.cached is not None:
            focused=self.focus()
            if focused is not None:
                self.current_detection=focused
                return focused
        active=getattr(getattr(self,"surface_search",None),"active",None)
        if self.task_stage in ("pickup_search","placement_search") and self.cached is None and active is not None:
            return self.inspect_search_surface(finish=False)
        found=super().scan_sweep()
        if found is not None:return found
        # A single shallow pitch omits nearby low surfaces from the image.
        self.phase="Inspect nearby surfaces"
        self.io.body[1]=.65
        detections=[]
        # Continue from the shallow sweep's side. This cyclic view order
        # needs a +2 count offset to preserve which directions receive tiles.
        previous_tile_phase=getattr(self,"search_tile_phase_offset",0)
        self.search_tile_phase_offset=previous_tile_phase+2
        try:
            for pan in (.8,0.,-.8):
                self.io.body[0]=pan
                for _ in range(12):self.io.command()
                result=self.observe()
                if result is not None:
                    detections.append(result)
                    break
        finally:
            self.search_tile_phase_offset=previous_tile_phase
        self.current_detection=max(detections,key=lambda d:d.score) if detections else None
        self.event("near_surface_head_scan",pitch_rad=.65,detected=bool(detections),
                   sensor="fetch_head")
        if self.current_detection is None:
            self.last_negative_travel_scan=dict(pose=self.io.pose().tolist(),query=self.query,step=self.io.step_id)
        return self.current_detection

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
        self.observed_carry_shape=None
        self.pending_flat_carry_shape=None
        retained=super().attempt_grasp()
        if retained and (self.instruction.placement_relation=="in" or self.perceived_payload_height>.20):
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
        if retained and self.perceived_payload_height<=.04:
            fit=next(e for e in reversed(self.events) if e["event"]=="depth_grasp_center")
            approach=next(e for e in reversed(self.events) if e["event"]=="grasp_tcp_approach")
            observation=self.memory.observations[fit["observation_id"]]
            detection=getattr(self,"current_detection",None)
            if detection is not None and detection.observation_id==observation.frame_id:
                from instruction_flat_carry import observed_flat_body
                try:
                    self.pending_flat_carry_shape=observed_flat_body(observation,detection,fit,
                        np.asarray(approach["tcp_matrix"]))
                    self.event("observed_flat_body",**self.pending_flat_carry_shape)
                except ValueError as error:
                    self.event("observed_flat_body_declined",reason=str(error),
                        observation_id=observation.frame_id)
        return retained

    def hold_release_pose_step(self,goal,rotation):
        # Zero delta commands follow the measured tool and permit gravity drift.
        # Keep a fixed world pose with the existing bounded Cartesian feedback.
        before=self.io.step_id
        self.io.move_tcp(goal,tolerance=.0005,steps=1,target_rotation=rotation)
        if self.io.step_id==before:self.io.command_ee()

    def planned_release_tcp(self,release,rotation):
        """Place the observed object centre at the goal, accounting for the grasp."""
        frame=getattr(self,"perceived_payload_frame",None)
        if self.instruction.placement_relation!="on" or frame is None:
            return release.copy(),np.zeros(3)
        offset=np.asarray(frame["center_tcp_m"],dtype=float)
        return release-np.asarray(rotation)@offset,offset

    def align_observed_release(self,release,rotation):
        if self.instruction.placement_relation=="on" and getattr(self,"perceived_payload_frame",None) is not None:
            goal,offset=self.planned_release_tcp(release,rotation)
            self.event("observed_grasp_frame_release",commanded_tcp_world=goal.tolist(),offset_tcp=offset.tolist())
            return goal,offset
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
        frequency=float(self.io.base.control_freq)
        self.settled_release_goal=np.asarray(goal).copy()
        alignment_bias=np.zeros(2)
        correcting_alignment=False
        horizontal_error=None;clearance=None
        for attempt in range(4):
            recovery=attempt>0
            if recovery:
                if release is None or self.instruction.placement_relation!="on":break
                target=min(.385,float(self.io.body[2])+.06)
                if target<=float(self.io.body[2])+.01:break
                self.io.body[2]=target
                self.event("release_alignment_height_recovery",torso_target_m=target)
                self.io.settle_torso(check=self.require_retained_payload,hold_tcp=True)
            stable=0;previous=array(self.io.robot.tcp_pose.p)[0].copy()
            for step in range(int(round((12 if recovery else 2)*frequency))):
                self.require_retained_payload()
                commanded=np.asarray(goal).copy()
                if recovery:
                    matrix=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
                    commanded=release-matrix[:3,:3]@offset
                commanded[:2]+=alignment_bias
                self.hold_release_pose_step(commanded,rotation)
                current=array(self.io.robot.tcp_pose.p)[0].copy()
                speed=float(np.linalg.norm(current-previous)*frequency)
                positioned=True
                if release is not None:
                    matrix=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
                    center=current+matrix[:3,:3]@offset
                    clearance=geometry["observed_safe_radius_m"]-self.perceived_payload_radius-.004
                    horizontal_error=float(np.linalg.norm(center[:2]-release[:2]))
                    if horizontal_error>=min(clearance,.025):
                        correcting_alignment=True
                    alignment_limit=(min(.5*clearance,max(.015,min(.025,clearance/3)))
                                     if correcting_alignment else min(clearance,.025))
                    positioned=(horizontal_error<alignment_limit
                                and abs(center[2]-release[2])<.025)
                    if clearance>0 and speed<.02 and not positioned:
                        # Correct steady Cartesian bias for every observed
                        # receptacle. Keep the original geometric acceptance
                        # checks and limit how far/fast the command can move.
                        delta=.5*(release[:2]-center[:2])/frequency
                        delta*=min(1.,(.003/frequency)/max(np.linalg.norm(delta),1e-12))
                        alignment_bias+=delta
                        maximum_bias=(.01 if clearance<.01 else min(.03,.75*clearance))
                        alignment_bias*=min(1.,maximum_bias/max(np.linalg.norm(alignment_bias),1e-12))
                stable=stable+1 if speed<.01 and positioned else 0
                previous=current
                if stable>=3:
                    self.settled_release_goal=commanded.copy()
                    self.event("release_pose_settled",speed_m_s=speed,controls=step+1,
                               height_recovery=recovery)
                    return True
        self.event("release_pose_unsettled",speed_m_s=speed,maximum_seconds=38.,
            horizontal_error_m=horizontal_error,observed_alignment_clearance_m=clearance,
            bounded_alignment_bias_m=alignment_bias.tolist())
        return False

    def require_retained_payload(self):
        # Robot proprioception only, not the evaluator's actor contact query.
        gap=float(array(self.io.robot.robot.get_qpos())[0,-2:].sum())
        minimum_gap=max(.003,min(.015,.35*getattr(self,"perceived_payload_width",.05)))
        if gap<minimum_gap:
            self.event("payload_lost_by_proprioception",finger_gap_m=gap)
            raise RuntimeError("Closed gripper is empty; abort delivery rather than perform an empty placement")

    def carry_rotation(self,measured_rotation):
        # Preserve the measured tool orientation during retraction.
        return measured_rotation.copy()

    def update_carry_navigation_radius(self):
        # Use the same transformed payload and robot geometry as heading planning.
        from dream_fetch_footprint import capture_robot_footprint
        from instruction_carry_geometry import navigation_payload
        radius,height,frame=navigation_payload(self)
        footprint=capture_robot_footprint(self.io,payload_radius=radius,
            payload_height=height,payload_frame=frame)
        radius=(footprint["maximum_combined_radius_m"]+footprint["padding_m"]
                +math.sqrt(2)*self.occupancy.resolution)
        self.occupancy.radius=max(.40,radius)
        self.event("consistent_carry_navigation_bound",navigation_radius_m=self.occupancy.radius,
                   combined_radius_m=footprint["maximum_combined_radius_m"],
                   padding_m=footprint["padding_m"],resolution_m=self.occupancy.resolution,
                   source="same measured footprint used by heading navigation")
        return self.occupancy.radius

    def reobserve_carried_geometry(self):
        template=getattr(self,"payload_template",None)
        if template is None or self.perceived_payload_height<=.20:return False
        from instruction_carry_geometry import observed_carry_shape
        saved=self.io.body[:2].copy()
        tcp=array(self.io.robot.tcp_pose.p)[0]
        # The head proposal uses the retained grasp and old measured extent;
        # the fit independently observes the actual body axis.
        focus=tcp+np.array([0.,0.,-.5*self.perceived_payload_height])
        delta=focus[:2]-self.io.pose()[:2]
        bearing=math.atan2(delta[1],delta[0])-self.io.pose()[2]
        self.io.body[0]=float(np.clip(math.atan2(math.sin(bearing),math.cos(bearing)),-1.3,1.3))
        self.io.body[1]=float(np.clip(math.atan2(self.focus_camera_height()-focus[2],
                                              np.linalg.norm(delta)),-.3,.9))
        try:
            self.settle_head_view();self.require_retained_payload()
            obs=self.io.capture("fetch_head")
            obs.save(self.output/f"carried_geometry_{obs.frame_id:05d}.npz")
            matrix=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
            shape=observed_carry_shape(obs,template,matrix,
                                      self.perceived_payload_radius,self.perceived_payload_height)
            self.observed_carry_shape=shape
            self.event("observed_carry_geometry",**shape)
            return True
        except ValueError as error:
            self.event("observed_carry_geometry_declined",reason=str(error))
            return False
        finally:
            self.io.body[:2]=saved
            self.settle_head_view();self.require_retained_payload()

    def fold_arm_in_place(self,loaded=False):
        from instruction_arm_return import SelfContactGuard
        try:
            with SelfContactGuard(self.io.base):
                return self._fold_arm_in_place(loaded=loaded)
        finally:
            self.io.hold_measured_arm()

    def _fold_arm_in_place(self,loaded=False):
        """Lift clear of the support, then fold with the base held stationary."""
        check=self.require_retained_payload if loaded else None
        if check is not None:check()
        start_base=self.io.pose().copy()
        initial=array(self.io.robot.tcp_pose.to_transformation_matrix())[0].copy()
        self.phase="Lift arm clear before folding"
        self.io.hold_measured_arm()
        self.io.body[2]=.385
        # Keep arm joint targets during torso elevation so the hand rises with
        # the torso, instead of holding the hand low over the support.
        self.io.settle_torso(check=check)
        # A vertical Cartesian command with fixed wrist orientation can reach
        # an elbow/wrist singularity. Elevate the shoulder using measured arm
        # joints instead; this also pulls the hand toward the base.
        for _ in range(3):
            current=array(self.io.robot.tcp_pose.p)[0].copy()
            rise=float(initial[2,3]+.15-current[2])
            if rise<.03:break
            arm=array(self.io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]].copy()
            limits=array(self.io.robot.robot.get_qlimits())[0,7]
            lever=max(.20,float(np.linalg.norm(current[:2]-self.io.pose()[:2])))
            goal=arm.copy()
            goal[1]=max(float(limits[0])+.02,float(arm[1])-min(.35,(rise+.02)/lever))
            if abs(goal[1]-arm[1])<.005:break
            steps=int(math.ceil(max(1.,1.875*abs(goal[1]-arm[1])/.6)*self.io.base.control_freq))
            for step in range(1,steps+1):
                if check is not None and step%4==1:check()
                t=step/steps;blend=10*t**3-15*t**4+6*t**5
                self.io.arm=arm+(goal-arm)*blend
                self.io.command()
                measured=array(self.io.robot.robot.get_qpos())[0,[5,7,8,9,10,11,12]]
                if float(np.max(np.abs(measured-self.io.arm)))>.25:
                    self.io.hold_measured_arm()
                    raise RuntimeError("Arm lift joint tracking exceeded0.25rad")
            for _ in range(int(.5*self.io.base.control_freq)):
                if check is not None:check()
                self.io.command()
        if array(self.io.robot.tcp_pose.p)[0,2]-initial[2,3]<.12:
            from instruction_lift_recovery import recover_vertical_lift
            self.phase="Recover vertical lift with coordinated arm joints"
            recovery=recover_vertical_lift(self.io,float(initial[2,3]+.15),check=check)
            self.event("vertical_lift_joint_limit_recovery",**recovery)
        if array(self.io.robot.tcp_pose.p)[0,2]-initial[2,3]<.12:
            raise RuntimeError("Arm could not lift clear within its joint limits")
        self.io.hold_measured_arm()
        lifted=array(self.io.robot.tcp_pose.p)[0].copy()
        self.phase="Fold arm with base stationary"
        minimum_tcp_height=float(initial[2,3]+.08)
        if loaded:
            frame=getattr(self,"perceived_payload_frame",None)
            if frame is None:
                raise RuntimeError("Arm return requires the observed payload grasp frame")
            fit=next(e for e in reversed(self.events) if e["event"]=="depth_grasp_center")
            rotation=array(self.io.robot.tcp_pose.to_transformation_matrix())[0,:3,:3]
            offset=rotation@np.asarray(frame["center_tcp_m"])
            axis=rotation@np.asarray(frame["axis_tcp"])
            vertical_extent=(abs(axis[2])*self.perceived_payload_height/2
                +math.sqrt(max(0.,1-axis[2]**2))*self.perceived_payload_radius)
            minimum_tcp_height=float(fit["support_height_m"]+.06-offset[2]+vertical_extent)
        folded=self.io.fold_default_arm(check=check,
            payload_radius=self.perceived_payload_radius if loaded else 0.,
            payload_height=self.perceived_payload_height if loaded else 0., withdraw=True,
            minimum_tcp_height=minimum_tcp_height)
        displacement=float(np.linalg.norm(self.io.pose()[:2]-start_base[:2]))
        self.event("stationary_arm_fold",loaded=loaded,vertical_clearance_gain_m=float(lifted[2]-initial[2,3]),
            base_displacement_m=displacement,requested_base_translation_m=0.,
            maximum_tracking_error_rad=folded['maximum_tracking_error_rad'])
        if displacement>.02:raise RuntimeError("Base moved during the stationary arm fold")
        return folded

    def prepare_carry(self):
        self.require_retained_payload()
        self.io.display_sensor="fetch_head"
        folded=self.fold_arm_in_place(loaded=True)
        self.event("carried_arm_folded",**folded)
        self.io.carry_motion_limited=self.perceived_payload_height>.20
        self.event("carry_motion_limits",enabled=self.io.carry_motion_limited,
            observed_payload_height_m=self.perceived_payload_height,
            maximum_yaw_rate_rad_s=.20,forward_acceleration_m_s2=.10,
            yaw_acceleration_rad_s2=.30,checked_motion_curvature_preserved=True)
        for _ in range(40):self.io.command()
        pending=getattr(self,"pending_flat_carry_shape",None)
        if pending is not None:
            self.observed_carry_shape=pending
            self.event("flat_carry_geometry_activated",**pending)
        self.reobserve_carried_geometry()
        from inspect_fetch_envelope import measure
        envelope=measure(self.io)
        tcp=array(self.io.robot.tcp_pose.p)[0]
        payload_radius=np.linalg.norm(tcp[:2]-self.io.pose()[:2])+self.perceived_payload_radius
        self.update_carry_navigation_radius()
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
        # Flat plates need not protrude enough to enter pickup surface search.
        # Spend at most two attempts approaching freshly observed named tables;
        # the plate still needs its own detection and full support/geometry fit.
        from instruction_surface_search import SurfaceSearch
        self.surface_search=SurfaceSearch(inspection_budget=2,require_protrusions=False)
        self.navigation_deferrals=[]
        self.placement_progress.reset()
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
                self.io.settle_torso(check=self.require_retained_payload, hold_tcp=False)
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
        if remembered is not None:
            detection=self.offset_head_view(remembered,require_geometry=True)
            if detection is not None:
                observation=self.memory.observations[detection.observation_id]
                matched,_=self._support_for(observation,detection)
                if matched:
                    try:
                        release,geometry=receptacle_region(observation,detection,
                            self.instruction.placement_relation,self.perceived_payload_radius,
                            self.perceived_payload_height)
                    except ValueError as error:
                        self.event("placement_recovery_geometry_rejected",
                            observation_id=observation.frame_id,reason=str(error))
                    else:
                        self.event("placement_geometry_view_accepted",view="side_recovery",
                            observation_id=observation.frame_id,
                            torso_height_m=float(array(self.io.robot.robot.get_qpos())[0,3]))
                        return release,geometry
        self.event("placement_aborted_no_valid_observed_geometry")
        return None

    def align_base_before_placement(self,release,geometry):
        """Approach a verified receptacle slightly before extending the held arm."""
        if (not getattr(self,"heading_navigation",False)
                or self.instruction.placement_relation!="on"):
            return None
        clearance=geometry["observed_safe_radius_m"]-self.perceived_payload_radius-.004
        if clearance<=0.:return None
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
                                          payload_height=self.perceived_payload_height,
                                          payload_frame=getattr(self,"perceived_payload_frame",None)),
            self.geometry_observe,self.event,arrival_tolerance_m=.004)
        self.event("observed_placement_base_alignment",planned_advance_m=advance,
                   observed_release_world=np.asarray(release).tolist(),
                   observed_xy_clearance_m=clearance,base=base.tolist())
        completed=executor.translate(goal,planned_heading=base[2])
        self.require_retained_payload()
        self.event("observed_placement_base_alignment_result",completed=completed,
                   measured_advance_m=float(np.linalg.norm(self.io.pose()[:2]-base[:2])),
                   last_failure=executor.last_failure)
        return completed

    def move_release_reference(self,goal,rotation,maximum_speed_m_s=.02):
        initial=array(self.io.robot.tcp_pose.p)[0].copy()
        count=max(40,int(np.ceil(1.875*np.linalg.norm(goal-initial)
            /maximum_speed_m_s*self.io.base.control_freq)))
        self.event("smooth_release_descent",maximum_reference_speed_m_s=maximum_speed_m_s,
                   reference_controls=count,goal_tcp_world=np.asarray(goal).tolist())
        for step in range(1,count+1):
            fraction=step/count
            blend=10*fraction**3-15*fraction**4+6*fraction**5
            if step%8==0:self.require_retained_payload()
            self.hold_release_pose_step(initial+(goal-initial)*blend,rotation)
        for step in range(160):
            if step%8==0:self.require_retained_payload()
            self.hold_release_pose_step(goal,rotation)
        return float(np.linalg.norm(array(self.io.robot.tcp_pose.p)[0]-goal))

    def defer_unusable_receptacle(self):
        """Resume search only after rejecting geometry before placement motion."""
        point=getattr(self,"placement_rejected_candidate",None)
        if point is None or getattr(self,"placement_motion_attempted",False):
            return False
        self.require_retained_payload()
        # Only head/base views preceded this rejection; confirm the existing
        # transport posture without another lift above the old pickup surface.
        self.phase="Fold arm with base stationary"
        folded=self.io.fold_default_arm(check=self.require_retained_payload,
            payload_radius=self.perceived_payload_radius,
            payload_height=self.perceived_payload_height)
        self.event("carried_arm_folded",**folded)
        point=np.asarray(point,dtype=float)[:2].copy()
        self.memory.reject(point,10**12,query=self.instruction.placement_query,radius=.30)
        self.event("placement_candidate_deferred",point_xy=point.tolist(),radius_m=.30,
            reason="No executable observed receptacle geometry; continue searching")
        self.placement_rejected_candidate=None
        return True

    def place_observed(self):
        self.require_retained_payload()
        self.task_stage="place"
        candidate=None if self.cached is None else self.cached.copy()
        self.placement_rejected_candidate=None
        proposal=self.propose_placement()
        if proposal is None:
            self.placement_rejected_candidate=candidate
            return False
        release,geometry=proposal
        self.event("observed_placement_proposal",release_world=release.tolist(),**geometry)
        yaw=math.atan2(*(release[:2]-self.io.pose()[:2])[::-1])
        self.turn_to(yaw)
        distance=float(np.linalg.norm(release[:2]-self.io.pose()[:2]))
        alternative_dock=(getattr(self,"heading_navigation",False)
            and self.instruction.placement_relation=="on" and np.isfinite(distance) and distance>.74
            and geometry["observed_safe_radius_m"]-self.perceived_payload_radius-.004>0.)
        self.align_base_before_placement(release,geometry)
        if alternative_dock:
            from instruction_placement_dock import alternative_placement_dock
            self.phase="Approach an observed alternative placement dock"
            alternative_placement_dock(self,release)
            yaw=math.atan2(*(release[:2]-self.io.pose()[:2])[::-1])
            self.event("observed_alternative_placement_dock_reached",base=self.io.pose().tolist(),
                initial_receptacle_distance_m=distance,lift_before_unfold=True)
        world_yaw=np.array([[math.cos(yaw),-math.sin(yaw),0],
                           [math.sin(yaw),math.cos(yaw),0],[0,0,1]])
        vertical=world_yaw@np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        carry=getattr(self,"observed_carry_shape",None)
        preserve=(carry is not None and self.perceived_payload_height>.20
                  and self.instruction.placement_relation=="on")
        if preserve:
            vertical=array(self.io.robot.tcp_pose.to_transformation_matrix())[0,:3,:3].copy()
            offset=np.asarray(carry["frame"]["center_tcp_m"])
            initial_command=release-vertical@offset
            self.event("observed_carry_release_frame",offset_tcp=offset.tolist(),
                commanded_tcp_world=initial_command.tolist(),source_observation_id=carry["frame"]["observation_id"])
        else:
            initial_command,_=self.planned_release_tcp(release,vertical)
            if alternative_dock:self.io.restore_grasp_seed(grip=-1.,raise_first=True)
            else:self.io.restore_grasp_seed(grip=-1.)
            self.io.position_torso_for_grasp(initial_command[2])
        self.observed_carry_shape=None
        self.io.robot.set_control_mode("pd_ee_delta_pose")
        self.io.robot.controller.reset()
        self.io.display_sensor="fetch_head"
        self.phase="Place in observed receptacle"
        self.placement_motion_attempted=True
        if preserve:
            pre=self.io.move_tcp(initial_command+[0,0,.15],target_rotation=vertical,steps=720)
            self.require_retained_payload()
            residual=self.move_release_reference(initial_command,vertical)
            command_release=initial_command.copy()
        else:
            pre=self.io.move_tcp(initial_command+[0,0,.15],target_rotation=vertical)
            residual=self.io.move_tcp(initial_command,tolerance=.008,target_rotation=vertical)
            command_release,offset=self.align_observed_release(release,vertical)
        if not self.wait_for_release_stillness(command_release,vertical,
                release=release,geometry=geometry,offset=offset):return False
        command_release=self.settled_release_goal.copy()
        tcp_pose=array(self.io.robot.tcp_pose.to_transformation_matrix())[0]
        tcp=tcp_pose[:3,3]
        estimated_center=tcp+tcp_pose[:3,:3]@offset
        clearance=geometry["observed_safe_radius_m"]-self.perceived_payload_radius-.004
        inside=np.linalg.norm(estimated_center[:2]-release[:2])<clearance
        height=abs(estimated_center[2]-release[2])<.025
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
        self.phase="Check released placement"
        for _ in range(60):self.io.command_ee()
        folded=self.fold_arm_in_place(loaded=False)
        self.event("idle_arm_folded",**folded)
        self.task_stage="finished"
        self.event("placement_motion_complete",success_requires_independent_evaluation=True)
        return True
