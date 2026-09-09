"""Fetch-compatible DREAM A* search, with observed-depth safety replanning.

All task/room annotations remain on the evaluator side. This policy receives
the same RGB-D, learned memory, odometry and task bin as the v2 adapter.
"""
from dataclasses import asdict
import math

import numpy as np
from scipy.ndimage import binary_dilation

from maniskill_learned_dynamic import LearnedSearchPilot, motion_detection_changed
from maniskill_learned_probe import array
from dream_fetch_navigation import FetchObservedMap, FetchAStar
from dream_learned_core import select_frontier_goal
from houseexpo_cross_room import weighted_distances


class CrossRoomSearchPilot(LearnedSearchPilot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Measured folded envelope ~0.338 m; 0.40 m leaves >6 cm nominal
        # clearance. A 4 cm raster reduces doorway quantization error to
        # <2.9 cm, unlike the old 8 cm raster that could seal a valid passage.
        self.occupancy = FetchObservedMap(origin=self.io.pose()[:2]-24.,size=1200,resolution=.04,radius=.40)
        self.planner = FetchAStar(self.occupancy)
        self.planned_path = []
        self.navigation_goal = None
        self.waypoint = None
        self.frontier_visits = np.zeros_like(self.occupancy.known, np.int16)
        self.safe_base_history = []
        self.exploration_goal = None
        self.search_anchor = None
        self.search_region_center = None
        self.regional_frontier_queries = 0

    def focus(self):
        old=None if self.cached is None else self.cached.copy()
        result=super().focus()
        if old is not None and result is None and self.cached is None:
            # A rejected object location is not a valid grasp target. It is
            # still a legitimate last-seen search region, obtained from RGB-D
            # memory rather than the environment's relocated target endpoint.
            self.search_anchor=old[:2].copy()
            self.search_region_center=old[:2].copy()
            self.regional_frontier_queries=0
            self.event("last_seen_search_anchor",point_xy=self.search_anchor.tolist())
        return result

    def look_around(self):
        self.phase="Observe search region"
        yaw=self.io.pose()[2]
        self.io.body[0]=0.
        self.io.body[1]=.25
        for offset in (0.,math.pi/2,math.pi,-math.pi/2):
            self.turn_to(yaw+offset)
            result=self.observe()
            if result is not None:
                return result
        return None

    def robot_filter_spheres(self):
        names=("shoulder_pan_link","shoulder_lift_link","upperarm_roll_link","elbow_flex_link",
               "forearm_roll_link","wrist_flex_link","wrist_roll_link","gripper_link")
        links={link.name:array(link.pose.p)[0] for link in self.io.robot.robot.get_links()}
        points=[links[name] for name in names if name in links]
        spheres=[(point,.11) for point in points]
        for first,second in zip(points,points[1:]):
            spheres.extend((point,.10) for point in np.linspace(first,second,5))
        if self.io.grip < 0:
            spheres.append((array(self.io.robot.tcp_pose.p)[0],.13))
        return spheres

    def observe(self, sensor_name="fetch_head"):
        result=super().observe(sensor_name)
        return result

    def geometry_observe(self):
        # High-frequency navigation sensing is depth only, not a fake VLM
        # update. Retain the exact input and mark it separately in the log.
        obs=self.io.capture("fetch_head")
        obs.save(self.output/f"navigation_depth_{obs.frame_id:05d}.npz")
        self.occupancy.integrate(obs,self_spheres=self.robot_filter_spheres())
        self.event("navigation_depth_update",frame_id=obs.frame_id,sensor=obs.sensor,
                   occupied_voxels=len(self.occupancy.occupied_xyz))

    def compact_navigation_arm(self):
        if self.io.grip < 0:
            return
        if self.io.robot.control_mode != "pd_joint_pos":
            self.io.robot.set_control_mode("pd_joint_pos")
            self.io.robot.controller.reset()
        initial=self.io.arm.copy()
        desired=np.array([-1.2,1.3,0.,1.8,0.,1.2,0.])
        for alpha in np.linspace(0,1,100):
            self.io.arm=initial*(1-alpha)+desired*alpha
            self.io.command()
        self.event("navigation_arm_compacted",measured_envelope_calibration_m=.338,
                   navigation_clearance_radius_m=self.occupancy.radius)

    def survey(self):
        from dream.dynamic_memory import initial_scan_yaws
        self.compact_navigation_arm()
        self.io.body[1]=.25
        yaw=self.io.pose()[2]
        for target_yaw in initial_scan_yaws(yaw):
            self.turn_to(target_yaw)
            self.observe()

    def turn_to(self, yaw):
        try:
            self.io.turn(yaw)
        except RuntimeError as error:
            if "yaw control timed out" not in str(error):
                raise
            self.event("rotation_blocked",goal_yaw=float(yaw))
            self.geometry_observe()
            if not self.retreat(.18):
                raise
            self.io.turn(yaw)

    def retreat(self, distance=.18):
        pose=self.io.pose().copy()
        direction=np.array([math.cos(pose[2]),math.sin(pose[2])])
        goal=pose[:2]-distance*direction
        if not self.occupancy.segment_safe(pose[:2],goal):
            return False
        for _ in range(110):
            if np.linalg.norm(self.io.pose()[:2]-pose[:2])>=distance:
                self.geometry_observe()
                return True
            self.io.command(forward=-.05)
        return np.linalg.norm(self.io.pose()[:2]-pose[:2]) > .06

    def move_chunk(self,path,max_cells=9,stop_on_detection=True,speed=.12,terminal_xy=None):
        max_cells=max(max_cells,int(round(.70/self.occupancy.resolution)))
        if not path or (len(path)<2 and terminal_xy is None):
            return False
        tracked_point=None if self.cached is None else self.cached.copy()
        base=self.io.pose()[:2].copy()
        self.planned_path=[self.occupancy.world(cell).tolist() for cell in path]
        self.navigation_goal=self.planned_path[-1]
        chosen=None
        for cell in path[1:max_cells+1]:
            point=self.occupancy.world(cell)
            if not self.occupancy.segment_safe(base,point):
                break
            chosen=point
        terminal=False
        if terminal_xy is not None and len(path)<=max_cells+1 and self.occupancy.segment_safe(base,terminal_xy):
            chosen=np.asarray(terminal_xy,dtype=float)
            self.planned_path[-1]=chosen.tolist()
            self.navigation_goal=chosen.tolist()
            terminal=True
        if chosen is None or np.linalg.norm(chosen-base)<(.010 if terminal else .025):
            return False
        arrival_tolerance=.012 if terminal else (.025 if np.linalg.norm(chosen-base)<.20 else .055)
        self.waypoint=chosen.tolist()
        self.event("astar_plan",planner="dream.motion.algo.a_star.AStar.run_astar",
                   path_xy=self.planned_path,waypoint_xy=self.waypoint,
                   robot_radius_m=self.occupancy.radius)
        self.turn_to(math.atan2(*(chosen-base)[::-1]))
        self.io.body[0]=0.
        self.io.body[1]=.25 if self.io.grip>=0 else .10
        self.geometry_observe()
        if not self.occupancy.segment_safe(self.io.pose()[:2],chosen):
            self.event("safety_replan_after_turn")
            return False
        initial=self.io.pose()[:2].copy()
        best=float(np.linalg.norm(chosen-initial))
        stagnant=0
        max_steps=max(160,int(best/max(.02,speed)*self.io.base.control_freq*1.8))
        for step in range(max_steps):
            pose=self.io.pose()
            delta=chosen-pose[:2]
            distance=float(np.linalg.norm(delta))
            if distance<arrival_tolerance:
                self.safe_base_history.append(pose[:2].tolist())
                self.observe()
                self.event("executed_observed_route_chunk",goal=chosen.tolist(),base=pose.tolist())
                return True
            if step and step%16==0:
                self.geometry_observe()
                if not self.occupancy.segment_safe(pose[:2],chosen):
                    self.event("safety_stop_new_obstacle",waypoint=chosen.tolist())
                    return np.linalg.norm(pose[:2]-initial)>.10
            if step and step%80==0:
                self.observe()
                if stop_on_detection and motion_detection_changed(tracked_point,self.current_detection):
                    self.event("live_detection_during_motion",detection=asdict(self.current_detection))
                    return True
            heading=math.atan2(delta[1],delta[0])
            error=math.atan2(math.sin(heading-pose[2]),math.cos(heading-pose[2]))
            velocity=min(speed,max(.025,distance*.8)) if abs(error)<.12 else 0.
            self.io.command(forward=velocity,yaw_rate=float(np.clip(2.*error,-.30,.30)))
            if distance<best-.004:
                best=distance
                stagnant=0
            else:
                stagnant+=1
            if stagnant>=55:
                self.event("controller_stall",goal=chosen.tolist())
                self.geometry_observe()
                self.occupancy.infer_stall_obstacle(pose[:2],heading)
                self.retreat(.16)
                return False
        self.event("route_chunk_budget_exhausted")
        return False

    def pick_frontier(self,start,distances):
        safe=self.occupancy.traversable()
        frontier=safe & binary_dilation(self.occupancy.known==0,iterations=2) & np.isfinite(distances)
        frontier &= distances>=int(round(.32/self.occupancy.resolution))
        frontier &= self.frontier_visits<3
        if self.search_region_center is not None and self.regional_frontier_queries<12:
            # Last-seen memory defines a local search prior, not the moved
            # target's location. Avoid immediately abandoning a doorway for
            # an unrelated far frontier after rejecting its old observation.
            cells=np.argwhere(frontier)
            local=np.linalg.norm(self.occupancy.world(cells)-self.search_region_center,axis=1)<=4.
            if local.any():
                frontier[:]=False
                selected=cells[local]
                frontier[selected[:,0],selected[:,1]]=True
            self.regional_frontier_queries+=1
        age=self.occupancy.last_seen.max()-self.occupancy.last_seen.astype(float)
        age/=max(1.,age.max())
        age-=np.minimum(self.frontier_visits,3)*.35
        decision=select_frontier_goal(age,self.occupancy.semantic_field(self.memory,self.query),
                                     frontier,np.array(start),semantic_rate=.1)
        return decision.index

    def navigate(self,budget=90,wrist_scans=False):
        self.compact_navigation_arm()
        stalled=0
        for iteration in range(budget):
            self.phase="Search across rooms"
            if self.exploration_goal is not None and np.linalg.norm(self.io.pose()[:2]-self.exploration_goal)<.22:
                self.event("frontier_goal_reached",point_xy=self.exploration_goal.tolist())
                cell=self.occupancy.cells(self.exploration_goal)
                rr,cc=np.ogrid[:self.frontier_visits.shape[0],:self.frontier_visits.shape[1]]
                visit_radius=int(round(.48/self.occupancy.resolution))
                self.frontier_visits[(rr-cell[0])**2+(cc-cell[1])**2<=visit_radius**2]+=1
                self.exploration_goal=None
                self.look_around()
            if self.search_anchor is not None and np.linalg.norm(self.io.pose()[:2]-self.search_anchor)<.30:
                self.event("last_seen_region_reached",point_xy=self.search_anchor.tolist())
                self.search_anchor=None
                self.look_around()
            result=self.retrieve(use_live=True)
            if result is not None:
                distance=np.linalg.norm(self.cached[:2]-self.io.pose()[:2])
                if distance<2.0:
                    self.focus()
                    result=self.retrieve(use_live=True)
                if result is not None and self.current_detection is not None:
                    xyz=np.asarray(self.current_detection.point_world)
                    if np.linalg.norm(xyz[:2]-self.io.pose()[:2])<.68:
                        self.event("visual_approach_complete",detection=asdict(self.current_detection))
                        self.planned_path=[]
                        self.waypoint=None
                        return True
            self.geometry_observe()
            safe=self.occupancy.traversable()
            start=tuple(self.occupancy.cells(self.io.pose()[:2]))
            if not safe[start]:
                self.event("base_cell_not_depth_navigable",iteration=iteration)
                if self.retreat():
                    continue
                self.scan_sweep()
                if not self.occupancy.traversable()[tuple(self.occupancy.cells(self.io.pose()[:2]))]:
                    return False
                continue
            distances,_=weighted_distances(safe,start)
            goal=None
            if self.cached is not None:
                cells=np.argwhere(np.isfinite(distances)&safe)
                offsets=np.linalg.norm(self.occupancy.world(cells)-self.cached[:2],axis=1)
                ring=(offsets>=.58)&(offsets<=.64)
                if ring.any():
                    choices=cells[ring]
                    goal=tuple(choices[np.argmin(distances[choices[:,0],choices[:,1]])])
                else:
                    current=np.linalg.norm(self.cached[:2]-self.io.pose()[:2])
                    partial=(offsets>=.58)&(offsets<current-.18)
                    if partial.any():
                        choices=cells[partial]
                        cost=offsets[partial]+.04*distances[choices[:,0],choices[:,1]]
                        goal=tuple(choices[np.argmin(cost)])
            exploring=goal is None
            if exploring:
                if self.search_anchor is not None:
                    cells=np.argwhere(np.isfinite(distances)&safe)
                    offsets=np.linalg.norm(self.occupancy.world(cells)-self.search_anchor,axis=1)
                    if len(cells):
                        best=int(np.argmin(offsets+.01*distances[cells[:,0],cells[:,1]]))
                        if offsets[best]<np.linalg.norm(self.io.pose()[:2]-self.search_anchor)-.12:
                            goal=tuple(cells[best])
                        else:
                            self.search_anchor=None
                if goal is None and self.exploration_goal is not None:
                    candidate=tuple(self.occupancy.cells(self.exploration_goal))
                    if safe[candidate] and np.isfinite(distances[candidate]):
                        goal=candidate
                    else:
                        self.exploration_goal=None
                if goal is None:
                    goal=self.pick_frontier(start,distances)
                    if goal is not None:
                        self.exploration_goal=self.occupancy.world(goal)
            if goal is None:
                self.event("no_observed_reachable_goal",iteration=iteration)
                if stalled>=3:
                    return False
                self.scan_sweep()
                stalled+=1
                continue
            path=self.planner.path(self.io.pose()[:2],self.occupancy.world(goal))
            moved=self.move_chunk(path)
            if moved:
                stalled=0
                if wrist_scans and iteration%2==0:
                    self.wrist_sweep()
                elif not wrist_scans:
                    # Fetch has an independently actuated head camera. Keep
                    # the arm folded and gather views while the base is stopped.
                    self.scan_sweep()
            else:
                stalled+=1
                self.event("route_execution_incomplete",iteration=iteration)
                self.scan_sweep()
                if stalled>=8:
                    return False
        return False

    def carry_and_place(self,home_xy,bin_xy,bin_floor=.68):
        # The parent's manipulation selects self.planner when this adapter is
        # present, so its return path also calls the production A*.
        return super().carry_and_place(home_xy,bin_xy,bin_floor)
