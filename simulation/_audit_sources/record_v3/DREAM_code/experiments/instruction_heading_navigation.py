"""Opt-in Fetch embodiment fallback; all goals come from observed DREAM memory.

The old circular path remains the default in wide passages. When its start or
docking ring is disconnected, propose goals using observed heading feasibility,
then use DREAM's actual A* with sweep-checked rotate/forward edges. The heading
union is only a proposal mask, never an executable path or ground-truth map.
"""
from dataclasses import asdict
import math

import numpy as np

from dream_fetch_footprint import capture_robot_footprint
from dream_fetch_heading_astar import FetchHeadingAStar
from fetch_heading_control import HeadingRouteExecutor,route_chunks,wrapped
from houseexpo_cross_room import weighted_distances
from maniskill_learned_dynamic import motion_detection_changed


def translation_prefix(route):
    """Keep all required preceding turns and at most one short translation."""
    chunks=route_chunks(route)
    for index in range(1,len(chunks)):
        if np.linalg.norm(np.asarray(chunks[index][:2])-chunks[index-1][:2])>1e-7:
            return chunks[:index+1]
    return chunks


class InstructionHeadingNavigation:
    def measured_navigation_footprint(self):
        held=self.task_stage=="placement_search"
        return capture_robot_footprint(self.io,
            payload_radius=self.perceived_payload_radius if held else None,
            payload_height=self.perceived_payload_height if held else None)

    def heading_executor(self,semantic_replan=None):
        def display(phase,waypoint):
            self.phase=phase
            if waypoint is not None:self.waypoint=waypoint
        return HeadingRouteExecutor(self.io,self.occupancy,self.measured_navigation_footprint,
            self.geometry_observe,self.event,semantic_replan,display)

    def try_heading_navigation(self,reason):
        if not getattr(self,"heading_navigation",False):return None
        base=self.io.pose().copy()
        footprint=self.measured_navigation_footprint()
        planner=FetchHeadingAStar(self.occupancy,footprint,base)
        safe=planner.pose_masks.any(axis=0)
        start=tuple(self.occupancy.cells(base[:2]))
        if not safe[start]:
            self.event("heading_start_not_observed_clear",reason=reason)
            return False
        distances,_=weighted_distances(safe,start)
        cells=np.argwhere(safe&np.isfinite(distances))
        candidates=[]
        if self.cached is not None:
            positions=self.occupancy.world(cells)+planner.offset
            offsets=np.linalg.norm(positions-self.cached[:2],axis=1)
            ring=(offsets>=.58)&(offsets<=.64)
            selected=cells[ring]
            order=np.argsort(distances[selected[:,0],selected[:,1]])
            candidates=[(cell,self.cached[:2]) for cell in selected[order]]
            if not candidates:
                closer=(offsets>=.58)&(offsets<np.linalg.norm(self.cached[:2]-base[:2])-.18)
                selected=cells[closer]
                order=np.argsort(offsets[closer]+.04*distances[selected[:,0],selected[:,1]])
                candidates=[(cell,self.cached[:2]) for cell in selected[order]]
        exploring=not candidates
        if exploring:
            goal=self.pick_frontier(start,distances,safe_override=safe)
            if goal is not None:candidates=[(np.asarray(goal),None)]
        route=[];trials=[]
        for cell,look_at in candidates:
            xy=self.occupancy.world(cell)+planner.offset
            desired=math.atan2(*(np.asarray(look_at)-xy)[::-1]) if look_at is not None else base[2]
            headings=[desired] if look_at is not None else sorted(
                [h*math.pi/4 for h in range(8)],key=lambda yaw:abs(wrapped(yaw-base[2])))
            for yaw in headings:
                goal=np.r_[xy,yaw]
                if planner.point_is_occupied(*planner.to_pt(goal)):continue
                proposed=planner.path(goal)
                valid=bool(proposed) and planner.validate_route(proposed)
                trials.append(dict(goal_xyyaw=goal.tolist(),expanded=planner.expanded,
                    budget_exceeded=planner.search_budget_exceeded,swept_valid=valid))
                if valid:route=proposed;break
                if len(trials)>=12:break
            if route or len(trials)>=12:break
        self.event("heading_astar_search",reason=reason,
            planner="dream.motion.algo.a_star.AStar.run_astar",trials=trials,
            circular_radius_unchanged_m=self.occupancy.radius,
            measured_robot_radius_m=footprint["maximum_robot_radius_m"],
            footprint_vertices_base_xy=footprint["vertices_base_xy"],
            grid_offset_xy=planner.offset.tolist(),route_found=bool(route))
        if not route:return False
        self.planned_path=[pose[:2] for pose in route]
        self.navigation_goal=self.planned_path[-1]
        if exploring:self.exploration_goal=np.asarray(self.navigation_goal)
        prefix=translation_prefix(route)
        self.waypoint=prefix[-1][:2]
        self.event("astar_plan",planner="dream.motion.algo.a_star.AStar.run_astar",
            embodiment="measured_heading_footprint",path_xy=self.planned_path,
            path_xyyaw=route,waypoint_xy=self.waypoint,robot_radius_m=self.occupancy.radius)
        tracked=None if self.cached is None else self.cached.copy()
        def observe_change():
            self.io.body[0]=0.;self.io.body[1]=.25 if self.io.grip>=0 else .10
            for _ in range(12):self.io.command()
            self.observe()
            changed=motion_detection_changed(tracked,self.current_detection)
            missing=tracked is not None and self.cached is None
            if changed:self.event("live_detection_during_motion",detection=asdict(self.current_detection))
            return changed or missing
        executor=self.heading_executor(observe_change)
        completed=executor.execute(prefix)
        travelled=float(np.linalg.norm(self.io.pose()[:2]-base[:2]))
        self.event("heading_route_chunk_result",completed=completed,
            last_failure=executor.last_failure,base=self.io.pose().tolist(),travelled_m=travelled)
        # A fresh changed observation requests a new query/plan, not continued
        # motion toward the old target or a navigation success assertion.
        return completed or travelled>.10 or executor.last_failure=="semantic_replan_requested"
