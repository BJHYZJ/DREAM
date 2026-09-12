"""Opt-in Fetch embodiment fallback; all goals come from observed DREAM memory.

The old circular path remains the default for reachable docks and frontiers.
Exploration also needs fallback heading reachability: a circular component alone can
hide a physically accessible doorway before the target has been seen. Propose
goals using observed heading feasibility, then use DREAM's actual A* with
sweep-checked rotate/forward edges. The heading
union is only a proposal mask, never an executable path or ground-truth map.
"""
from dataclasses import asdict
import math

import numpy as np

from dream_fetch_footprint import capture_robot_footprint,swept_clear
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


def candidate_route(planner,candidates,base,*,maximum_trials=12,source="memory_dock"):
    """Bounded heading search; a feasible pose mask alone never certifies a route."""
    trials=[]
    for cell,look_at in candidates:
        xy=planner.occupancy.world(cell)+planner.offset
        desired=math.atan2(*(np.asarray(look_at)-xy)[::-1]) if look_at is not None else base[2]
        headings=[desired] if look_at is not None else sorted(
            [h*math.pi/4 for h in range(8)],key=lambda yaw:abs(wrapped(yaw-base[2])))
        for yaw in headings:
            goal=np.r_[xy,yaw]
            if planner.point_is_occupied(*planner.to_pt(goal)):continue
            proposed=planner.path(goal)
            valid=bool(proposed) and planner.validate_route(proposed)
            trials.append(dict(goal_xyyaw=goal.tolist(),expanded=planner.expanded,
                budget_exceeded=planner.search_budget_exceeded,swept_valid=valid,goal_source=source))
            if valid:return proposed,trials
            if len(trials)>=maximum_trials:return [],trials
    return [],trials


class InstructionHeadingNavigation:
    def note_forward_progress(self,distance):
        # Count measured progress across short route chunks. Replanning does
        # not reset progress, and a turn or stopped controller earns none.
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Forward progress must be a finite nonnegative distance")
        accumulated=getattr(self,"heading_forward_progress_m",0.)+float(distance)
        self.heading_forward_progress_m=accumulated
        if accumulated>=.30:
            self.heading_reverse_recovery_used=False
            self.heading_reverse_distance_m=0.
            self.heading_forward_progress_m=0.

    def try_short_reverse_recovery(self,base,footprint):
        """At most .48 m per escape, checked in <=.16 m observed-free segments."""
        used=getattr(self,"heading_reverse_distance_m",0.)
        allowance=.48-used
        if allowance<.04:
            self.event("heading_reverse_recovery_declined",reason="escape_distance_budget",
                       reverse_distance_m=used)
            return False
        distance=min(.16,allowance)
        goal=base.copy();goal[:2]-=distance*np.array([math.cos(base[2]),math.sin(base[2])])
        # A rearward escape needs the WHOLE swept footprint in already observed
        # space. The forward exploration rule can use fresh front depth; no
        # rear-facing sensor is invented here. Never mutate the observed map.
        observed_only=np.where(self.occupancy.known==0,-1,self.occupancy.known)
        if not swept_clear(observed_only,self.occupancy.origin,self.occupancy.resolution,
                np.asarray(footprint["vertices_base_xy"]),base,goal,padding=footprint["padding_m"],
                inferred=self.occupancy.inferred_blocked):
            self.event("heading_reverse_recovery_declined",reason="rear_sweep_not_observed_free")
            return False
        self.heading_reverse_recovery_used=True
        self.planned_path=[base[:2].tolist(),goal[:2].tolist()]
        self.navigation_goal=goal[:2].tolist();self.waypoint=self.navigation_goal
        self.event("heading_short_reverse_recovery",maximum_distance_m=distance,
                   path_xy=self.planned_path,reason="no_forward_swept_route",
                   rear_evidence="previously_observed_free_full_sweep")
        executor=self.heading_executor()
        completed=executor.translate(goal[:2],planned_heading=base[2])
        travelled=float(np.linalg.norm(self.io.pose()[:2]-base[:2]))
        # Reserve the attempted segment even if its controller stalls: no
        # unlimited repeated escape commands at the same location.
        self.heading_reverse_distance_m=used+max(distance,travelled)
        self.heading_forward_progress_m=0.
        self.event("heading_reverse_recovery_result",completed=completed,travelled_m=travelled,
                   last_failure=executor.last_failure)
        return completed or travelled>.06

    def heading_exploration_candidates(self,planner,safe,distances,start):
        """Retain a last-seen region/active frontier, then original DREAM scoring.

        The temporary proposal mask suppresses duplicate local candidates only;
        it never modifies observed obstacles or A* motion feasibility masks.
        """
        proposal=safe.copy();base=self.io.pose()[:2]
        def exclude(cell):
            row,col=cell
            rr,cc=np.ogrid[:proposal.shape[0],:proposal.shape[1]]
            proposal[(rr-row)**2+(cc-col)**2<=(.32/self.occupancy.resolution)**2]=False
        cells=np.argwhere(proposal&np.isfinite(distances))
        positions=self.occupancy.world(cells)+planner.offset
        anchor=getattr(self,"search_anchor",None)
        active=getattr(self,"exploration_goal",None)
        if anchor is not None and active is None and len(cells):
            offsets=np.linalg.norm(positions-anchor,axis=1)
            index=int(np.argmin(offsets+.01*distances[cells[:,0],cells[:,1]]))
            if offsets[index]<np.linalg.norm(base-anchor)-.12:
                yield cells[index],None
                exclude(cells[index])
            else:self.search_anchor=None
        active=getattr(self,"exploration_goal",None)
        if active is not None:
            cell=self.occupancy.cells(active)
            if self.occupancy.inside(cell) and proposal[tuple(cell)] and np.isfinite(distances[tuple(cell)]):
                yield cell,None
                exclude(cell)
        for _ in range(3):
            goal=self.pick_frontier(start,distances,safe_override=proposal)
            if goal is None:return
            yield np.asarray(goal),None
            exclude(goal)

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

    def try_exploration_expansion(self,start,distances,safe):
        """On repeated blind delivery searches, test frontiers beyond the circle component."""
        if not getattr(self,"heading_navigation",False) or self.task_stage!="placement_search":return None
        pending=getattr(self,"heading_expansion_goal",None)
        if pending is not None:
            if self.cached is not None or self.exploration_goal is None:
                self.heading_expansion_goal=None
            else:
                moved=self.try_heading_navigation("continue_observed_component_expansion")
                self.heading_expansion_goal=(None if self.exploration_goal is None else self.exploration_goal.copy())
                return moved
        if (self.cached is not None or self.exploration_goal is not None or self.search_anchor is not None
                or self.frontier_visits[start]<2):return None
        if (self.search_region_center is not None
                and self.regional_frontier_queries<self.regional_frontier_limit):return None
        if (getattr(self,"heading_expansion_attempts",0)>=4
                or self.io.step_id-getattr(self,"heading_expansion_last_step",-10000)<400):return None
        self.heading_expansion_attempts=getattr(self,"heading_expansion_attempts",0)+1
        self.heading_expansion_last_step=self.io.step_id
        base=self.io.pose().copy();footprint=self.measured_navigation_footprint()
        planner=FetchHeadingAStar(self.occupancy,footprint,base,allow_reverse=False)
        union=planner.pose_masks.any(axis=0)
        if not union[start]:return None
        heading_distances,_=weighted_distances(union,start)
        from scipy.ndimage import distance_transform_edt
        circle_component=safe&np.isfinite(distances)
        # The union proposes destinations only. Full pose and swept-route
        # checks still determine whether any candidate can be executed.
        extra=(union&np.isfinite(heading_distances)
               &(distance_transform_edt(~circle_component)*self.occupancy.resolution>.60))
        candidates=[];proposal=extra.copy()
        for _ in range(3):
            cell=self.pick_frontier(start,heading_distances,safe_override=proposal)
            if cell is None:break
            candidates.append((np.asarray(cell),None))
            rr,cc=np.ogrid[:proposal.shape[0],:proposal.shape[1]]
            proposal[(rr-cell[0])**2+(cc-cell[1])**2<=(.32/self.occupancy.resolution)**2]=False
        route,trials=candidate_route(planner,candidates,base,maximum_trials=12,
            source="observed_frontier_beyond_circular_component")
        self.event("observed_component_expansion",attempt=self.heading_expansion_attempts,
            circular_reachable_cells=int(circle_component.sum()),additional_proposal_cells=int(extra.sum()),
            trials=trials,route_found=bool(route),source="observed_depth_and_measured_robot_footprint")
        if not route:return None
        moved=self.execute_heading_route(base,route,True)
        self.heading_expansion_goal=(None if self.exploration_goal is None else self.exploration_goal.copy())
        return moved

    def try_heading_navigation(self,reason):
        if not getattr(self,"heading_navigation",False):return None
        base=self.io.pose().copy()
        footprint=self.measured_navigation_footprint()
        # DREAM-style search faces the direction of translation. Reverse A*
        # edges can otherwise turn a frontier route into metres of backing up
        # while the only camera faces away. Keep reverse only as bounded escape.
        planner=FetchHeadingAStar(self.occupancy,footprint,base,allow_reverse=False)
        safe=planner.pose_masks.any(axis=0)
        start=tuple(self.occupancy.cells(base[:2]))
        if not safe[start]:
            self.event("heading_start_not_observed_clear",reason=reason)
            return self.try_short_reverse_recovery(base,footprint)
        distances,_=weighted_distances(safe,start)
        cells=np.argwhere(safe&np.isfinite(distances))
        candidates=[]
        if self.cached is not None and self.exploration_goal is None:
            dock_min,dock_max,_=self.docking_limits()
            positions=self.occupancy.world(cells)+planner.offset
            offsets=np.linalg.norm(positions-self.cached[:2],axis=1)
            ring=(offsets>=dock_min)&(offsets<=dock_max)
            selected=cells[ring]
            order=np.argsort(distances[selected[:,0],selected[:,1]])
            candidates=[(cell,self.cached[:2]) for cell in selected[order]]
            if not candidates:
                closer=(offsets>=dock_min)&(offsets<np.linalg.norm(self.cached[:2]-base[:2])-.18)
                selected=cells[closer]
                order=np.argsort(offsets[closer]+.04*distances[selected[:,0],selected[:,1]])
                candidates=[(cell,self.cached[:2]) for cell in selected[order]]
        route,trials=candidate_route(planner,candidates,base)
        exploring=not route
        if exploring:
            proposals=self.heading_exploration_candidates(planner,safe,distances,start)
            route,exploration_trials=candidate_route(planner,proposals,base,source="observed_exploration")
            trials.extend(exploration_trials)
        self.event("heading_astar_search",reason=reason,
            planner="dream.motion.algo.a_star.AStar.run_astar",trials=trials,
            search_motion="rotate_then_forward",reverse_edges_enabled=False,
            circular_radius_unchanged_m=self.occupancy.radius,
            measured_robot_radius_m=footprint["maximum_robot_radius_m"],
            footprint_vertices_base_xy=footprint["vertices_base_xy"],
            grid_offset_xy=planner.offset.tolist(),route_found=bool(route),exploring=exploring)
        if not route:return self.try_short_reverse_recovery(base,footprint)
        return self.execute_heading_route(base,route,exploring)

    def execute_heading_route(self,base,route,exploring):
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
        self.note_forward_progress(travelled)
        self.event("heading_route_chunk_result",completed=completed,
            last_failure=executor.last_failure,base=self.io.pose().tolist(),travelled_m=travelled)
        # A fresh changed observation requests a new query/plan, not continued
        # motion toward the old target or a navigation success assertion.
        return completed or travelled>.10 or executor.last_failure=="semantic_replan_requested"
