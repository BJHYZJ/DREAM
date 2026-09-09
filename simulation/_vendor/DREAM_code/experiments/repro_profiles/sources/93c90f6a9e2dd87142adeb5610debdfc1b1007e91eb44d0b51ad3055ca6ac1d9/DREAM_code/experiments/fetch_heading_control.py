"""Measured-pose execution of heading-state routes; no task actor inputs."""
import math

import numpy as np

from dream_fetch_footprint import swept_clear


def wrapped(angle):
    return math.atan2(math.sin(angle),math.cos(angle))


def route_chunks(route,max_distance=.65,max_reverse_distance=.18):
    """Merge only collinear fixed-heading translations, never skip a turn."""
    if not route:return []
    result=[np.asarray(route[0],dtype=float)];anchor=result[0]
    for i in range(1,len(route)):
        previous=np.asarray(route[i-1]);current=np.asarray(route[i])
        rotation=abs(wrapped(current[2]-previous[2]))>1e-7
        next_rotates=(i==len(route)-1 or abs(wrapped(route[i+1][2]-current[2]))>1e-7)
        delta=current[:2]-previous[:2]
        reverse=np.dot(delta,[math.cos(current[2]),math.sin(current[2])])<0
        next_reverses=(i<len(route)-1 and np.dot(delta,np.asarray(route[i+1][:2])-current[:2])<0)
        limit=max_reverse_distance if reverse else max_distance
        if rotation or next_rotates or next_reverses or np.linalg.norm(current[:2]-anchor[:2])>=limit:
            result.append(current);anchor=current
    return [p.tolist() for p in result]


class HeadingRouteExecutor:
    """PD feedback with measured footprint and fresh-depth guard callbacks."""
    def __init__(self,io,occupancy,footprint_provider,geometry_observe,event=lambda *a,**k:None,
                 semantic_replan=None,state_listener=None):
        self.io=io;self.occupancy=occupancy;self.footprint_provider=footprint_provider
        self.geometry_observe=geometry_observe;self.event=event
        self.phase="Heading navigation diagnostic";self.waypoint=None;self.last_failure=None
        self.semantic_replan=semantic_replan;self.state_listener=state_listener

    def command(self,**kwargs):
        if self.state_listener is not None:self.state_listener(self.phase,self.waypoint)
        self.io.command(**kwargs)

    def clear(self,start,end):
        footprint=self.footprint_provider()
        return swept_clear(self.occupancy.known,self.occupancy.origin,self.occupancy.resolution,
            np.asarray(footprint["vertices_base_xy"]),start,end,padding=footprint["padding_m"],
            inferred=self.occupancy.inferred_blocked)

    def stop(self,reason):
        self.last_failure=reason;self.event("heading_motion_stopped",reason=reason,base=self.io.pose().tolist())
        # Apply a real zero-velocity command, never a pose correction.
        self.command()
        return False

    def turn(self,yaw):
        self.phase="Turn in observed clearance"
        start=self.io.pose();end=start.copy();end[2]=yaw
        if abs(wrapped(yaw-start[2]))<.02:return True
        self.geometry_observe()
        if not self.clear(start,end):return self.stop("rotation_sweep_not_observed_clear")
        for step in range(450):
            pose=self.io.pose();error=wrapped(yaw-pose[2])
            if abs(error)<.02:return True
            if step%8==0:self.geometry_observe()
            rate=float(np.clip(1.6*error,-.45,.45));predicted=pose.copy()
            predicted[2]+=rate/self.io.base.control_freq*4
            if not self.clear(pose,predicted):return self.stop("fresh_depth_blocks_rotation")
            self.command(yaw_rate=rate)
        return self.stop("rotation_control_budget")

    def translate(self,goal,planned_heading=None):
        goal=np.asarray(goal);self.waypoint=goal.tolist()
        base=self.io.pose();delta=goal-base[:2]
        if np.linalg.norm(delta)<.018:return True
        yaw=math.atan2(delta[1],delta[0]) if planned_heading is None else float(planned_heading)
        reverse=np.dot(delta,[math.cos(yaw),math.sin(yaw)])<0
        if not self.turn(yaw):return False
        # Fresh semantic evidence may trigger environmental motion, or show a
        # moved/lost target. No environment answer is returned to this callback.
        if self.semantic_replan is not None and self.semantic_replan():
            return self.stop("semantic_replan_requested")
        self.phase="Reverse through observed clearance" if reverse else "Translate through observed passage"
        self.geometry_observe()
        initial=self.io.pose();distance=np.linalg.norm(goal-initial[:2])
        endpoint=np.r_[goal,yaw]
        if not self.clear(initial,endpoint):return self.stop("translation_sweep_not_observed_clear")
        best=distance;stagnant=0
        for step in range(max(180,int(distance/.08*self.io.base.control_freq*2))):
            pose=self.io.pose();delta=goal-pose[:2];remaining=np.linalg.norm(delta)
            if remaining<.018:
                self.command();self.geometry_observe()
                self.event("heading_translation_complete",goal=goal.tolist(),base=self.io.pose().tolist())
                return True
            if step%8==0:self.geometry_observe()
            if step and step%80==0 and self.semantic_replan is not None and self.semantic_replan():
                return self.stop("semantic_replan_requested")
            error=wrapped(math.atan2(delta[1],delta[0])+(math.pi if reverse else 0)-pose[2])
            speed=min(.06 if reverse else .10,max(.02,remaining*.65)) if abs(error)<.12 else 0.
            if reverse:speed=-speed
            rate=float(np.clip(1.8*error,-.25,.25));predicted=pose.copy()
            predicted[:2]+=speed/self.io.base.control_freq*4*np.array([math.cos(pose[2]),math.sin(pose[2])])
            predicted[2]+=rate/self.io.base.control_freq*4
            if not self.clear(pose,predicted):return self.stop("fresh_depth_blocks_translation")
            self.command(forward=speed,yaw_rate=rate)
            if remaining<best-.003:best=remaining;stagnant=0
            else:stagnant+=1
            if stagnant>70:return self.stop("translation_controller_stall")
        return self.stop("translation_control_budget")

    def execute(self,route):
        chunks=route_chunks(route)
        for previous,current in zip(chunks,chunks[1:]):
            if np.linalg.norm(np.asarray(previous[:2])-current[:2])<1e-7:
                if not self.turn(current[2]):return False
            elif not self.translate(current[:2],planned_heading=current[2]):return False
        return bool(chunks)
