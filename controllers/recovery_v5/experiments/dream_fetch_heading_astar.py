"""Heading-state embodiment adapter using DREAM's original AStar.run_astar.

Edges are in-place rotations or fixed-heading forward/reverse translations. Robot collision geometry
is measured separately; only observed occupancy and optional proprioceptive
stall evidence enter this planner. It does not move the robot or certify a task.
The original 2-D A* default is unchanged by its neighbor hook.
"""
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dream.motion.algo.a_star import AStar
from dream_fetch_footprint import orientation_masks,swept_mask,swept_clear,navigation_layer_views,navigation_swept_clear


class SearchBudgetExceeded(RuntimeError):
    pass


class FetchHeadingAStar(AStar):
    def __init__(self,occupancy,footprint,start_pose,max_expanded=250000,allow_reverse=True):
        self.occupancy=occupancy
        self.footprint=footprint
        self.layer_views=navigation_layer_views(occupancy,footprint)
        self.vertices=np.asarray(footprint["vertices_base_xy"],dtype=float)
        self.padding=float(footprint["padding_m"])
        self.start_pose=np.asarray(start_pose,dtype=float)
        start_cell=occupancy.cells(self.start_pose[:2])
        # Anchor state coordinates exactly at the measured start, not at a
        # snapped grid center. Collision kernels include this subcell offset.
        self.offset=self.start_pose[:2]-occupancy.world(start_cell)
        self.max_expanded=max_expanded;self.expanded=0;self.search_budget_exceeded=False
        self.allow_reverse=allow_reverse
        self.steps=[(int(round(math.sin(h*math.pi/4))),int(round(math.cos(h*math.pi/4)))) for h in range(8)]
        settings=[dict(known=known,resolution=occupancy.resolution,vertices=vertices,
                       padding=self.padding,inferred=occupancy.inferred_blocked,offset=self.offset)
                  for known,vertices in self.layer_views]
        self.pose_masks=np.logical_and.reduce([orientation_masks(**common) for common in settings])
        self.forward_masks=[];self.backward_masks=[];self.turn_ccw_masks=[]
        for h,(dr,dc) in enumerate(self.steps):
            self.forward_masks.append(np.logical_and.reduce([swept_mask(**common,start_yaw=h*math.pi/4,
                translation=np.array([dc,dr])*occupancy.resolution) for common in settings]))
            if allow_reverse:
                self.backward_masks.append(np.logical_and.reduce([swept_mask(**common,start_yaw=h*math.pi/4,
                    translation=-np.array([dc,dr])*occupancy.resolution) for common in settings]))
            self.turn_ccw_masks.append(np.logical_and.reduce([
                swept_mask(**common,start_yaw=h*math.pi/4,angle=math.pi/4) for common in settings]))
        self.forward_masks=np.asarray(self.forward_masks);self.turn_ccw_masks=np.asarray(self.turn_ccw_masks)
        self.backward_masks=np.asarray(self.backward_masks)
        space=SimpleNamespace(voxel_map=SimpleNamespace(get_2d_map=self._map),
                              is_valid=lambda pose:not self.point_is_occupied(*self.to_pt(pose)))
        super().__init__(space)

    def _map(self):
        return ~self.pose_masks.any(axis=0),self.occupancy.known!=0

    def to_pt(self,pose):
        pose=np.asarray(pose)
        xy=np.rint((pose[:2]-self.occupancy.origin-self.offset)/self.occupancy.resolution-.5).astype(int)
        heading=int(round(pose[2]/(math.pi/4)))%8
        return int(xy[1]),int(xy[0]),heading

    def to_xy(self,state):
        return (*self.occupancy.world(state[:2])+self.offset,state[2]*math.pi/4)

    def point_is_occupied(self,row,col,heading):
        return (row<0 or col<0 or row>=self.occupancy.known.shape[0]
                or col>=self.occupancy.known.shape[1] or not self.pose_masks[heading,row,col])

    def get_unoccupied_neighbor(self,pt,goal_pt=None):
        # No endpoint snapping or implicit change to an occupied pose.
        return None if self.point_is_occupied(*pt) else pt

    def neighbor_points(self,state):
        self.expanded+=1
        if self.expanded>self.max_expanded:
            raise SearchBudgetExceeded("Heading-state expansion budget exhausted")
        row,col,heading=state;dr,dc=self.steps[heading]
        result=[(row,col,(heading+1)%8),(row,col,(heading-1)%8),(row+dr,col+dc,heading)]
        if self.allow_reverse:result.append((row-dr,col-dc,heading))
        return result

    def edge_is_valid(self,start,end):
        if self.point_is_occupied(*end):return False
        row,col,heading=start
        if start[:2]==end[:2]:
            if end[2]==(heading+1)%8:return bool(self.turn_ccw_masks[heading,row,col])
            if end[2]==(heading-1)%8:return bool(self.turn_ccw_masks[end[2],row,col])
            return False
        dr,dc=self.steps[heading]
        reverse=end==(row-dr,col-dc,heading)
        if reverse:
            if not self.allow_reverse:return False
            dr,dc=-dr,-dc
        elif end!=(row+dr,col+dc,heading):return False
        if dr and dc and (self.occupancy.known[row+dr,col]!=1 or self.occupancy.known[row,col+dc]!=1):
            return False
        return bool((self.backward_masks if reverse else self.forward_masks)[heading,row,col])

    def compute_heuristic(self,a,b,weight=6,avoid=3):
        angular=min((a[2]-b[2])%8,(b[2]-a[2])%8)*math.pi/4
        return math.hypot(a[0]-b[0],a[1]-b[1])+(.30/self.occupancy.resolution)*angular

    def path(self,goal_pose):
        self.expanded=0;self.search_budget_exceeded=False
        goal_pose=np.asarray(goal_pose,dtype=float)
        snapped_start=np.asarray(self.to_xy(self.to_pt(self.start_pose)))
        if not navigation_swept_clear(self.occupancy,self.footprint,self.start_pose,snapped_start,
                                      views=self.layer_views):return []
        try:
            route=self.run_astar(tuple(snapped_start),tuple(goal_pose),remove_line_of_sight_points=False)
        except SearchBudgetExceeded:
            self.search_budget_exceeded=True;return []
        if route is None:return []
        final=np.asarray(route[-1]);oriented=final.copy();oriented[2]=goal_pose[2]
        if not navigation_swept_clear(self.occupancy,self.footprint,final,oriented,
                                      views=self.layer_views):return []
        result=[self.start_pose.tolist()]
        if np.linalg.norm(self.start_pose-snapped_start)>1e-9:result.append(snapped_start.tolist())
        result.extend([list(p) for p in route[1:]])
        if np.linalg.norm(final-oriented)>1e-9:result.append(oriented.tolist())
        return result

    def validate_route(self,route):
        if not route:return False
        return all(navigation_swept_clear(self.occupancy,self.footprint,a,b,views=self.layer_views)
            for a,b in zip(route,route[1:]))
