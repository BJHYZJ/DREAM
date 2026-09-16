"""Bound pursuit by measured improvement, including oscillating route chunks."""
import numpy as np


class SearchProgressBudget:
    def __init__(self, patience=6, improvement_m=.12, target_change_m=.4):
        self.patience=patience
        self.improvement_m=improvement_m
        self.target_change_m=target_change_m
        self.reset()

    def reset(self):
        self.target=None
        self.best_distance=None
        self.stale_calls=0
        self.history=[]

    def exhausted(self, target, base_xy, route_distance=None, route_progress=None):
        if target is None:
            self.target=None
            return False
        target=np.asarray(target,dtype=float)[:2]
        base_xy=np.asarray(base_xy,dtype=float)
        distance=float(np.linalg.norm(target-base_xy))
        if route_distance is not None and (not np.isfinite(route_distance) or route_distance<0):
            raise ValueError('Route distance must be finite and nonnegative')
        if route_progress is not None and (not np.isfinite(route_progress) or route_progress<0):
            raise ValueError('Executed route progress must be finite and nonnegative')
        record=next((row for row in self.history
                     if np.linalg.norm(target-row['target'])<=self.target_change_m),None)
        if record is None:
            record=dict(target=target.copy(),best=distance,stale=0,
                        route_best=route_distance,route_base=base_xy.copy(),
                        credited_positions=[base_xy.copy()])
            self.history.append(record)
            self.history=self.history[-64:]
        else:
            route_improved=(route_distance is not None and record['route_best'] is not None
                and route_distance<record['route_best']-self.improvement_m
                and np.linalg.norm(base_xy-record['route_base'])>=self.improvement_m/2)
            # A newly planned detour may be longer than an older blocked
            # route. Credit motion along that checked approach route only at
            # a new measured base location; repeated oscillations earn none.
            novel_route_progress=(route_progress is not None and route_progress>=self.improvement_m
                and all(np.linalg.norm(base_xy-old)>=self.improvement_m
                        for old in record['credited_positions']))
            if distance<record['best']-self.improvement_m or route_improved or novel_route_progress:
                record['best']=min(distance,record['best']);record['stale']=0
                record['credited_positions'].append(base_xy.copy())
            else:
                record['stale']+=1
            if route_distance is not None and (record['route_best'] is None or route_improved):
                record['route_best']=route_distance;record['route_base']=base_xy.copy()
        self.target=record['target']
        self.best_distance=record['best']
        self.stale_calls=record['stale']
        return self.stale_calls>=self.patience


def remaining_approach_route(path, base_xy, target_xy, maximum_dock_range=1., maximum_off_path=.25):
    """Remaining length of an observed route that actually terminates at the target.

    Unrelated exploration frontiers are excluded. Projection uses the nearest
    route segment; leaving that route is not credited as approach progress.
    """
    points=np.asarray(path,dtype=float)
    if points.ndim!=2 or points.shape[0]<2 or points.shape[1]!=2:return None
    base=np.asarray(base_xy,dtype=float)[:2];target=np.asarray(target_xy,dtype=float)[:2]
    if not np.isfinite(points).all() or np.linalg.norm(points[-1]-target)>maximum_dock_range:return None
    delta=points[1:]-points[:-1];lengths=np.linalg.norm(delta,axis=1)
    valid=lengths>1e-8
    if not valid.any():return None
    fractions=np.zeros(len(lengths))
    fractions[valid]=np.clip(np.sum((base-points[:-1])[valid]*delta[valid],axis=1)/lengths[valid]**2,0.,1.)
    projections=points[:-1]+fractions[:,None]*delta
    offsets=np.linalg.norm(projections-base,axis=1);offsets[~valid]=np.inf
    index=int(np.argmin(offsets))
    if offsets[index]>maximum_off_path:return None
    return float(offsets[index]+(1-fractions[index])*lengths[index]+lengths[index+1:].sum())


def executed_approach_progress(path,base_xy,target_xy):
    remaining=remaining_approach_route(path,base_xy,target_xy)
    if remaining is None:return None
    points=np.asarray(path,dtype=float)
    length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    return max(0.,length-remaining)
