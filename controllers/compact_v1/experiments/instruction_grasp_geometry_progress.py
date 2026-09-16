"""Prioritize another location after repeated nearby unexecutable candle fits."""
import numpy as np

def record_geometry_view(history, *, query, point_world, observation_id,
                         observation_step, current_step, base_xy, error):
    history=[r for r in history if current_step-r['last_step']<=1200]
    if str(query).strip().lower()!='candle':return history,None
    point=np.asarray(point_world,dtype=float)[:2]
    if point.shape!=(2,) or not np.isfinite(point).all():return history,None
    near=next((r for r in history if np.linalg.norm(point-np.asarray(r['point_xy']))<.20),None)
    if error is None:
        return [r for r in history if r is not near],None
    if error!='No observed top-clear jaw section; acquire another view':return history,None
    if not 0<=current_step-observation_step<=200:return history,None
    if np.linalg.norm(point-np.asarray(base_xy)[:2])>1.2:return history,None
    if near is None:
        near=dict(point_xy=point.tolist(),last_step=observation_step,views=[])
        history.append(near)
    if any(v['observation_id']==observation_id for v in near['views']):return history,None
    if near['views'] and observation_step-near['views'][-1]['step']<20:return history,None
    near['views'].append(dict(observation_id=observation_id,step=observation_step))
    near['last_step']=observation_step
    if len(near['views'])<3:return history,None
    evidence=dict(point_xy=point.tolist(),observation_ids=[v['observation_id'] for v in near['views']],
        distinct_nearby_views=len(near['views']),reason=error,defer_controls=4000,
        boundary='Temporary search priority only; fresh geometry and all task checks remain required.')
    return [r for r in history if r is not near],evidence
