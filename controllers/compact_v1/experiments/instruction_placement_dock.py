"""Reposition a held payload using only accumulated RGB-D and robot state."""
import numpy as np
from dream_fetch_footprint import capture_robot_footprint
from fetch_heading_control import HeadingRouteExecutor


def alternative_placement_dock(policy,release):
 io=policy.io;occupancy=policy.occupancy
 from dream_fetch_heading_astar import FetchHeadingAStar
 from instruction_heading_navigation import candidate_route
 policy.geometry_observe()
 for reverse_first in (False,True):
  base_pose=io.pose().copy()
  if reverse_first:
   backwards=base_pose[:2]-.18*np.array([np.cos(base_pose[2]),np.sin(base_pose[2])])
   executor=HeadingRouteExecutor(io,occupancy,lambda:capture_robot_footprint(io,payload_radius=policy.perceived_payload_radius,payload_height=policy.perceived_payload_height,payload_frame=policy.perceived_payload_frame),policy.geometry_observe,policy.event,arrival_tolerance_m=.006)
   policy.event('observed_placement_dock_reverse',maximum_distance_m=.18)
   if not executor.translate(backwards,planned_heading=base_pose[2]):raise RuntimeError('Observed reverse preparation rejected: '+str(executor.last_failure))
   base_pose=io.pose().copy()
  footprint=capture_robot_footprint(io,payload_radius=policy.perceived_payload_radius,payload_height=policy.perceived_payload_height,payload_frame=policy.perceived_payload_frame)
  planner=FetchHeadingAStar(occupancy,footprint,base_pose,allow_reverse=False)
  reachable=planner.reachable_pose_masks().any(axis=0)
  cells=np.argwhere(reachable);positions=occupancy.world(cells)+planner.offset
  radius=np.linalg.norm(positions-release[:2],axis=1);displacement=np.linalg.norm(positions-base_pose[:2],axis=1)
  selected=(radius>=.60)&(radius<=.72)&(displacement>=.24)
  # Positive observed obstacles along a proposed hand extension veto the dock.
  # This is a candidate filter, not a claim that unseen glass is free space.
  points=occupancy.occupied_xyz
  points=points[(points[:,2]>=release[2]+.03)&(points[:,2]<=release[2]+.40)]
  rejected=0
  for index in np.flatnonzero(selected):
   start=positions[index];delta=release[:2]-start;length=np.linalg.norm(delta);direction=delta/length
   local=points[:,:2]-start;along=local@direction;across=np.abs(local[:,0]*direction[1]-local[:,1]*direction[0])
   blocked=(along>.12)&(along<length-.08)&(across<.12)
   if blocked.any():selected[index]=False;rejected+=1
  policy.event('observed_placement_hand_corridor_filter',rejected_candidates=rejected,remaining_candidates=int(selected.sum()),half_width_m=.12,height_above_release_m=[.03,.40])
  chosen=cells[selected];score=displacement[selected]+.5*np.abs(radius[selected]-.66)
  route,trials=candidate_route(planner,[(cell,release[:2]) for cell in chosen[np.argsort(score)]],base_pose,maximum_trials=24,source='observed_placement_alternative')
  policy.event('observed_placement_alternative_dock_plan',reverse_first=reverse_first,candidates=len(chosen),route=route,trials=trials)
  if not route:continue
  route_length=sum(np.linalg.norm(np.asarray(x[:2])-y[:2]) for x,y in zip(route,route[1:]))
  if route_length>2.5:raise RuntimeError('Alternative observed dock route exceeds2.5m bound')
  executor=HeadingRouteExecutor(io,occupancy,lambda:capture_robot_footprint(io,payload_radius=policy.perceived_payload_radius,payload_height=policy.perceived_payload_height,payload_frame=policy.perceived_payload_frame),policy.geometry_observe,policy.event,arrival_tolerance_m=.006)
  alignment_completed=executor.execute(route);policy.require_retained_payload()
  if not alignment_completed:raise RuntimeError('Observed alternative route rejected: '+str(executor.last_failure))
  return True
 raise RuntimeError('No observed reachable alternative placement dock')
