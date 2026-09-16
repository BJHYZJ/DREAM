"""A bounded observed exit after the ordinary short reverse budget is spent."""
import math
import numpy as np

from dream_fetch_footprint import navigation_layer_views, navigation_swept_clear


def observed_exit_plan(occupancy, footprint, base, maximum_reverse_m=.96):
    """Require every rear segment, turn and forward exit to be observed free."""
    if not np.isfinite(maximum_reverse_m) or not 0 < maximum_reverse_m <= .96:
        raise ValueError('Additional reverse distance must be positive and at most 0.96 m')
    base = np.asarray(base, dtype=float)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError('Expected a finite measured XY/yaw pose')
    views = navigation_layer_views(occupancy, footprint)
    previous = base.copy()
    reverse = []
    for index in range(1, int(math.floor((maximum_reverse_m + 1e-9) / .16)) + 1):
        distance = index * .16
        candidate = base.copy()
        candidate[:2] -= distance * np.array([math.cos(base[2]), math.sin(base[2])])
        if not navigation_swept_clear(occupancy, footprint, previous, candidate, views, require_observed=True):
            break
        reverse.append(candidate.tolist())
        for angle in (math.pi/4, -math.pi/4, math.pi/2, -math.pi/2, math.pi):
            turned = candidate.copy()
            turned[2] += angle
            forward = turned.copy()
            forward[:2] += .32 * np.array([math.cos(turned[2]), math.sin(turned[2])])
            if (navigation_swept_clear(occupancy, footprint, candidate, turned, views, require_observed=True)
                    and navigation_swept_clear(occupancy, footprint, turned, forward, views, require_observed=True)):
                return dict(reverse_poses=reverse, exit_yaw=float(turned[2]),
                    forward_pose=forward.tolist(), additional_reverse_m=distance,
                    evidence='previously observed free full sweep for rear segments, exit turn and forward segment')
        previous = candidate
    return None


def try_observed_exit(pilot):
    """Return measured movement; report full exit completion separately."""
    used = float(getattr(pilot, 'heading_reverse_distance_m', 0.))
    if used < .44 or getattr(pilot, 'heading_observed_exit_attempted', False):
        return False
    allowance = min(.96, 1.44-used)
    if allowance < .16:
        return False
    pilot.heading_observed_exit_attempted = True
    start = pilot.io.pose().copy()
    plan = observed_exit_plan(pilot.occupancy, pilot.measured_navigation_footprint(), start, allowance)
    if plan is None:
        pilot.event('observed_reverse_exit_declined', reason='no_complete_observed_exit',
                    maximum_additional_reverse_m=allowance)
        return False
    pilot.event('observed_reverse_exit_plan', **plan, previous_reverse_budget_m=used,
                maximum_total_reverse_m=1.44, maximum_attempts_without_forward_progress=1)
    # Reserve the entire attempted escape, including a partially blocked one.
    pilot.heading_reverse_distance_m = used + plan['additional_reverse_m']
    pilot.heading_forward_progress_m = 0.
    executor = pilot.heading_executor()
    completed = False
    failure = None
    forward_start = None
    reverse_travelled = 0.
    for pose in plan['reverse_poses']:
        current = pilot.io.pose().copy()
        footprint = pilot.measured_navigation_footprint()
        if not navigation_swept_clear(pilot.occupancy, footprint, current, pose, require_observed=True):
            failure = 'rear_segment_no_longer_observed_free'
            break
        moved = executor.translate(pose[:2], planned_heading=pose[2])
        reverse_travelled += float(np.linalg.norm(pilot.io.pose()[:2]-current[:2]))
        if not moved:
            failure = executor.last_failure
            break
    else:
        current = pilot.io.pose().copy()
        turned = current.copy()
        turned[2] = plan['exit_yaw']
        if not navigation_swept_clear(pilot.occupancy, pilot.measured_navigation_footprint(),
                                      current, turned, require_observed=True):
            failure = 'exit_turn_no_longer_observed_free'
        elif not executor.turn(plan['exit_yaw']):
            failure = executor.last_failure
        else:
            forward_start = pilot.io.pose().copy()
            goal = forward_start.copy()
            goal[:2] += .32*np.array([math.cos(goal[2]), math.sin(goal[2])])
            if not navigation_swept_clear(pilot.occupancy, pilot.measured_navigation_footprint(),
                                          forward_start, goal, require_observed=True):
                failure = 'forward_exit_no_longer_observed_free'
            else:
                completed = bool(executor.translate(goal[:2], planned_heading=goal[2]))
                failure = None if completed else executor.last_failure
    pilot.heading_reverse_distance_m = used + max(plan['additional_reverse_m'], reverse_travelled)
    forward_distance = (0. if forward_start is None else
                        float(np.linalg.norm(pilot.io.pose()[:2]-forward_start[:2])))
    if forward_distance > 0:
        pilot.note_forward_progress(forward_distance)
    if completed:
        old_goal = getattr(pilot, 'exploration_goal', None)
        if old_goal is not None:
            cell = pilot.occupancy.cells(old_goal)
            rr, cc = np.ogrid[:pilot.frontier_visits.shape[0], :pilot.frontier_visits.shape[1]]
            region = (rr-cell[0])**2 + (cc-cell[1])**2 <= (.48/pilot.occupancy.resolution)**2
            pilot.frontier_visits[region] = np.maximum(pilot.frontier_visits[region], 3)
            pilot.event('unproductive_frontier_deferred', point_xy=np.asarray(old_goal).tolist(),
                        scope='exploration_priority_only_obstacles_unchanged')
        pilot.exploration_goal = None
        pilot.heading_expansion_goal = None
    displacement = float(np.linalg.norm(pilot.io.pose()[:2]-start[:2]))
    pilot.event('observed_reverse_exit_result', completed=completed, failure=failure,
                reverse_travelled_m=reverse_travelled, exit_forward_distance_m=forward_distance,
                base_displacement_m=displacement)
    return completed or displacement > .06
