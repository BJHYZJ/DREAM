"""Private RGBD jaw-axis trial; all existing motion checks remain necessary."""
import numpy as np


def pad_observations(world, target, axis):
    """Count observed points inside the open Fetch fingers' downward sweep."""
    along = np.array([-axis[1], axis[0]])
    local = (world[:, :2] - target[:2]) @ np.column_stack((axis, along))
    vertical = (world[:, 2] >= target[2] - .031) & (world[:, 2] <= target[2] + .329)
    # Public Fetch finger collision dimensions, in metres, at full opening.
    return [int(np.count_nonzero(vertical & (sign * local[:, 0] >= .04970)
            & (sign * local[:, 0] <= .06400) & (np.abs(local[:, 1]) <= .01315)))
            for sign in (-1, 1)]


def prefer_less_obstructed_axis(observation, target, fit, shape):
    """Reconsider only a short object's optional lateral-axis substitution.

    Absence of observed points does not establish free space. This chooses
    between two already fitted axes, retaining all alignment/contact guards.
    Neither the fitted target center nor its grasp height is changed.
    """
    if (fit.get('jaw_axis_choice') != 'observed_feasible_robot_lateral_axis'
            or not .04 < fit['height_m'] <= .12
            or fit.get('grasp_section_points') is not None
            or fit.get('grasp_column_extended', False)):
        return target, fit
    original = np.asarray(fit.get('original_closing_axis_xy'), dtype=float)
    lateral = np.asarray(fit['closing_axis_xy'], dtype=float)
    if (original.shape != (2,) or lateral.shape != (2,)
            or not np.isfinite(original).all() or not np.isfinite(lateral).all()
            or abs(np.linalg.norm(original) - 1.) > 1e-5
            or abs(np.linalg.norm(lateral) - 1.) > 1e-5):
        return target, fit
    points = np.asarray(shape.points)
    if len(points) < 48 or not np.isfinite(points).all():
        return target, fit
    width = float(np.diff(np.quantile(points[:, :2] @ original, [.02, .98]))[0])
    if not .012 < width < min(.094, fit['closing_width_m'] - .006):
        return target, fit
    world = observation.world_points().reshape(-1, 3)
    depth = observation.depth_m.reshape(-1)
    world = world[np.isfinite(world).all(-1) & np.isfinite(depth) & (depth > .10) & (depth < 4.)]
    lateral_counts = pad_observations(world, target, lateral)
    original_counts = pad_observations(world, target, original)
    # The original fitted axis can provide the 6 mm per-finger allowance
    # already used by gripper preshaping when the optional lateral axis
    # consumes it. This only ranks fitted axes; it does not prove free space.
    clearance_width = 2 * .04970 - 2 * .006
    if (fit['closing_width_m'] > clearance_width >= width
            and not sum(original_counts)):
        return target, dict(fit, closing_axis_xy=original.tolist(), closing_width_m=width,
            jaw_axis_choice='observed_original_axis_preserves_open_finger_clearance',
            open_finger_axis_evidence=dict(lateral_obstacle_points=lateral_counts,
                original_obstacle_points=original_counts, minimum_width_reduction_m=.006,
                maximum_width_for_finger_clearance_m=clearance_width,
                per_finger_clearance_m=.006,
                source='same RGBD observation and public Fetch finger dimensions',
                boundary='unseen space remains unproven; existing motion guards remain active'))
    if sum(lateral_counts) < 24 or sum(original_counts):
        return target, fit
    return target, dict(fit, closing_axis_xy=original.tolist(), closing_width_m=width,
        jaw_axis_choice='observed_original_axis_avoids_open_finger_obstacles',
        open_finger_axis_evidence=dict(lateral_obstacle_points=lateral_counts,
            original_obstacle_points=original_counts, minimum_width_reduction_m=.006,
            source='same RGBD observation and public Fetch finger dimensions',
            boundary='unseen space remains unproven; existing motion guards remain active'))


def wrap_tabletop_grasp(original, geometry):
    def wrapped(observation, detection, *args, **kwargs):
        target, fit = original(observation, detection, *args, **kwargs)
        if (fit.get('jaw_axis_choice') != 'observed_feasible_robot_lateral_axis'
                or not .04 < fit['height_m'] <= .12
                or fit.get('grasp_section_points') is not None
                or fit.get('grasp_column_extended', False)):
            return target, fit
        shape = geometry.observed_shape(observation, detection, refine_foreground=True)
        return prefer_less_obstructed_axis(observation, target, fit, shape)
    return wrapped
