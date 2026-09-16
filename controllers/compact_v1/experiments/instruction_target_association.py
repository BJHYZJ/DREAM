"""Associate pickup detections using observation age and spatial continuity."""
import numpy as np


def pickup_track_matches(candidate, previous, *, previous_rejected,
                         elapsed_seconds=0.,association_radius=.45):
    if candidate is None or previous is None or previous_rejected:
        return True
    if not np.isfinite(elapsed_seconds) or elapsed_seconds<0:
        raise ValueError('Observation age must be finite and nonnegative')
    # A past view can no longer enforce stationary identity indefinitely.
    # The speed allowance is a controller prior, not a measured actor speed.
    if elapsed_seconds>=25.:return True
    separation = np.linalg.norm(np.asarray(candidate.point_world)
                                - np.asarray(previous.point_world))
    return bool(separation <= association_radius+.12*elapsed_seconds)


def stronger_enclosing_pickup_box(part, whole):
    """Identify a confident whole-object box around a nearby partial detection.

    This only compares ordinary detections in one calibrated image. Separate
    objects keep their spatial ordering; no new box or 3D point is fabricated.
    """
    if (getattr(part, "observation_id", None) is None
            or getattr(part, "query", None) is None
            or part.observation_id != getattr(whole, "observation_id", None)
            or part.query != getattr(whole, "query", None)):
        return False
    a = np.asarray(getattr(part, "box_xyxy", []), dtype=float)
    b = np.asarray(getattr(whole, "box_xyxy", []), dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(np.r_[a,b]).all():
        return False
    side_a, side_b = a[2:] - a[:2], b[2:] - b[:2]
    if np.any(side_a <= 0) or np.any(side_b <= 0):
        return False
    area_a, area_b = float(np.prod(side_a)), float(np.prod(side_b))
    coverage = float(np.prod(np.maximum(0., np.minimum(a[2:],b[2:])
                                               - np.maximum(a[:2],b[:2])))) / area_a
    separation = np.linalg.norm(np.asarray(part.point_world)-np.asarray(whole.point_world))
    return bool(1.5 <= area_b / area_a <= 6. and coverage >= .90
                and np.isfinite(separation) and separation <= .08
                and np.isfinite([part.score,whole.score]).all()
                and whole.score >= max(part.score + .10, 1.5 * part.score))


def order_pickup_candidates(candidates,previous):
    """Keep local continuity, promoting a strong enclosing box for the same view."""
    if previous is None:return list(candidates)
    ordered = sorted(candidates,key=lambda d:(
        np.linalg.norm(np.asarray(d.point_world)-np.asarray(previous.point_world)),-d.score))
    result, seen = [], set()
    for part in ordered:
        enclosing = [d for d in ordered if stronger_enclosing_pickup_box(part,d)]
        whole = max(enclosing,key=lambda d:d.score) if enclosing else part
        for detection in (whole,part):
            if id(detection) not in seen:
                result.append(detection)
                seen.add(id(detection))
    return result
