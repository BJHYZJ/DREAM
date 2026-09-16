"""Lower-score rechecks require a recent, nearby, confident visual anchor."""
import numpy as np


def can_recheck_track(anchor, prior, base_xy, elapsed_seconds, standard_threshold=.15):
    if anchor is None:return False
    prior=np.asarray(prior,dtype=float)
    point=np.asarray(anchor.point_world,dtype=float)
    base=np.asarray(base_xy,dtype=float)[:2]
    if prior.shape!=(3,) or point.shape!=(3,) or base.shape!=(2,):return False
    if not (np.isfinite(prior).all() and np.isfinite(point).all()
            and np.isfinite(base).all() and np.isfinite(elapsed_seconds)):return False
    return bool(anchor.score>=standard_threshold and 0<=elapsed_seconds<=10.
                and np.linalg.norm(prior[:2]-base)<=1.2
                and np.linalg.norm(prior-point)<=.25)


def continuous_crop_candidates(candidates, prior, maximum_displacement=.15):
    prior=np.asarray(prior,dtype=float)
    return sorted((d for d in candidates if np.linalg.norm(np.asarray(d.point_world)-prior)<=maximum_displacement),
                  key=lambda d:(np.linalg.norm(np.asarray(d.point_world)-prior),-d.score))
