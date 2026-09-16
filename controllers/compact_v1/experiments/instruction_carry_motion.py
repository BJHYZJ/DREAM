"""Limit acceleration without extending a planner's checked base-motion arc."""
import numpy as np


def bounded_carry_twist(previous,requested,dt):
    previous=np.asarray(previous,dtype=float)
    requested=np.asarray(requested,dtype=float)
    if previous.shape!=(2,) or requested.shape!=(2,) or not np.isfinite([*previous,*requested,dt]).all() or dt<=0:
        raise ValueError('Expected finite forward/yaw commands and positive control interval')
    if np.any(previous*requested<0):
        return 0.,0.
    scale=1.
    for index,acceleration in enumerate((.10,.30)):
        magnitude=abs(requested[index])
        if magnitude:
            scale=min(scale,(abs(previous[index])+acceleration*dt)/magnitude)
    if requested[1]:scale=min(scale,.20/abs(requested[1]))
    # A shared scalar preserves the requested curvature. Executed motion is
    # a prefix of the already checked constant-twist arc, including reversal.
    # Stops and deceleration are immediate so clearance stops are never delayed.
    return tuple(float(x) for x in requested*scale)
