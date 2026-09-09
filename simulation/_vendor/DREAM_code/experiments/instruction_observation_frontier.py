"""Safe observation standoffs derived exclusively from the observed map."""
import numpy as np
from scipy.ndimage import binary_dilation,distance_transform_edt


def observation_standoffs(known,safe,distances,visits,resolution,radius):
    # Raw frontiers can lie in the inflated band of a door frame or a movable
    # obstruction. Never drive into that band: observe it from a reachable cell.
    boundary=(known==1)&binary_dilation(known==0,iterations=2)
    if not boundary.any():return np.zeros_like(safe,dtype=bool)
    boundary_distance=distance_transform_edt(~boundary)*resolution
    return (safe&np.isfinite(distances)&(distances>=round(.32/resolution))&
            (visits<3)&(boundary_distance<=radius+.35))
