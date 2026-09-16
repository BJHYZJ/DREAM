"""Fit a circular jaw section only when measured surface curvature supports it."""
import numpy as np


def observed_circular_section(xy):
    xy=np.asarray(xy,dtype=float)
    if len(xy)<48 or xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():return None
    origin=xy.mean(0);local=xy-origin
    coefficients,_,rank,_=np.linalg.lstsq(np.c_[2*local,np.ones(len(local))],
                                         (local*local).sum(1),rcond=None)
    radius_squared=coefficients[2]+np.dot(coefficients[:2],coefficients[:2])
    if rank<3 or radius_squared<=0:return None
    center=origin+coefficients[:2];radius=float(np.sqrt(radius_squared))
    if not .012<radius<.06:return None
    delta=xy-center
    residual=np.abs(np.linalg.norm(delta,axis=1)-radius)
    p95=float(np.quantile(residual,.95))
    if p95>.002:return None
    angles=np.sort(np.arctan2(delta[:,1],delta[:,0]))
    coverage=float(2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max())
    if coverage<1.5:return None
    return dict(center_xy=center.tolist(),radius_m=radius,residual_p95_m=p95,
                angular_coverage_rad=coverage,points=len(xy))
