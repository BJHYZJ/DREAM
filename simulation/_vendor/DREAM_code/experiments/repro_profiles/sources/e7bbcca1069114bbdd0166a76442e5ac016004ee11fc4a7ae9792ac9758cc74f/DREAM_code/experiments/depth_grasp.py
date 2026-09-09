"""Observation-only geometric grasp centre for small approximately round objects.

This is an explicit embodiment/task-family adapter, not AnyGrasp or DREAM's
physical grasp predictor. The fitted radius comes from RGB-D, not actor sizes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def fit_round_grasp_center(points):
    points=np.asarray(points,dtype=float)
    points=points[np.isfinite(points).all(axis=1)]
    if len(points)<20:
        raise ValueError("Insufficient observed surface for geometric grasp")
    origin=np.median(points,axis=0)
    local=points-origin
    matrix=np.c_[2*local,np.ones(len(local))]
    value=np.einsum("ij,ij->i",local,local)
    solution,*_=np.linalg.lstsq(matrix,value,rcond=None)
    radius=np.sqrt(max(1e-8,solution[3]+solution[:3]@solution[:3]))
    radius=float(np.clip(radius,.015,.055))
    initial=np.r_[np.clip(solution[:3],-.08,.08),radius]
    result=least_squares(lambda x:np.linalg.norm(local-x[:3],axis=1)-x[3],initial,
        bounds=([-0.1,-0.1,-0.1,.01],[.1,.1,.1,.06]),loss="soft_l1",f_scale=.004)
    residual=float(np.median(np.abs(result.fun)))
    if not result.success or residual>.008:
        raise ValueError(f"Observed surface is not reliably round: residual={residual:.4f}")
    return result.x[:3]+origin,dict(radius_m=float(result.x[3]),median_surface_residual_m=residual,
                                   observed_points=len(points))


def grasp_center_from_detection(observation,detection):
    h,w=observation.depth_m.shape
    box=np.asarray(detection.box_xyxy)
    x0,y0=np.maximum(np.floor(box[:2]).astype(int),0)
    x1,y1=np.minimum(np.ceil(box[2:]).astype(int),(w,h))
    depth=observation.depth_m[y0:y1,x0:x1]
    valid=np.isfinite(depth)&(depth>.10)&(depth<4.)
    if valid.sum()<20:
        raise ValueError("No usable target depth")
    median=float(np.median(depth[valid]))
    # Reject background surfaces in the box by current depth only. There is
    # no segmentation ID, evaluator pose, known color or known object radius.
    valid&=np.abs(depth-median)<.06
    points=observation.world_points()[y0:y1,x0:x1][valid]
    return fit_round_grasp_center(points)
