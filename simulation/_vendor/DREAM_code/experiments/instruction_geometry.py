"""RGB-D-only tabletop grasp and receptacle proposals.

No scene assets, actor poses, instance masks or task coordinates are inputs.
This is an explicit geometric Fetch adapter, not AnyGrasp or a general grasp
network. Horizontal, visible, isolated table-top objects are its current scope.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull


@dataclass(frozen=True)
class ObservedShape:
    observation_id: int
    points: np.ndarray
    support_height_m: float
    support_points: int


def base_lateral_carry_rotation(current_rotation,base_yaw):
    """Calibration candidate: rotate a held grasp around gravity, not tilt it.

    Uses only measured gripper orientation and robot yaw. The horizontal jaw
    direction is aligned with the robot lateral axis to test a compact IK
    posture. This is not used by the default policy until physically validated.
    """
    rotation=np.asarray(current_rotation,dtype=float)
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all():
        raise ValueError("Invalid measured gripper rotation")
    jaw=rotation[:2,1]
    if np.linalg.norm(jaw)<1e-6:
        raise ValueError("No measured horizontal jaw direction for carry yaw")
    angle=base_yaw+np.pi/2-np.arctan2(jaw[1],jaw[0])
    angle=float(np.arctan2(np.sin(angle),np.cos(angle)))
    c,s=np.cos(angle),np.sin(angle)
    around_gravity=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    return around_gravity@rotation,angle


def point_observed_empty(observation,point_world,tolerance_m=.10):
    """Missing detection is removal evidence only when a depth ray sees beyond it.

    Occlusion, looking away, invalid depth and detector uncertainty are not
    evidence that an object moved. Pose is measured camera calibration.
    """
    world_to_camera=np.linalg.inv(observation.camera_to_world_cv)
    point=world_to_camera[:3,:3]@np.asarray(point_world)+world_to_camera[:3,3]
    if not np.isfinite(point).all() or point[2]<=.10:
        return False
    pixel=observation.intrinsics@point
    u,v=np.rint(pixel[:2]/pixel[2]).astype(int)
    h,w=observation.depth_m.shape
    if not 2<=u<w-2 or not 2<=v<h-2:
        return False
    depth=observation.depth_m[v-2:v+3,u-2:u+3]
    valid=np.isfinite(depth)&(depth>.10)&(depth<4.)
    return bool(valid.sum()>=15 and np.mean(depth[valid]>point[2]+tolerance_m)>=.8)


def observed_shape(observation, detection, minimum_points=24):
    if detection.observation_id != observation.frame_id:
        raise ValueError("Detection and RGB-D observation IDs must match")
    h,w=observation.depth_m.shape
    box=np.asarray(detection.box_xyxy,dtype=float)
    if box.shape!=(4,) or not np.isfinite(box).all():
        raise ValueError("Invalid observed detection box")
    x0,y0=np.maximum(np.floor(box[:2]).astype(int),0)
    x1,y1=np.minimum(np.ceil(box[2:]).astype(int),(w,h))
    if x1<=x0 or y1<=y0:
        raise ValueError("Empty observed detection box")
    xyz=observation.world_points()
    valid=np.isfinite(xyz).all(-1)&np.isfinite(observation.depth_m)
    valid&=(observation.depth_m>.10)&(observation.depth_m<4.)
    inner=xyz[y0:y1,x0:x1]
    usable=valid[y0:y1,x0:x1]
    if usable.sum()<minimum_points:
        raise ValueError("Insufficient object RGB-D")
    # A local image annulus estimates the support surface from actual depth.
    # Restrict height relative to observed contents to exclude distant floors.
    border=max(8,int(max(x1-x0,y1-y0)*.35))
    ring=np.zeros((h,w),bool)
    ring[max(0,y0-border):min(h,y1+border),max(0,x0-border):min(w,x1+border)]=True
    ring[y0:y1,x0:x1]=False
    heights=xyz[ring&valid,2]
    reference=float(np.median(inner[usable,2]))
    heights=heights[(heights>reference-.30)&(heights<reference+.02)]
    if len(heights)<minimum_points:
        raise ValueError("No observed support around the detection")
    bins=np.round(heights/.008).astype(int)
    values,counts=np.unique(bins,return_counts=True)
    peak=values[np.argmax(counts)]*.008
    supported=heights[np.abs(heights-peak)<.012]
    if len(supported)<minimum_points:
        raise ValueError("Support plane is not sufficiently observed")
    support=float(np.median(supported))
    foreground=usable&(inner[...,2]>support+.003)&(inner[...,2]<support+.30)
    components,n=label(foreground)
    if n==0:
        raise ValueError("No object separated above the observed support")
    counts=np.bincount(components.ravel()); counts[0]=0
    selected=components==int(np.argmax(counts))
    points=inner[selected]
    if len(points)<minimum_points:
        raise ValueError("Insufficient connected object surface")
    return ObservedShape(observation.frame_id,points,support,len(supported))


def tabletop_grasp(observation,detection,maximum_jaw_width_m=.10):
    shape=observed_shape(observation,detection)
    points=shape.points
    center=np.median(points[:,:2],axis=0)
    covariance=np.cov((points[:,:2]-center).T)
    _,axes=np.linalg.eigh(covariance)
    local=(points[:,:2]-center)@axes
    lo,hi=np.quantile(local,[.02,.98],axis=0)
    center=center+axes@((lo+hi)/2)
    widths=hi-lo
    short=int(np.argmin(widths))
    width=float(widths[short])
    if not .012<width<maximum_jaw_width_m-.006:
        raise ValueError(f"Observed grasp width {width:.3f} m is outside the calibrated gripper range")
    top=float(np.quantile(points[:,2],.98))
    height=top-shape.support_height_m
    if not .018<height<.20:
        raise ValueError(f"Observed object height {height:.3f} m is outside the tabletop adapter range")
    target=np.r_[center,shape.support_height_m+height*.50]
    radius=float(np.max(np.linalg.norm(points[:,:2]-center,axis=1)))
    fit=dict(radius_m=radius,height_m=height,closing_width_m=width,
             closing_axis_xy=axes[:,short].tolist(),support_height_m=shape.support_height_m,
             observed_points=len(points),support_points=shape.support_points,
             observation_id=shape.observation_id,proposal="observed_tabletop_short_axis")
    return target,fit


def align_symmetric_grasp_axis(target,fit,base_xy):
    """Resolve the +/- ambiguity of a PCA parallel-jaw axis using robot pose.

    The jaw closing LINE, observed width, target position and object dimensions
    do not change. Selecting the equivalent sign nearest the robot's lateral
    axis avoids an arbitrary 180-degree wrist choice. This is robot kinematics,
    not an actor-pose/mesh prior and not a relaxed collision/grasp threshold.
    """
    if "closing_axis_xy" not in fit:return fit
    axis=np.asarray(fit["closing_axis_xy"],dtype=float)
    forward=np.asarray(target,dtype=float)[:2]-np.asarray(base_xy,dtype=float)[:2]
    if axis.shape!=(2,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError("Invalid observed parallel-jaw axis")
    if not np.isfinite(forward).all() or np.linalg.norm(forward)<1e-8:
        raise ValueError("A nonzero measured approach direction is required")
    left=np.array([-forward[1],forward[0]])
    dot=float(axis@left)
    flip=dot<0 or (abs(dot)<1e-10 and float(axis@forward)<0)
    return dict(fit,closing_axis_xy=(-axis if flip else axis).tolist(),
        pca_closing_axis_xy=axis.tolist(),equivalent_axis_sign_flipped=bool(flip),
        axis_sign_reference="measured robot lateral axis at visual approach")


def receptacle_region(observation,detection,relation,payload_radius_m,payload_height_m):
    if relation not in ("in","on"):
        raise ValueError("Only observed in/on placement is supported")
    if not (np.isfinite(payload_radius_m) and np.isfinite(payload_height_m)
            and payload_radius_m>0 and payload_height_m>0):
        raise ValueError("Payload dimensions must come from a valid observed grasp")
    shape=observed_shape(observation,detection)
    points=shape.points
    lo,hi=np.quantile(points[:,:2],[.01,.99],axis=0)
    center=(lo+hi)/2
    hull=ConvexHull(points[:,:2])
    equations=hull.equations
    # Distance from proposed center to the actual observed footprint boundary.
    distances=-(equations[:,:2]@center+equations[:,2])/np.linalg.norm(equations[:,:2],axis=1)
    radius=float(np.min(distances))*(.70 if relation=="in" else .90)
    if radius<=payload_radius_m+.004:
        raise ValueError("Observed receptacle does not provide payload clearance")
    top=float(np.quantile(points[:,2],.98))
    central=points[np.linalg.norm(points[:,:2]-center,axis=1)<max(.018,radius*.35)]
    if relation=="in":
        if len(central)<12:
            raise ValueError("Receptacle interior is not visible; take a better observation")
        floor=float(np.quantile(central[:,2],.20))
        if top-floor<.02:
            raise ValueError("No observable cavity supports an 'in' placement")
    else:
        if len(central)<12:
            raise ValueError("Placement surface is not sufficiently observed")
        floor=float(np.median(central[:,2]))
    # An 'in' action must not drive closed fingers and the retained payload
    # into the observed bowl floor. Release above the observed rim, then let
    # native gravity/contact settle the object. The floor is still required
    # to verify a visible cavity; it is not replaced with task-asset geometry.
    # Use the larger observed half-envelope to allow for an off-centre grasp.
    # 'On' placement keeps the existing surface-relative placement proposal.
    reference=top if relation=="in" else floor
    clearance=max(payload_radius_m,payload_height_m*.5)+.020 if relation=="in" else payload_height_m*.5+.015
    release=np.r_[center,reference+clearance]
    return release,dict(observation_id=shape.observation_id,relation=relation,
        center_xy=center.tolist(),observed_safe_radius_m=radius,
        observed_surface_height_m=floor,observed_rim_height_m=top,
        support_height_m=shape.support_height_m,observed_points=len(points),
        payload_radius_from_depth_m=float(payload_radius_m),
        payload_height_from_depth_m=float(payload_height_m),
        release_height_reference="observed_rim" if relation=="in" else "observed_surface",
        release_clearance_m=float(clearance),
        proposal="observed_receptacle_center_above_rim" if relation=="in" else "observed_receptacle_center_and_surface")


def support_relation_matches(observation,receptacle_detection,support_detection):
    """Check a language-detected support lies beneath the detected receptacle.

    Language identity comes from the detector queries; this function checks the
    spatial relation only. It cannot turn a failed text grounding into success.
    """
    if support_detection is None or support_detection.observation_id!=observation.frame_id:
        return False
    if receptacle_detection.observation_id!=observation.frame_id:
        return False
    bx=np.asarray(receptacle_detection.box_xyxy)
    sx=np.asarray(support_detection.box_xyxy)
    center=(bx[:2]+bx[2:])/2
    overlap=(min(bx[2],sx[2])-max(bx[0],sx[0]))/max(1.,bx[2]-bx[0])
    if overlap<.6 or not sx[0]<=center[0]<=sx[2] or sx[3]<center[1]:
        return False
    try:
        shape=observed_shape(observation,receptacle_detection)
    except ValueError:
        return False
    support_z=float(support_detection.point_world[2])
    # A support detection's median includes its apron/legs, so it may be below
    # the table top. It must not instead be an object floating above the rim.
    return shape.support_height_m-.65<=support_z<=shape.support_height_m+.04
