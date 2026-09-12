"""RGB-D-only tabletop grasp and receptacle proposals.

No scene assets, actor poses, instance masks or task coordinates are inputs.
This is an explicit geometric Fetch adapter, not AnyGrasp or a general grasp
network. Horizontal, visible, isolated table-top objects are its current scope.
"""
from dataclasses import dataclass,replace

import numpy as np
import cv2
from scipy.ndimage import label
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree


def focused_rgbd_crop(observation,remembered_point,side=320):
    """Crop a recorded view around a projected visual prior, preserving calibration."""
    point=np.asarray(remembered_point,dtype=float)
    if point.shape!=(3,) or not np.isfinite(point).all():return None
    transform=observation.camera_to_world_cv
    local=(point-transform[:3,3])@transform[:3,:3]
    if not .10<=local[2]<=4.:return None
    pixel=observation.intrinsics@local;pixel=pixel[:2]/pixel[2]
    h,w=observation.depth_m.shape
    if not (0<=pixel[0]<w and 0<=pixel[1]<h):return None
    side=min(int(side),h,w)
    if side<1:raise ValueError("Crop size must be positive")
    x=int(np.clip(round(pixel[0]-side/2),0,w-side))
    y=int(np.clip(round(pixel[1]-side/2),0,h-side))
    intrinsics=observation.intrinsics.copy();intrinsics[0,2]-=x;intrinsics[1,2]-=y
    crop=replace(observation,rgb=observation.rgb[y:y+side,x:x+side].copy(),
        depth_m=observation.depth_m[y:y+side,x:x+side].copy(),intrinsics=intrinsics)
    return crop,(x,y,x+side,y+side)


def observed_payload_template(observation,detection,center_world,grasp_tcp):
    """Store measured object surfaces in the measured pre-closure tool frame."""
    shape=observed_shape(observation,detection,refine_foreground=True)
    points=shape.points
    if len(points)>2048:
        points=points[np.linspace(0,len(points)-1,2048,dtype=int)]
    camera=np.linalg.inv(observation.camera_to_world_cv)
    local=points@camera[:3,:3].T+camera[:3,3]
    pixels=local@observation.intrinsics.T
    uv=np.rint(pixels[:,:2]/pixels[:,2:3]).astype(int)
    lab=cv2.cvtColor(observation.rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    tool=np.asarray(grasp_tcp,dtype=float)
    return dict(points_tcp=(points-tool[:3,3])@tool[:3,:3],
                colors_lab=lab[uv[:,1],uv[:,0]],
                center_tcp=(np.asarray(center_world)-tool[:3,3])@tool[:3,:3],
                source_observation_id=np.asarray(observation.frame_id))


def register_payload_points(template,observed_points,observed_lab,tcp):
    """Bounded translation fit; observed partial surfaces query the grasp template."""
    tool=np.asarray(tcp,dtype=float)
    predicted=np.asarray(template['points_tcp'])@tool[:3,:3].T+tool[:3,3]
    colors=np.asarray(template['colors_lab'])
    live=np.asarray(observed_points,dtype=float)
    lab=np.asarray(observed_lab,dtype=float)
    valid=np.isfinite(live).all(1)&np.isfinite(lab).all(1)
    valid &= np.linalg.norm(live-tool[:3,3],axis=1)<.15
    live=live[valid];lab=lab[valid]
    if len(live)<48:raise ValueError('Insufficient visible payload surface')
    same_color=cKDTree(colors).query(lab)[0]<20
    live=live[same_color];lab=lab[same_color]
    if len(live)<48:raise ValueError('Insufficient observed appearance agreement')
    shift=np.zeros(3)
    for _ in range(12):
        tree=cKDTree(np.c_[(predicted+shift)*100,colors*.03])
        _,ids=tree.query(np.c_[live*100,lab*.03])
        residual=live-(predicted[ids]+shift)
        distance=np.linalg.norm(residual,axis=1)
        keep=(distance<.035)&(np.linalg.norm(lab-colors[ids],axis=1)<25)
        if keep.sum()<48:raise ValueError('Too few geometric correspondences')
        keep &= distance<=np.quantile(distance[keep],.8)
        update=np.median(residual[keep],axis=0)
        shift+=update
        if np.linalg.norm(shift)>.035:raise ValueError('Observed grasp template displaced too far')
        if np.linalg.norm(update)<.0002:break
    distance,ids=cKDTree(predicted+shift).query(live)
    keep=(distance<.012)&(np.linalg.norm(lab-colors[ids],axis=1)<25)
    if keep.sum()<48 or keep.mean()<.60:
        raise ValueError('Payload registration lacks geometric support')
    if np.linalg.norm(np.ptp(live[keep],axis=0))<.020:
        raise ValueError('Visible payload patch is too small')
    center=np.asarray(template['center_tcp'])@tool[:3,:3].T+tool[:3,3]+shift
    offset=(center-tool[:3,3])@tool[:3,:3]
    if np.linalg.norm(offset)>.12:raise ValueError('Payload center is outside the held envelope')
    return offset,dict(inliers=int(keep.sum()),observed_color_points=len(live),
                      rms_m=float(np.sqrt(np.mean(distance[keep]**2))),
                      translation_fit_m=shift.tolist(),observed_center_world=center.tolist())


def observed_payload_offset(observation,detections,template,tcp):
    """Require a fresh text-grounded box and matching measured object surfaces."""
    world=observation.world_points()
    lab=cv2.cvtColor(observation.rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    h,w=observation.depth_m.shape
    for detection in detections:
        if detection.observation_id!=observation.frame_id:continue
        x0,y0,x1,y1=np.rint(detection.box_xyxy).astype(int)
        if x0<2 or y0<2 or x1>w-2 or y1>h-2:continue
        depth=observation.depth_m[y0:y1,x0:x1].reshape(-1)
        points=world[y0:y1,x0:x1].reshape(-1,3)
        colors=lab[y0:y1,x0:x1].reshape(-1,3)
        valid=np.isfinite(depth)&(depth>=.10)&(depth<=4.)
        try:
            offset,fit=register_payload_points(template,points[valid],colors[valid],tcp)
        except ValueError:
            continue
        return offset,dict(fit,observation_id=observation.frame_id,box_xyxy=detection.box_xyxy,
                          source_grasp_observation_id=int(template['source_observation_id']))
    raise ValueError('No fresh held-object observation supports an offset correction')


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


def observed_shape(observation, detection, minimum_points=24, refine_foreground=False):
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
    if refine_foreground:
        selected=separate_object_surface(observation.rgb[y0:y1,x0:x1],selected,
                                        inner[...,2],support,minimum_points)
    points=inner[selected]
    if len(points)<minimum_points:
        raise ValueError("Insufficient connected object surface")
    return ObservedShape(observation.frame_id,points,support,len(supported))


def separate_object_surface(rgb,selected,heights,support,minimum_points=24):
    """Separate a tabletop object's appearance from a connected support rim.

    Seeds come from the observed upper object surface. The depth component
    bounds the result; no pixels outside that observed component are added.
    Low-contrast or insufficient foreground evidence retains the original fit.
    """
    top=float(np.quantile(heights[selected],.98))
    core=selected&(heights>=support+.5*(top-support))
    if core.sum()<minimum_points or (~selected).sum()<minimum_points:
        return selected
    mask=np.full(selected.shape,cv2.GC_BGD,np.uint8)
    mask[selected]=cv2.GC_PR_FGD
    mask[core]=cv2.GC_FGD
    background=np.zeros((1,65),np.float64)
    foreground=np.zeros((1,65),np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(np.ascontiguousarray(rgb,dtype=np.uint8),mask,None,
                background,foreground,5,cv2.GC_INIT_WITH_MASK)
    refined=selected&((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD))
    if refined.sum()<max(minimum_points,.60*selected.sum()):
        return selected
    return refined


def tabletop_grasp(observation,detection,maximum_jaw_width_m=.10):
    shape=observed_shape(observation,detection,refine_foreground=True)
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


def visible_inner_wall(observation,detection,center,radius,rim_height,support_height):
    """Find inward, upward-facing cavity surfaces inside the observed rim.

    A front rim can hide the central floor while the opposite inner wall is
    visible. Depth normals distinguish that wall from a solid object's outer
    slope. Discontinuous depth edges cannot contribute surface evidence.
    """
    xyz=observation.world_points()
    points=xyz[1:-1,1:-1]
    dx=xyz[1:-1,2:]-xyz[1:-1,:-2]
    dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
    normals=np.cross(dx,dy)
    norm=np.linalg.norm(normals,axis=2)
    normals/=np.maximum(norm[...,None],1e-12)
    facing=np.sum(normals*(observation.camera_to_world_cv[:3,3]-points),axis=2)
    normals*=np.where(facing<0,-1.,1.)[...,None]
    radial=np.asarray(center)-points[...,:2]
    distance=np.linalg.norm(radial,axis=2)
    inward=np.sum(normals[...,:2]*radial,axis=2)/np.maximum(distance,1e-12)
    x0,y0,x1,y1=np.rint(detection.box_xyxy).astype(int)
    valid=np.zeros(norm.shape,bool)
    valid[max(0,y0-1):max(0,y1-1),max(0,x0-1):max(0,x1-1)]=True
    valid &= (distance<radius*.85)&(points[...,2]>support_height+.003)
    valid &= (points[...,2]<rim_height-.02)&(norm>1e-10)
    valid &= (normals[...,2]>.15)&(inward>.25)
    for neighbour in (xyz[:-2,1:-1],xyz[2:,1:-1],xyz[1:-1,:-2],xyz[1:-1,2:]):
        valid &= np.linalg.norm(neighbour-points,axis=2)<.02
    return points[valid]


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
    cavity_evidence="visible_central_surface"
    if relation=="in":
        if len(central)<12:
            central=visible_inner_wall(observation,detection,center,radius,top,shape.support_height_m)
            cavity_evidence="visible_inward_sloping_wall"
            if len(central)<12:
                raise ValueError("Receptacle interior is not visible; take a better observation")
        surface_height=float(np.quantile(central[:,2],.20))
        if top-surface_height<.02:
            raise ValueError("No observable cavity supports an 'in' placement")
    else:
        if len(central)<12:
            raise ValueError("Placement surface is not sufficiently observed")
        surface_height=float(np.median(central[:,2]))
    # An 'in' action must not drive closed fingers and the retained payload
    # into the observed bowl floor. Release above the observed rim, then let
    # native gravity/contact settle the object. Cavity evidence comes from
    # the observed surfaces, without importing task-asset geometry.
    # A visible inner wall also certifies a cavity without assuming the height
    # of an occluded floor. Release remains referenced to the observed rim.
    # Use the larger observed half-envelope to allow for an off-centre grasp.
    # 'On' placement keeps the existing surface-relative placement proposal.
    reference=top if relation=="in" else surface_height
    clearance=max(payload_radius_m,payload_height_m*.5)+.020 if relation=="in" else payload_height_m*.5+.015
    release=np.r_[center,reference+clearance]
    return release,dict(observation_id=shape.observation_id,relation=relation,
        center_xy=center.tolist(),observed_safe_radius_m=radius,
        observed_surface_height_m=surface_height,observed_rim_height_m=top,
        support_height_m=shape.support_height_m,observed_points=len(points),
        payload_radius_from_depth_m=float(payload_radius_m),
        payload_height_from_depth_m=float(payload_height_m),
        cavity_evidence=cavity_evidence if relation=="in" else None,
        cavity_surface_points=len(central) if relation=="in" else None,
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
