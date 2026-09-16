"""Reobserve a retained elongated payload after the tool has folded.

Inputs are the original RGB-D appearance/dimensions, current RGB-D, and TCP
proprioception. A partial neck image alone cannot change the navigation bound.
The measured cylinder must explain a nearly complete visible body surface.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


def fit_carry_cylinder(points, tcp_matrix, radius, height):
    points=np.asarray(points,dtype=float);tool=np.asarray(tcp_matrix,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)<96
            or not np.isfinite(points).all() or tool.shape!=(4,4)
            or not np.isfinite(tool).all() or not .012<radius<.06
            or not .20<height<.50):
        raise ValueError("Insufficient elongated held-surface geometry")
    origin=points.mean(0)
    _,_,basis=np.linalg.svd(points-origin,full_matrices=False)
    axis=basis[0]
    # Choose the body end away from the observed gripper, independent of world
    # orientation. The grasped end may contain a narrow neck and shoulder.
    if np.dot(tool[:3,3]-origin,axis)<0:axis=-axis
    transverse=basis[1:]
    projection=(points-origin)@axis
    lo,hi=np.quantile(projection,[.005,.995])
    if hi-lo<.75*height:
        raise ValueError("Held surface is clipped or lacks full axial extent")
    body=points[(projection>lo+.06*(hi-lo))&(projection<lo+.55*(hi-lo))]
    if len(body)<96:raise ValueError("Insufficient visible held body")
    anchor=body.mean(0)
    initial_axis=axis.copy()

    def decode(values):
        direction=initial_axis+values[2:]@transverse
        direction/=np.linalg.norm(direction)
        return anchor+values[:2]@transverse,direction

    def residual(values):
        center,direction=decode(values);delta=body-center
        radial=delta-np.outer(delta@direction,direction)
        return np.linalg.norm(radial,axis=1)-radius

    fitted=least_squares(residual,np.zeros(4),loss="soft_l1",f_scale=.002,
        bounds=([-2*radius,-2*radius,-.7,-.7],[2*radius,2*radius,.7,.7]),max_nfev=100)
    center,axis=decode(fitted.x)
    delta=points-center;projection=delta@axis
    lo,hi=np.quantile(projection,[.001,.999]);span=float(hi-lo)
    radial=np.linalg.norm(delta-np.outer(projection,axis),axis=1)
    errors=np.abs(residual(fitted.x));p95=float(np.quantile(errors,.95))
    # Reject planar fragments, an incorrect prior radius, and object leakage.
    if (not fitted.success or p95>.003 or np.quantile(radial,.99)>radius+.008
            or not .95*height<span<1.10*height):
        raise ValueError("Held body does not support the observed cylinder bound")
    normal=np.cross(axis,transverse[0]);normal/=np.linalg.norm(normal)
    tangent=np.cross(normal,axis)
    # Validate curvature on all visible sections that match the body radius,
    # including the upper body omitted from the neck-resistant fit.
    offsets=delta[np.abs(radial-radius)<.003]
    angles=np.sort(np.arctan2(offsets@normal,offsets@tangent))
    coverage=float(2*np.pi-np.diff(np.r_[angles,angles[0]+2*np.pi]).max())
    if coverage<1.5:raise ValueError("Held body curvature is insufficiently observed")
    center=center+.5*(lo+hi)*axis
    grasp=tool[:3,3]-center
    axial=float(np.dot(grasp,axis))
    if (np.linalg.norm(grasp-axial*axis)>radius+.025
            or abs(axial-.5*span)>.075):
        raise ValueError("Observed cylinder is not attached to the retained grasp")
    return dict(frame=dict(center_tcp_m=((center-tool[:3,3])@tool[:3,:3]).tolist(),
                           axis_tcp=(axis@tool[:3,:3]).tolist()),
        radius_m=float(radius),height_m=max(float(height),span),
        observed_center_world=center.tolist(),observed_axis_world=axis.tolist(),
        observed_height_m=span,points=len(points),body_points=len(body),
        residual_p95_m=p95,angular_coverage_rad=coverage)


def observed_carry_shape(observation, template, tcp_matrix, radius, height):
    """Match the held object's recorded appearance within its measured reach."""
    from instruction_geometry import connected_depth_surface
    colors=np.asarray(template["colors_lab"],dtype=float)
    # Suppress a small contrasting label/background fragment without assuming
    # a particular object's color. Geometry independently validates the patch.
    dominant=colors[np.linalg.norm(colors-np.median(colors,axis=0),axis=1)<10]
    if len(dominant)<48:raise ValueError("No stable observed payload appearance")
    world=observation.world_points();tcp=np.asarray(tcp_matrix)[:3,3]
    distance=np.linalg.norm(world-tcp,axis=2)
    valid=np.isfinite(world).all(2)&np.isfinite(observation.depth_m)
    valid&=(observation.depth_m>.10)&(distance<height+radius)
    if valid.sum()<96:raise ValueError("No nearby held surface")
    lab=cv2.cvtColor(observation.rgb.astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    matched=np.zeros(valid.shape,bool)
    matched[valid]=cKDTree(dominant).query(lab[valid])[0]<12
    matched=connected_depth_surface(world,matched)
    # A clipped object cannot establish a new full-body bound.
    if (matched[:2].any() or matched[-2:].any()
            or matched[:,:2].any() or matched[:,-2:].any()):
        raise ValueError("Held surface touches the image boundary")
    result=fit_carry_cylinder(world[matched],tcp_matrix,radius,height)
    result.update(observation_id=int(observation.frame_id),
        source_grasp_observation_id=int(template["source_observation_id"]),
        source="Fresh RGB-D body curvature, grasp appearance, and measured TCP")
    result["frame"]["observation_id"]=int(observation.frame_id)
    return result


def navigation_payload(policy):
    """Keep the carry observation separate from the pre-fold release frame."""
    shape=getattr(policy,"observed_carry_shape",None)
    if shape is not None:
        return shape["radius_m"],shape["height_m"],shape["frame"]
    return (policy.perceived_payload_radius,policy.perceived_payload_height,
            getattr(policy,"perceived_payload_frame",None))
