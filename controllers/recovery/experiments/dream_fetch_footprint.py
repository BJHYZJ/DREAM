"""Robot-only measured horizontal collision footprint for navigation diagnostics.

Task actor poses/meshes are never accepted. A held-object proxy, when supplied,
is the policy's already recorded RGB-D radius centered at the measured TCP.
This does not itself authorize motion or replace independent physics checks.
"""
import math

import cv2
import numpy as np
from scipy.spatial import ConvexHull


def as_array(value):
    return value.detach().cpu().numpy() if hasattr(value,"detach") else np.asarray(value)


def rotation(yaw):
    return np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])


def convex_boundary(points):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError("Footprint points must be finite XY coordinates")
    hull=ConvexHull(points)
    return points[hull.vertices]


def capture_robot_footprint(io,payload_radius=None,padding=.025,payload_height=None):
    """Read actual cooked robot collision meshes; no simulator state setters."""
    if padding<0 or not np.isfinite(padding):
        raise ValueError("Invalid footprint padding")
    if payload_radius is not None and (not np.isfinite(payload_radius) or payload_radius<=0):
        raise ValueError("Invalid observed payload radius")
    if payload_height is not None and (not np.isfinite(payload_height) or payload_height<=0):
        raise ValueError("Invalid observed payload height")
    base=np.asarray(io.pose(),dtype=float);world_to_base=rotation(base[2])
    pieces=[];links=[]
    for link in io.robot.robot.get_links():
        for shape in link._objs[0].get_collision_shapes():
            if not hasattr(shape,"vertices"):
                raise ValueError(f"Unsupported robot collision shape: {link.name}/{type(shape).__name__}")
            vertices=np.asarray(shape.vertices,dtype=float)
            if hasattr(shape,"scale"):vertices=vertices*np.asarray(shape.scale)
            matrix=(link._objs[0].pose*shape.local_pose).to_transformation_matrix()
            world=vertices@matrix[:3,:3].T+matrix[:3,3]
            xy=(world[:,:2]-base[:2])@world_to_base
            pieces.append(xy)
            links.append(dict(link=link.name,vertices=len(vertices),
                maximum_radius_m=float(np.linalg.norm(xy,axis=1).max())))
    if not pieces:raise ValueError("No robot collision geometry available")
    robot=np.vstack(pieces);points=robot;payload=None
    if payload_radius is not None:
        tcp=as_array(io.robot.tcp_pose.p)[0]
        local=(tcp[:2]-base[:2])@world_to_base
        # Circumscribe, not inscribe, the observed-radius disk.
        count=64;angles=np.arange(count)*2*np.pi/count
        bound=math.hypot(payload_radius,payload_height) if payload_height is not None else payload_radius
        disk=local+(bound/math.cos(math.pi/count))*np.c_[np.cos(angles),np.sin(angles)]
        points=np.vstack((robot,disk))
        payload=dict(observed_radius_m=float(payload_radius),observed_height_m=payload_height,
                     conservative_projected_disk_radius_m=bound,tcp_base_xy=local.tolist(),
                     source="RGB-D policy dimensions plus measured robot TCP, not task actor geometry; full observed height included when available")
    boundary=convex_boundary(points)
    hull=ConvexHull(boundary)
    if np.any(hull.equations[:,2]>1e-6):
        raise ValueError("Measured convex footprint does not contain the base origin")
    return dict(vertices_base_xy=boundary.tolist(),padding_m=float(padding),
        base_xyyaw=base.tolist(),robot_qpos=as_array(io.robot.robot.get_qpos())[0].tolist(),
        maximum_robot_radius_m=float(np.linalg.norm(robot,axis=1).max()),
        maximum_combined_radius_m=float(np.linalg.norm(points,axis=1).max()),
        inscribed_radius_m=float((-hull.equations[:,2]).min()),
        robot_links=links,payload_proxy=payload,
        boundary="Current measured cooked robot collision meshes and optional observed payload disk. Projected convex hull is conservative; no scene/target truth. Snapshot only, not a safe trajectory or task verdict.")


def distances_to_polygon(points,polygon):
    """Zero inside a convex polygon, Euclidean edge distance outside."""
    points=np.asarray(points,dtype=float);polygon=np.asarray(polygon,dtype=float)
    hull=ConvexHull(polygon)
    inside=np.all(points@hull.equations[:,:2].T+hull.equations[:,2]<=1e-10,axis=1)
    starts=polygon;edges=np.roll(polygon,-1,axis=0)-polygon
    diff=points[:,None,:]-starts
    alpha=np.clip(np.sum(diff*edges,axis=2)/np.sum(edges*edges,axis=1),0.,1.)
    distance=np.linalg.norm(diff-alpha[:,:,None]*edges,axis=2).min(axis=1)
    distance[inside]=0.
    return distance


def pose_clear(known,origin,resolution,vertices,xy,yaw,padding=.04,inferred=None):
    """Collision check against observed occupied cells at an exact continuous pose.

    Every occupied raster cell is enclosed by its circumscribed disk. Only the
    base-center cell must be observed free, matching the existing map's center
    exploration rule; unknown cells are not treated as new occupied evidence.
    """
    known=np.asarray(known);origin=np.asarray(origin);xy=np.asarray(xy)
    center=np.floor((xy-origin)/resolution).astype(int)[::-1]
    if np.any(center<0) or np.any(center>=known.shape) or known[tuple(center)]!=1:
        return False
    polygon=np.asarray(vertices)@rotation(yaw).T+xy
    extent=padding+resolution/math.sqrt(2)
    lo=np.floor((polygon.min(axis=0)-extent-origin)/resolution).astype(int)[::-1]
    hi=np.floor((polygon.max(axis=0)+extent-origin)/resolution).astype(int)[::-1]+1
    if np.any(lo<0) or np.any(hi>known.shape):return False
    blocked=known[lo[0]:hi[0],lo[1]:hi[1]]==-1
    if inferred is not None:blocked|=inferred[lo[0]:hi[0],lo[1]:hi[1]]
    cells=np.argwhere(blocked)+lo
    if not len(cells):return True
    world=origin+(cells[:,::-1]+.5)*resolution
    return bool(np.all(distances_to_polygon(world,polygon)>extent))


def orientation_masks(known,resolution,vertices,padding=.04,inferred=None,headings=8,offset=(0.,0.)):
    """Raster pose feasibility, not a connected heading-space route or motion.

    Correlation uses float32 counts (not uint8, which could overflow and mark an
    occupied footprint clear). Each obstacle cell's full circumscribed disk is
    included. The union of headings is an optimistic connectivity diagnostic.
    """
    if headings<4:raise ValueError("At least four headings are required")
    known=np.asarray(known);vertices=np.asarray(vertices,dtype=float)
    occupied=known==-1
    if inferred is not None:occupied|=inferred
    obstacle_image=occupied.astype(np.float32)
    extent=padding+resolution/math.sqrt(2)
    offset=np.asarray(offset)
    radius=int(math.ceil((np.linalg.norm(vertices,axis=1).max()+np.linalg.norm(offset)+extent)/resolution))
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    points=np.c_[xx.ravel(),yy.ravel()]*resolution
    masks=[]
    for index in range(headings):
        polygon=vertices@rotation(index*2*np.pi/headings).T+offset
        kernel=(distances_to_polygon(points,polygon)<=extent).reshape(xx.shape).astype(np.float32)
        counts=cv2.filter2D(obstacle_image,-1,kernel,borderType=cv2.BORDER_CONSTANT)
        clear=(counts<.5)&(known==1)
        clear[:radius,:]=False;clear[-radius:,:]=False
        clear[:,:radius]=False;clear[:,-radius:]=False
        masks.append(clear)
    return np.asarray(masks)


def swept_clear(known,origin,resolution,vertices,start,end,padding=.04,inferred=None):
    """Conservative sampled rigid sweep with an explicit between-sample margin."""
    start=np.asarray(start,dtype=float);end=np.asarray(end,dtype=float)
    angle=math.atan2(math.sin(end[2]-start[2]),math.cos(end[2]-start[2]))
    radius=float(np.linalg.norm(vertices,axis=1).max())
    motion=np.linalg.norm(end[:2]-start[:2])+radius*abs(angle)
    intervals=max(1,int(math.ceil(motion/(resolution*.25))))
    for alpha in np.linspace(0.,1.,intervals+1):
        xy=start[:2]+alpha*(end[:2]-start[:2]);yaw=start[2]+alpha*angle
        if not pose_clear(known,origin,resolution,vertices,xy,yaw,
                          padding=padding+resolution*.125,inferred=inferred):
            return False
    return True


def swept_mask(known,resolution,vertices,start_yaw,angle=0.,translation=(0.,0.),
               padding=.04,inferred=None,offset=(0.,0.)):
    """Feasible discrete motion origins, including the entire rigid sweep."""
    vertices=np.asarray(vertices);translation=np.asarray(translation);offset=np.asarray(offset)
    motion=np.linalg.norm(translation)+np.linalg.norm(vertices,axis=1).max()*abs(angle)
    intervals=max(1,int(math.ceil(motion/(resolution*.25))))
    extent=padding+resolution/math.sqrt(2)+resolution*.125
    radius=int(math.ceil((np.linalg.norm(vertices,axis=1).max()+np.linalg.norm(offset)
                         +np.linalg.norm(translation)+extent)/resolution))
    yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    points=np.c_[xx.ravel(),yy.ravel()]*resolution
    covered=np.zeros(len(points),bool)
    for alpha in np.linspace(0.,1.,intervals+1):
        polygon=vertices@rotation(start_yaw+alpha*angle).T+offset+alpha*translation
        covered|=distances_to_polygon(points,polygon)<=extent
    occupied=np.asarray(known)==-1
    if inferred is not None:occupied|=inferred
    counts=cv2.filter2D(occupied.astype(np.float32),-1,
        covered.reshape(xx.shape).astype(np.float32),borderType=cv2.BORDER_CONSTANT)
    clear=(counts<.5)&(known==1)
    clear[:radius,:]=False;clear[-radius:,:]=False
    clear[:,:radius]=False;clear[:,-radius:]=False
    return clear
