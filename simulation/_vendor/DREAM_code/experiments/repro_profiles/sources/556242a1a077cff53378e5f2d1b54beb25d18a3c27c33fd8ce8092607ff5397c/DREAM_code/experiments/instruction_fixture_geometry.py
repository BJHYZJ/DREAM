"""Environment-only furniture fitting; never an input to the learned policy."""
import numpy as np
import cv2
from scipy.spatial import ConvexHull


def mesh_footprint(spec):
    import trimesh
    vertices=np.asarray(trimesh.load(spec.path,force="scene").to_geometry().vertices)
    points=vertices[:,[0,2]]*spec.scale
    # Asset y-up -> world z-up maps (x,z) to (x,-z).
    points[:,1]*=-1
    center=(spec.lower[:2]+spec.upper[:2])/2
    points-=center
    hull=ConvexHull(points)
    return points[hull.vertices],hull.equations,center


def furniture_center_candidates(room_map,footprint,room,reference,max_distance=3.):
    resolution=float(room_map["resolution"])
    radius=int(np.ceil(np.max(np.abs(footprint))/resolution))+2
    kernel=np.zeros((2*radius+1,2*radius+1),np.uint8)
    polygon=np.rint(footprint*np.array([1.,-1.])/resolution).astype(np.int32)+radius
    cv2.fillConvexPoly(kernel,polygon,1)
    kernel=cv2.dilate(kernel,np.ones((3,3),np.uint8))
    clear=cv2.erode(room_map["raw_free"].astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    cells=np.argwhere(clear&(room_map["room_cores"]==room))
    xy=np.column_stack((room_map["minimum_xy"][0]+cells[:,1]*resolution,
                        room_map["maximum_xy"][1]-cells[:,0]*resolution))
    distance=np.linalg.norm(xy-reference,axis=1)
    eligible=np.flatnonzero(distance<=max_distance)
    eligible=eligible[np.argsort(distance[eligible],kind="stable")]
    # Deterministic environment-only selection, independent of policy outcomes.
    return xy[eligible],dict(valid_furniture_centers=len(eligible),footprint_margin_cells=1,
                            reference_max_distance_m=max_distance)


def edge_placement(center,equations,observer_hint,receptacle_radius):
    direction=np.asarray(observer_hint)-center
    direction/=max(1e-9,np.linalg.norm(direction))
    projected=equations[:,:2]@direction
    positive=projected>1e-8
    reach=float(np.min(-equations[positive,2]/projected[positive]))
    inset=receptacle_radius+.025
    if reach<=inset:
        raise ValueError("Receptacle does not fit the support footprint")
    return center+direction*(reach-inset)
