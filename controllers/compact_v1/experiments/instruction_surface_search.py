"""Potential support surfaces from measured RGB-D, with no object/scene truth."""
import numpy as np
from scipy.ndimage import label


def compact_protrusions(xyz, region):
    """Measured shapes above a surface, without assigning an object identity."""
    height, width = xyz.shape[:2]
    x1, y1, x2, y2 = region['box_xyxy']
    crop = xyz[max(0, y1-40):min(height, y2+20),
               max(0, x1-20):min(width, x2+20)]
    point = np.asarray(region['point_world'])
    mask = np.isfinite(crop).all(axis=2)
    mask &= (crop[:, :, 2] > point[2]+.015) & (crop[:, :, 2] < point[2]+.30)
    mask &= np.linalg.norm(crop[:, :, :2]-point[:2], axis=2) < .55
    components, count = label(mask)
    shapes = []
    for component in range(1, count+1):
        cloud = crop[components == component]
        if len(cloud) < 4:
            continue
        extent = np.ptp(cloud, axis=0)
        # Exclude plane noise, thin vertical sheets, and broad furniture.
        if not (.012 <= min(extent[:2]) <= .20 and max(extent[:2]) <= .45
                and extent[2] >= .008):
            continue
        shapes.append(dict(point_world=np.median(cloud, axis=0).tolist(),
                           extent_m=extent.tolist(), observed_points=len(cloud)))
    return shapes


def observed_support_surfaces(observation):
    xyz=observation.world_points()
    dx=xyz[1:-1,2:]-xyz[1:-1,:-2]
    dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
    normal=np.cross(dx,dy);magnitude=np.linalg.norm(normal,axis=2)
    center=xyz[1:-1,1:-1];depth=observation.depth_m[1:-1,1:-1]
    horizontal=np.abs(normal[:,:,2])>=.94*magnitude
    valid=horizontal&np.isfinite(center).all(axis=2)&np.isfinite(magnitude)&(magnitude>1e-8)
    valid&=(depth>=.25)&(depth<=4.)&(center[:,:,2]>=.45)&(center[:,:,2]<=1.25)
    # Keep only local, depth-continuous surface tangents, not silhouette jumps.
    valid&=(np.linalg.norm(dx,axis=2)<.15)&(np.linalg.norm(dy,axis=2)<.15)
    valid&=np.linalg.norm(center[:,:,:2]-observation.base_xyyaw[:2],axis=2)>.70
    step=2;mask=valid[::step,::step];points=center[::step,::step]
    areas=np.abs(normal[::step,::step,2])/4*step**2
    components,count=label(mask);regions=[]
    for component in range(1,count+1):
        ys,xs=np.nonzero(components==component)
        if len(ys)<12:continue
        cloud=points[ys,xs];area=float(areas[ys,xs].sum())
        if area<.015 or np.ptp(cloud[:,2])>.035:continue
        point=np.median(cloud,axis=0)
        regions.append(dict(point_world=point.tolist(),observation_id=observation.frame_id,
            visible_area_m2=area,observed_points=len(cloud),height_spread_m=float(np.ptp(cloud[:,2])),
            box_xyxy=[int(xs.min()*step+1),int(ys.min()*step+1),int(xs.max()*step+2),int(ys.max()*step+2)],
            source='measured_horizontal_RGBD_surface_search_prior_only'))
    regions=sorted(regions,key=lambda r:-r['visible_area_m2'])[:24]
    for region in regions:
        region['compact_protrusions']=compact_protrusions(xyz,region)
    return regions


def named_support_surfaces(observation, detections):
    """Horizontal search priors inside a freshly detected support's image box.

    A detected table can guide a closer look for a flat receptacle. This
    provides no receptacle identity and cannot authorize a placement.
    """
    if not detections:return []
    regions=observed_support_surfaces(observation)
    selected=[]
    for region in regions:
        box=np.asarray(region['box_xyxy'],dtype=float)
        area=float(np.prod(np.maximum(0.,box[2:]-box[:2])))
        if area<=0:continue
        for detection in detections:
            if detection.observation_id!=observation.frame_id:continue
            support=np.asarray(detection.box_xyxy,dtype=float)
            if support.shape!=(4,) or not np.isfinite(support).all():continue
            overlap=np.maximum(0.,np.minimum(box[2:],support[2:])-np.maximum(box[:2],support[:2]))
            if float(np.prod(overlap))/area<.6:continue
            selected.append(dict(region,support_detection=dict(
                observation_id=detection.observation_id,query=detection.query,
                score=float(detection.score),box_xyxy=list(detection.box_xyxy)),
                source='observed_named_support_surface_search_prior_only'))
            break
    return selected


class SurfaceSearch:
    """A small budget of close inspections before ordinary frontier search.

    Surface coordinates are camera-derived exploration priors. They never
    become object detections, grasp proposals, or an assertion of success.
    """
    def __init__(self, inspection_budget=4, require_protrusions=True):
        self.regions=[]
        self.active=None
        self.inspections=0
        self.attempts=0
        self.attempted_surfaces=[]
        self.inspection_budget=inspection_budget
        self.require_protrusions=require_protrusions
        self.next_id=0

    def update(self, observation, observed_empty, *, regions=None):
        retained=[]
        for row in self.regions:
            if not observed_empty(observation,row['point_world']):retained.append(row)
            elif self.active is not None and self.active['region_id']==row['id']:
                self.active=None
        self.regions=retained
        for region in (observed_support_surfaces(observation) if regions is None else regions):
            point=np.asarray(region['point_world'])
            match=next((r for r in self.regions
                if np.linalg.norm(point[:2]-np.asarray(r['point_world'])[:2])<.40
                and abs(point[2]-r['point_world'][2])<.05),None)
            if match is None:
                self.next_id+=1
                self.regions.append(dict(region,id=self.next_id,inspected=False))
            else:
                match.update(region)
        self.regions=self.regions[-64:]

    def choose(self, occupancy, safe, distances, base_xy, excluded=(), region_center=None):
        self.active=None
        # Charge selection, including a route that later fails, so an
        # inaccessible surface cannot consume unbounded planning rounds.
        if self.attempts>=self.inspection_budget:return None
        cells=np.argwhere(safe&np.isfinite(distances))
        if not len(cells):return None
        positions=occupancy.world(cells)
        options=[]
        for row in self.regions:
            if row['inspected']:continue
            # Spend the small inspection budget on visibly nonempty surfaces.
            # Actual language-grounded perception must still identify the item.
            if self.require_protrusions and not row.get('compact_protrusions'):continue
            point=np.asarray(row['point_world'])
            if region_center is not None and np.linalg.norm(point[:2]-region_center)>4.:continue
            if any(np.linalg.norm(point[:2]-old[:2])<1.0 and abs(point[2]-old[2])<.15
                   for old in self.attempted_surfaces):continue
            if any(np.linalg.norm(point[:2]-np.asarray(p)[:2])<.45 for p in excluded):continue
            ranges=np.linalg.norm(positions-point[:2],axis=1)
            allowed=(ranges>=.85)&(ranges<=1.05)
            allowed&=np.linalg.norm(positions-np.asarray(base_xy),axis=1)>=.25
            if not allowed.any():continue
            indices=np.flatnonzero(allowed)
            costs=distances[cells[indices,0],cells[indices,1]]*occupancy.resolution
            index=int(indices[np.argmin(costs)])
            options.append((float(costs.min()),row,cells[index],positions[index]))
        if not options:return None
        _,row,cell,goal=min(options,key=lambda x:x[0])
        self.active=dict(region_id=row['id'],point_world=row['point_world'],goal_xy=goal.tolist(),
                         observation_id=row['observation_id'])
        if 'support_detection' in row:self.active['support_detection']=row['support_detection']
        self.attempts+=1
        self.attempted_surfaces.append(np.asarray(row['point_world']).copy())
        return tuple(cell)

    def finish(self, region_id):
        for row in self.regions:
            if row['id']==region_id:row['inspected']=True
        self.inspections+=1
        self.active=None
