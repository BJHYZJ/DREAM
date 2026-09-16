"""RGB-D body estimate for observed flat, elongated objects."""
import cv2
import numpy as np
from instruction_geometry import connected_depth_surface


def observed_flat_body(observation, detection, fit, tcp_matrix):
    if fit['height_m'] > .04:
        raise ValueError('Not an observed flat object')
    if observation.frame_id != detection.observation_id:
        raise ValueError('Observation mismatch')
    x0, y0 = np.floor(detection.box_xyxy[:2]).astype(int)
    x1, y1 = np.ceil(detection.box_xyxy[2:]).astype(int)
    h, w = observation.depth_m.shape
    if x0 < 1 or y0 < 1 or x1 >= w - 1 or y1 >= h - 1:
        raise ValueError('Clipped object observation')
    world = observation.world_points()
    px0,py0=max(0,x0-8),max(0,y0-8)
    px1,py1=min(w,x1+8),min(h,y1+8)
    xyz = world[py0:py1, px0:px1]
    rgb = observation.rgb[py0:py1, px0:px1]
    support = float(fit['support_height_m'])
    border = max(8,int(max(x1-x0,y1-y0)*.35))
    ring = np.zeros((h,w),bool)
    ring[max(0,y0-border):min(h,y1+border),max(0,x0-border):min(w,x1+border)] = True
    ring[y0:y1,x0:x1] = False
    plane = world[...,2][ring & np.isfinite(world).all(-1)
        & (np.abs(world[...,2]-support)<.003)]
    if len(plane)<48:
        raise ValueError('Insufficient support precision for a flat body')
    uncertainty = max(.0005,4*1.4826*float(np.median(np.abs(plane-np.median(plane)))))
    if uncertainty>.003:
        raise ValueError('Support depth is too uncertain for a thin body')
    inside_box=np.zeros(xyz.shape[:2],bool)
    inside_box[y0-py0:y1-py0,x0-px0:x1-px0]=True
    selected = inside_box & np.isfinite(xyz).all(-1)
    # Use confident depth for the foreground seed. Appearance may recover thin
    # ends at the measured support plane; the padded box supplies real background.
    selected &= (xyz[..., 2] > support - uncertainty) & (xyz[..., 2] < support + .04)
    selected = connected_depth_surface(xyz, selected)
    yy, xx = np.indices(selected.shape)
    core = selected & (xyz[...,2]>support+uncertainty)
    core &= (xx > x0-px0+.3*(x1-x0)) & (xx < x0-px0+.7*(x1-x0))
    core &= (yy > y0-py0+.25*(y1-y0)) & (yy < y0-py0+.75*(y1-y0))
    if core.sum() < 24 or (~selected).sum() < 24:
        raise ValueError('Insufficient appearance separation')
    mask = np.full(selected.shape, cv2.GC_BGD, np.uint8)
    mask[selected] = cv2.GC_PR_FGD
    mask[core] = cv2.GC_FGD
    cv2.setRNGSeed(0)
    cv2.grabCut(np.ascontiguousarray(rgb), mask, None, np.zeros((1, 65)),
                np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    chosen = selected & ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
    points = xyz[chosen]
    if len(points) < 48:
        raise ValueError('Incomplete body foreground')
    origin = points[:, :2].mean(0)
    _, axes = np.linalg.eigh(np.cov((points[:, :2] - origin).T))
    lo, hi = np.quantile((points[:, :2] - origin) @ axes, [.02, .98], axis=0)
    short, long = sorted(hi - lo)
    height = float(np.quantile(points[:, 2], .98) - support)
    if not (.012 < short < .06 and .08 < long < .30 and long / short >= 3
            and .003 < height < .04):
        raise ValueError('Foreground is not a complete flat elongated body')
    center = np.r_[origin + axes @ ((lo + hi) / 2), support + height / 2]
    radius = float(np.linalg.norm(points[:, :2] - center[:2], axis=1).max())
    if radius > .17:
        raise ValueError('Observed extent exceeds bounded flat body adapter')
    # Detector boxes include padding and can contain thin tips close to the
    # depth plane. Require the complete box projected onto the measured support
    # to fit inside the existing 25 mm self-model uncertainty. This bounds any
    # missed ends in metric space instead of counting background image pixels.
    corners = np.array([[x0,y0,1.],[x1-1,y0,1.],[x1-1,y1-1,1.],[x0,y1-1,1.]])
    camera = observation.camera_to_world_cv
    rays = (corners @ np.linalg.inv(observation.intrinsics).T) @ camera[:3,:3].T
    if np.any(np.abs(rays[:,2]) < 1e-6):
        raise ValueError('Support projection is ill-conditioned')
    distance = (support-camera[2,3])/rays[:,2]
    if np.any((distance <= .10) | (distance > 4.)):
        raise ValueError('Support projection is outside the measured depth range')
    projected = camera[:3,3]+distance[:,None]*rays
    corner_extent = float(np.linalg.norm(projected[:,:2]-center[:2],axis=1).max())
    if corner_extent > max(radius,fit['radius_m'])+.025:
        raise ValueError('Full observed object box exceeds the bounded body model')
    matrix = np.asarray(tcp_matrix)
    return dict(frame=dict(center_tcp_m=((center - matrix[:3, 3]) @ matrix[:3, :3]).tolist(),
                           axis_tcp=(np.array([0., 0., 1.]) @ matrix[:3, :3]).tolist(),
                           observation_id=observation.frame_id),
                radius_m=max(radius, fit['radius_m']), height_m=max(height, fit['height_m']),
                observed_center_world=center.tolist(), observed_height_m=height,
                observed_widths_xy_m=[float(short), float(long)], points=len(points),
                projected_box_extent_m=corner_extent,
                support_separation_m=uncertainty,
                source='Text-box central appearance and measured RGB-D support plane',
                boundary='Observed body estimate only; does not change the grasp action')
