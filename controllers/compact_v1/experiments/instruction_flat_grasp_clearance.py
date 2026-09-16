"""Select a nearby observed flat-body section with room for the open Fetch pads."""
import numpy as np
from instruction_geometry import observed_shape

def observed_clear_flat_grasp(observation,detection,target,fit):
    if fit.get('foreground_recovery') not in ('central_appearance_after_flat_width_rejection','central_appearance_for_flat_grasp'):
        return target,fit
    shape=observed_shape(observation,detection,refine_foreground=True,foreground_seed='central')
    closing=np.asarray(fit['closing_axis_xy']);along=np.array([-closing[1],closing[0]])
    axes=np.column_stack((closing,along));points=shape.points
    local=(points[:,:2]-target[:2])@axes
    bounds=np.quantile(local[:,1],[.10,.90])
    world=observation.world_points().reshape(-1,3)
    world=world[np.isfinite(world).all(-1)&(np.abs(world[:,2]-fit['support_height_m'])<.08)]
    trials=[]
    for offset in (0.,.02,-.02,.04,-.04,.06,-.06):
        if not bounds[0]+.015<=offset<=bounds[1]-.015:continue
        band=local[np.abs(local[:,1]-offset)<.015,0]
        if len(band)<48:continue
        lo,hi=np.quantile(band,[.02,.98]);width=float(hi-lo)
        if not .012<width<.094:continue
        center=target[:2]+offset*along+(lo+hi)/2*closing
        opening=min(.10,width+.012)
        measured=(world[:,:2]-center)@axes;pads=[]
        for sign in (-1,1):
            selected=world[(measured[:,0]*sign>opening/2-.001)
                &(measured[:,0]*sign<opening/2+.015)&(np.abs(measured[:,1])<.015)]
            pads.append(dict(points=len(selected),top_m=float(np.quantile(selected[:,2],.98)) if len(selected) else None))
        clear=all(p['points']>=12 and p['top_m']<=fit['support_height_m']+.003 for p in pads)
        trials.append(dict(longitudinal_offset_m=offset,closing_width_m=width,opening_m=opening,pads=pads,clear=clear))
        if clear:
            goal=target.copy();goal[:2]=center
            return goal,dict(fit,closing_width_m=width,grasp_clearance=dict(trials=trials,
                object_center_world=fit['object_center_world'],selected_grasp_world=goal.tolist(),
                opening_m=opening,finger_pad_width_m=.014,finger_pad_half_depth_m=.01315,
                source='Observed RGB-D surface heights in calibrated open-finger footprints'))
    raise ValueError('No observed flat body section has clearance for both open finger pads')
