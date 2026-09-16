"""Diagnostic RGB-D finger contact band fit, preserving observed grasp height."""
import numpy as np
import cv2

def contact_band_grasp(observation,target,fit,shape,choose_axis=False):
 from instruction_open_jaw_axis import pad_observations
 if fit['height_m']<=.20 or fit.get('grasp_section_points') is None:return target,fit
 points=np.asarray(shape.points);band=points[(points[:,2]>=target[2]-.031)&(points[:,2]<=target[2]+.031)]
 if len(band)<96:return target,fit
 axis=np.asarray(fit['closing_axis_xy'],dtype=float)
 if choose_axis:
  rectangle=cv2.boxPoints(cv2.minAreaRect(np.ascontiguousarray(band[:,:2],dtype=np.float32)))
  edge=rectangle[1]-rectangle[0];first=edge/np.linalg.norm(edge);axes=np.column_stack((first,[-first[1],first[0]]))
  widths=np.diff(np.quantile(band[:,:2]@axes,[.02,.98],axis=0),axis=0)[0];axis=axes[:,np.argmin(widths)]
 along=np.array([-axis[1],axis[0]]);axes=np.column_stack((axis,along));lo,hi=np.quantile(band[:,:2]@axes,[.02,.98],axis=0)
 center=axes@((lo+hi)/2);width=float(hi[0]-lo[0]);shift=float(np.linalg.norm(center-target[:2]))
 if not (.012<width<.0874 and width>1.5*fit['closing_width_m'] and .008<shift<.025):return target,fit
 proposed=np.r_[center,target[2]];world=observation.world_points().reshape(-1,3);depth=observation.depth_m.reshape(-1)
 world=world[np.isfinite(world).all(1)&np.isfinite(depth)&(depth>.10)&(depth<4.)]
 if sum(pad_observations(world,proposed,axis)):return target,fit
 return proposed,dict(fit,closing_axis_xy=axis.tolist(),closing_width_m=width,
     jaw_axis_choice='observed_full_finger_contact_band',contact_band_evidence=dict(points=len(band),center_shift_m=shift,half_band_m=.031,original_neck_width_m=fit['closing_width_m'],axis_reconsidered=choose_axis,source='same RGBD and physical finger length'))
