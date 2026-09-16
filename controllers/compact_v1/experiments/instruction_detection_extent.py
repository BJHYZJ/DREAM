"""Candidate completion of a detected upper part from the same measured RGB-D.

No category, actor, task layout, evaluator mask or known object size is an input.
Return None unless a compact, connected column and a local support are observed.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def refine_detected_extent(xyz,depth,box):
 h,w=depth.shape;box=np.asarray(box,dtype=float)
 if box.shape!=(4,) or not np.isfinite(box).all():return None
 x0,y0=np.maximum(np.floor(box[:2]).astype(int),0);x1,y1=np.minimum(np.ceil(box[2:]).astype(int),(w,h))
 if x1<=x0 or y1<=y0:return None
 valid=np.isfinite(xyz).all(-1)&np.isfinite(depth)&(depth>.10)&(depth<4.)
 seed=np.zeros((h,w),bool);seed[y0:y1,x0:x1]=valid[y0:y1,x0:x1]
 if seed.sum()<24:return None
 initial=xyz[seed];anchor=np.median(initial,axis=0)
 # A part box can also contain distant background through its empty corners.
 # Retain only the local surface around its robust measured depth, with
 # enough original support to avoid selecting an isolated noise point.
 local_seed=seed&(np.linalg.norm(xyz-anchor,axis=-1)<.12)
 if local_seed.sum()<max(24,.30*seed.sum()):return None
 seed=local_seed;initial=xyz[seed];anchor=np.median(initial,axis=0)
 low,high=np.quantile(initial[:,2],[.05,.95]);initial_height=high-low
 # Fully framed objects need no completion. This adapter targets small upper parts.
 if initial_height>.14 or np.ptp(np.quantile(initial[:,:2],[.05,.95],axis=0),axis=0).max()>.14:return None
 dx=xyz[1:-1,2:]-xyz[1:-1,:-2];dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
 normal=np.cross(dx,dy);magnitude=np.linalg.norm(normal,axis=-1);middle=xyz[1:-1,1:-1]
 plane=(np.abs(normal[:,:,2])>.97*magnitude)&(magnitude>1e-8)
 plane&=np.isfinite(magnitude)&valid[1:-1,1:-1]
 plane&=(np.linalg.norm(dx,axis=-1)<.06)&(np.linalg.norm(dy,axis=-1)<.06)
 plane&=np.linalg.norm(middle[:,:,:2]-anchor[:2],axis=-1)<.28
 plane&=(middle[:,:,2]>anchor[2]-.35)&(middle[:,:,2]<low-.025)
 heights=middle[:,:,2][plane]
 if len(heights)<80:return None
 bins=np.rint(heights/.008).astype(int);values,counts=np.unique(bins,return_counts=True)
 peak=values[np.argmax(counts)]*.008;support_points=heights[np.abs(heights-peak)<.012]
 if len(support_points)<80:return None
 support=float(np.median(support_points))
 mask=valid&(np.linalg.norm(xyz[:,:,:2]-anchor[:2],axis=-1)<.15)
 mask&=(xyz[:,:,2]>support+.008)&(xyz[:,:,2]<min(high+.035,support+.35))
 if (mask&seed).sum()<24:return None
 indices=np.full((h,w),-1,np.int32);indices[mask]=np.arange(mask.sum())
 rows=[];columns=[]
 for left,right in ((np.s_[:-1,:],np.s_[1:,:]),(np.s_[:,:-1],np.s_[:,1:])):
  adjacent=mask[left]&mask[right]&(np.linalg.norm(xyz[left]-xyz[right],axis=-1)<.025)
  rows.append(indices[left][adjacent]);columns.append(indices[right][adjacent])
 row=np.concatenate(rows);column=np.concatenate(columns)
 graph=coo_matrix((np.ones(len(row),np.uint8),(row,column)),shape=(mask.sum(),mask.sum())).tocsr()
 _,labels=connected_components(graph,directed=False)
 overlap=np.bincount(labels[seed[mask]],minlength=labels.max()+1);chosen=int(overlap.argmax())
 selected=np.zeros((h,w),bool);selected[mask]=labels==chosen
 if (selected&seed).sum()<.5*seed.sum():return None
 points=xyz[selected];extent=np.ptp(points,axis=0)
 if not (.06<extent[2]<.35 and extent[:2].max()<.22 and initial_height<.65*extent[2]):return None
 ys,xs=np.nonzero(selected);new_box=np.asarray([xs.min(),ys.min(),xs.max()+1,ys.max()+1],float)
 if np.prod(new_box[2:]-new_box[:2])<1.4*np.prod(box[2:]-box[:2]):return None
 return dict(box_xyxy=new_box.tolist(),point_world=np.median(points,axis=0).tolist(),
  support_height_m=support,extent_m=extent.tolist(),points=len(points),original_box_xyxy=box.tolist(),
  evidence='Connected compact RGBD surface above a measured local horizontal support; original confidence unchanged')
