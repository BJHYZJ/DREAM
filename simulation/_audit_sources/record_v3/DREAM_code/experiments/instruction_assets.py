"""Environment-only household asset construction for instruction task recipes.

No function from this module may be called to localize targets for the policy.
Static receptacles use their concave triangle mesh, never a convex hull that
would close the bowl opening. Dynamic pickups currently use a conservative
convex collision mesh; handle/cavity-dependent grasps are not supported by it.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import numpy as np
import sapien
import trimesh


@dataclass(frozen=True)
class HouseholdAsset:
    name: str
    path: Path
    scale: float
    lower: np.ndarray
    upper: np.ndarray
    sha256: str


def inspect_asset(asset_directory,name,scale=1.):
    if not re.fullmatch(r"[A-Za-z0-9_]+",name) or not np.isfinite(scale) or scale<=0:
        raise ValueError("Invalid asset name or physical scale")
    path=Path(asset_directory)/(name+".glb")
    scene=trimesh.load(path,force="scene")
    vertices=np.asarray(scene.to_geometry().vertices)
    rotation=np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    points=vertices@rotation.T*scale
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Invalid asset mesh")
    return HouseholdAsset(name,path,scale,points.min(0),points.max(0),hashlib.sha256(path.read_bytes()).hexdigest())


def build_household_asset(base,spec,*,xy,support_height,role):
    if role not in ("pickup","placement"):
        raise ValueError("Asset role must be pickup or placement")
    builder=base.scene.create_actor_builder()
    rotation=sapien.Pose(q=[math.sqrt(.5),math.sqrt(.5),0,0])
    scale=[spec.scale]*3
    material=sapien.physx.PhysxMaterial(.8,.6,0.)
    builder.add_visual_from_file(str(spec.path),pose=rotation,scale=scale)
    if role=="placement":
        builder.add_nonconvex_collision_from_file(str(spec.path),pose=rotation,scale=scale,material=material)
    else:
        builder.add_convex_collision_from_file(str(spec.path),pose=rotation,scale=scale,material=material,density=700)
    xyz=np.r_[np.asarray(xy,dtype=float),support_height-spec.lower[2]+.002]
    builder.set_initial_pose(sapien.Pose(xyz))
    actor=(builder.build_static if role=="placement" else builder.build)(name=f"instruction_{role}_{spec.name}")
    if role=="pickup":
        actor.linear_damping=1.
        actor.angular_damping=1.
    metadata=dict(asset=spec.name,asset_sha256=spec.sha256,physical_scale=spec.scale,
        local_bounds_z_up_m=[spec.lower.tolist(),spec.upper.tolist()],initial_origin_xyz=xyz.tolist(),
        collision="static_concave_triangle_mesh" if role=="placement" else "dynamic_convex_mesh",
        boundary="Environment construction/evaluation metadata, never a policy localization input.")
    return actor,metadata
