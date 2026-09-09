"""Environment/evaluator-only task fixtures, random starts and relocation.

Never import this module from instruction_policy or learned memory. Complete
scene geometry is used only to construct safe initial conditions, not navigation.
"""
from dataclasses import asdict
import math
import re

import cv2
import numpy as np
import sapien
from scipy.ndimage import distance_transform_edt

from instruction_assets import inspect_asset,build_household_asset
from instruction_task import DiscoveryRelocationGate,VisualDiscovery
from maniskill_learned_probe import array


def sample_start(room_map,initial_room,seed,fixture_discs,*,target_xy,docking_xy=None,robot_radius=.40):
    raw=room_map["raw_free"].astype(bool)
    resolution=float(room_map["resolution"])
    clearance=distance_transform_edt(raw)*resolution
    rows,cols=np.indices(raw.shape)
    xy=np.stack((room_map["minimum_xy"][0]+cols*resolution,
                 room_map["maximum_xy"][1]-rows*resolution),axis=-1)
    safe=raw&(clearance>=robot_radius)
    for center,radius in fixture_discs:
        safe&=np.linalg.norm(xy-np.asarray(center),axis=-1)>radius+robot_radius+.04
    # Environment geometry may reject disconnected initialization positions.
    # This does not give the policy a route, a map, or a target location.
    _,components=cv2.connectedComponents(safe.astype(np.uint8),connectivity=8)
    offsets=np.linalg.norm(xy-np.asarray(target_xy),axis=-1)
    docking_offsets=np.linalg.norm(xy-np.asarray(target_xy if docking_xy is None else docking_xy),axis=-1)
    docks=safe&(docking_offsets>.58)&(docking_offsets<.78)
    reachable_ids=np.unique(components[docks])
    reachable_ids=reachable_ids[reachable_ids!=0]
    candidates=np.argwhere(safe&(clearance>=robot_radius+.04)&(room_map["room_cores"]==initial_room)&(offsets>=2.)
                           &np.isin(components,reachable_ids))
    if not len(candidates):
        raise ValueError("No valid connected random start under the declared footprint/room constraints")
    rng=np.random.default_rng(seed)
    selected=candidates[int(rng.integers(len(candidates)))]
    return xy[tuple(selected)],float(rng.uniform(-math.pi,math.pi)),dict(
        seed=int(seed),valid_candidate_count=len(candidates),sampling="uniform over valid room-core grid cells",
        footprint_radius_m=robot_radius,initial_room=int(initial_room),minimum_pickup_distance_m=2.)


def build_pickup_cart(base,spec,xy,*,rim_height=.02):
    """A visible external moving support, with a separate dynamic payload.

    This is an explicit cart fixture, not an invisible grasp aid or attachment.
    Its low-friction floor contact and locked roll/pitch/yaw are recorded.
    """
    builder=base.scene.create_actor_builder()
    floor=sapien.physx.PhysxMaterial(.02,.01,0.)
    top=sapien.physx.PhysxMaterial(1.,.8,0.)
    for z,half,material in ((.02,[.08,.08,.02],floor),(.455,[.025,.025,.435],top),(.90,[.15,.15,.02],top)):
        builder.add_box_collision(pose=sapien.Pose([0,0,z]),half_size=half,material=material,density=150)
        builder.add_box_visual(pose=sapien.Pose([0,0,z]),half_size=half,material=[.18,.28,.33,1.])
    # A real, visible tray lip retains rolling household objects under external
    # motion. It is collision geometry, not a payload constraint/attachment.
    if rim_height:
        if not 0 < rim_height <= .04:
            raise ValueError("Invalid physical tray rim height")
        for center,half in (([.152,0,.92+rim_height/2],[.004,.156,rim_height/2]),
                            ([-.152,0,.92+rim_height/2],[.004,.156,rim_height/2]),
                            ([0,.152,.92+rim_height/2],[.148,.004,rim_height/2]),
                            ([0,-.152,.92+rim_height/2],[.148,.004,rim_height/2])):
            builder.add_box_collision(pose=sapien.Pose(center),half_size=half,material=top,density=150)
            builder.add_box_visual(pose=sapien.Pose(center),half_size=half,material=[.18,.28,.33,1.])
    builder.set_initial_pose(sapien.Pose([*xy,.001]))
    cart=builder.build(name="instruction_external_cart")
    cart.set_locked_motion_axes([False,False,False,True,True,True])
    cart.linear_damping=3.
    target,metadata=build_household_asset(base,spec,xy=xy,support_height=.921,role="pickup")
    return cart,target,dict(target=metadata,cart=dict(top_height_m=.921,
        top_half_extent_m=.15,tray_rim_height_m=rim_height,floor_friction=[.02,.01],locked_rotation_axes=True,
        boundary="Separate visible externally driven support; payload remains unattached and dynamic"))


class DiscoveryTriggeredMove:
    def __init__(self,io,cart,target,endpoint,waypoints,*,query,threshold=.15,maximum_speed=.12,evaluation_output=None):
        self.io,self.cart,self.target=io,cart,target
        self.points=[np.asarray(p,dtype=float) for p in [*waypoints,endpoint]]
        self.maximum_speed=maximum_speed
        self.index=0
        self.done_step=None
        self.gate=DiscoveryRelocationGate(query,minimum_confidence=threshold)
        self.stage="pickup_search"
        self.trace=[]
        self.observation_checks=[]
        self.initial=array(target.pose.p)[0].copy()
        self.evaluation_output=evaluation_output
        if evaluation_output is not None:
            evaluation_output.mkdir(parents=True,exist_ok=False)

    def observed(self,observation,detection,stage):
        self.stage=stage
        if detection is None or stage!="pickup_search":
            return
        # Independent ground-truth identity check only arms the environment
        # disturbance. Its answer/coordinates are never returned to the policy.
        error=float(np.linalg.norm(np.asarray(detection.point_world)-array(self.target.pose.p)[0]))
        # Read evaluator-only actor segmentation from the *same* captured view.
        # The learned observation schema never contains this channel.
        data=self.io.base._sensors[observation.sensor].get_obs(rgb=False,depth=False,position=False,segmentation=True)
        segmentation=array(data["segmentation"])[0,...,0]
        mask=segmentation==int(array(self.target.per_scene_id)[0])
        pixels=np.argwhere(mask)
        visible=len(pixels)
        iou=0.
        if visible:
            lo=pixels.min(0)[::-1]; hi=pixels.max(0)[::-1]+1
            box=np.asarray(detection.box_xyxy)
            overlap=np.maximum(0,np.minimum(hi,box[2:])-np.maximum(lo,box[:2])).prod()
            union=(hi-lo).prod()+np.maximum(0,box[2:]-box[:2]).prod()-overlap
            iou=float(overlap/max(1.,union))
        correct=bool(error<.15 and visible>=12 and iou>=.25)
        if self.evaluation_output is not None:
            np.savez_compressed(self.evaluation_output/f"visibility_{observation.frame_id:05d}.npz",target_mask=mask)
        self.observation_checks.append(dict(step=observation.sim_step,observation_id=observation.frame_id,
            detection=asdict(detection),evaluator_position_error_m=error,evaluator_correct_identity=correct,
            evaluator_visible_target_pixels=visible,evaluator_detection_box_iou=iou))
        if correct:
            self.gate.record(VisualDiscovery(observation.frame_id,observation.sim_step,detection.query,
                                            tuple(detection.point_world),detection.score))

    def before_step(self,stage):
        self.stage=stage
        translating=False
        if len(self.io.trace)>=2:
            displacement=np.linalg.norm(np.asarray(self.io.trace[-1]["base_xyyaw"][:2])-
                                        np.asarray(self.io.trace[-2]["base_xyyaw"][:2]))
            translating=displacement*self.io.base.control_freq>.02
        started=self.gate.should_start(sim_step=self.io.step_id,robot_xy=self.io.pose()[:2],
            translating=translating and stage=="pickup_search",manipulation_started=stage!="pickup_search")
        if self.gate.started_step is None or self.done_step is not None:
            return
        position=array(self.cart.pose.p)[0,:2]
        if self.index<len(self.points)-1 and np.linalg.norm(self.points[self.index]-position)<.05:
            self.index+=1
        delta=self.points[self.index]-position
        distance=float(np.linalg.norm(delta))
        velocity=array(self.cart.get_linear_velocity())[0,:2]
        desired=delta/max(distance,1e-9)*min(self.maximum_speed,distance*.8)
        force=np.clip(300*(desired-velocity),-30,30)
        self.cart.apply_force(np.r_[force,0.].astype(np.float32))
        self.trace.append(dict(step=self.io.step_id,position=position.tolist(),force=force.tolist(),
            waypoint=self.index,started_now=started,robot_translating=bool(translating)))
        if self.index==len(self.points)-1 and distance<.02 and np.linalg.norm(velocity)<.01:
            self.done_step=self.io.step_id
            self.cart.set_locked_motion_axes([True]*6)


def create_fixture(base,asset_directory,task):
    recipe=task["recipe"]
    pickup=inspect_asset(asset_directory,recipe["environment_assets"]["pickup"],recipe.get("pickup_scale",1.))
    receptacle=inspect_asset(asset_directory,recipe["environment_assets"]["placement"])
    table=inspect_asset(asset_directory,task["placement_table_asset"],task.get("placement_table_scale",1.))
    _,table_meta=build_household_asset(base,table,xy=task["placement_table_xy"],support_height=0.,role="placement")
    table_z=float(table.upper[2]-table.lower[2]+.002)
    destination,destination_meta=build_household_asset(base,receptacle,xy=task["placement_xy"],
                                                      support_height=table_z,role="placement")
    cart,target,cart_meta=build_pickup_cart(base,pickup,task["target_xy"],rim_height=task.get("cart_rim_height_m",0.))
    return cart,target,destination,pickup,receptacle,dict(table=table_meta,receptacle=destination_meta,**cart_meta)


def exclude_initial_categories(base,categories):
    """Make the two task nouns unambiguous at initialization, retaining edits.

    Only runtime actors of these exact asset categories are removed; official
    files and non-target clutter are untouched. This is not an open-set test.
    """
    pattern=re.compile(r"(?:^|/)(?:"+"|".join(re.escape(c) for c in categories)+r")_[0-9]+_[0-9]+$")
    removed=[]
    for name,actor in list(base.scene.actors.items()):
        if pattern.search(name):
            removed.append(dict(name=name,initial_pose=array(actor.pose.raw_pose).tolist()))
            for entity in actor._objs:
                base.scene.sub_scenes[0].remove_entity(getattr(entity,"entity",entity))
            base.scene.actors.pop(name)
    return removed
