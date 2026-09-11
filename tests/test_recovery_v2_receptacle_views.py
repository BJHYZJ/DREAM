import ast
from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'controllers/recovery_v2/experiments'


def function(name):
    tree=ast.parse((ROOT/'instruction_geometry.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    from dataclasses import replace
    namespace={'np':np,'replace':replace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'geometry','exec'),namespace)
    return namespace[name]


@dataclass(frozen=True)
class Observation:
    rgb:np.ndarray
    depth_m:np.ndarray
    intrinsics:np.ndarray
    camera_to_world_cv:np.ndarray
    frame_id:int=17
    sim_step:int=450
    sensor:str='fetch_head'


def observation():
    rgb=np.arange(480*640*3,dtype=np.uint8).reshape(480,640,3)
    depth=np.full((480,640),1.2)
    return Observation(rgb,depth,np.array([[400.,0,320],[0,400.,240],[0,0,1.]]),np.eye(4))


def world_points(obs):
    yy,xx=np.indices(obs.depth_m.shape)
    camera=np.stack([xx,yy,np.ones_like(xx)],axis=-1)@np.linalg.inv(obs.intrinsics).T*obs.depth_m[...,None]
    return camera@obs.camera_to_world_cv[:3,:3].T+obs.camera_to_world_cv[:3,3]


def test_crop_preserves_recorded_pixels_and_world_calibration():
    obs=observation();crop,(x,y,x1,y1)=function('focused_rgbd_crop')(obs,np.array([.2,.1,1.2]))
    assert np.array_equal(crop.rgb,obs.rgb[y:y1,x:x1])
    assert np.array_equal(crop.depth_m,obs.depth_m[y:y1,x:x1])
    assert np.allclose(world_points(crop),world_points(obs)[y:y1,x:x1],atol=1e-12)
    assert (crop.frame_id,crop.sim_step,crop.sensor)==(obs.frame_id,obs.sim_step,obs.sensor)


@pytest.mark.parametrize('point',[[0,0,-1],[0,0,5],[8,0,1],[0,float('nan'),1]])
def test_crop_rejects_unprojectable_visual_prior(point):
    assert function('focused_rgbd_crop')(observation(),point) is None


def method_class(names,bases=None):
    tree=ast.parse((ROOT/'instruction_policy.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef));node.bases=bases or []
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in names]
    from dataclasses import replace,asdict
    namespace={'np':np,'math':math,'array':np.asarray,'replace':replace,'asdict':asdict,
               'focused_rgbd_crop':function('focused_rgbd_crop'),
               'receptacle_region':lambda *a:None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'policy','exec'),namespace)
    return namespace['InstructionSearchPilot']


@dataclass(frozen=True)
class Detection:
    observation_id:int
    query:str
    score:float
    box_xyxy:list
    point_world:list


def test_focused_detection_restores_full_frame_box_and_keeps_query_and_score():
    obj=method_class(['focused_receptacle_detection'])();obs=observation();seen=[]
    detection=Detection(obs.frame_id,'plate',.3,[10,20,60,80],[.1,.2,1.2])
    def detect(crop,query):seen.append((crop,query));return [detection]
    obj.perception=SimpleNamespace(detect=detect);obj.instruction=SimpleNamespace(placement_query='plate')
    obj.event=lambda *a,**k:None
    prior=np.array([.2,.1,1.2]);crop,bounds=function('focused_rgbd_crop')(obs,prior)
    actual=obj.focused_receptacle_detection(obs,prior);x,y,_,_=bounds
    assert actual.box_xyxy==[10+x,20+y,60+x,80+y]
    assert actual.point_world==detection.point_world and actual.score==.3
    assert actual.observation_id==obs.frame_id and seen[0][1]=='plate'


def extension_pilot(monkeypatch,results):
    obj=method_class(['alternate_receptacle_views'])();matrix=np.eye(4);matrix[:3,3]=[.28,0.,.84]
    commands=[];modes=[];retained=[];events=[]
    io=SimpleNamespace(robot=SimpleNamespace(tcp_pose=SimpleNamespace(to_transformation_matrix=lambda:matrix[None],p=matrix[None,:3,3]),
        set_control_mode=modes.append,controller=SimpleNamespace(reset=lambda:None)),body=np.zeros(3),pose=lambda:np.zeros(3))
    io.hold_measured_arm=lambda:modes.append('pd_joint_pos')
    obj.io=io;obj.occupancy=SimpleNamespace(radius=.45);obj.perceived_payload_radius=.1
    obj.perceived_payload_height=.06;obj.last_obs=SimpleNamespace(frame_id=17)
    obj.instruction=SimpleNamespace(placement_relation='on')
    obj.require_retained_payload=lambda:retained.append(True)
    def move(goal,rotation):commands.append((goal.copy(),rotation.copy()));matrix[:3,3]=goal
    obj.hold_release_pose_step=move;obj.focus_camera_height=lambda:1.3
    outputs=iter(results);obj.observe=lambda:next(outputs);obj.event=lambda name,**kw:events.append((name,kw))
    monkeypatch.setitem(sys.modules,'inspect_fetch_envelope',SimpleNamespace(measure=lambda io:{'max_radius':.50}))
    return obj,commands,modes,retained,events


def test_view_extension_is_bounded_retains_orientation_and_updates_navigation_radius(monkeypatch):
    found=object();obj,commands,modes,retained,events=extension_pilot(monkeypatch,[None,found])
    assert obj.alternate_receptacle_views(np.array([.7,0.,.8])) is found
    assert np.allclose([goal[0] for goal,_ in commands],[.40,.52])
    assert all(np.allclose(rotation,np.eye(3)) for _,rotation in commands)
    assert all(goal[2]==.84 for goal,_ in commands)
    assert obj.occupancy.radius>=.66-1e-12 and len(retained)>=4
    assert modes[-1]=='pd_joint_pos' and obj.receptacle_view_prior is None
    assert obj.alternate_receptacle_views(np.array([.7,0.,.8])) is None
    assert len(commands)==2


def test_view_extension_restores_joint_hold_if_grasp_is_lost(monkeypatch):
    obj,commands,modes,retained,events=extension_pilot(monkeypatch,[])
    calls=[]
    def retained_check():
        calls.append(True)
        if len(calls)==2:raise RuntimeError('empty gripper')
    obj.require_retained_payload=retained_check
    with pytest.raises(RuntimeError,match='empty gripper'):
        obj.alternate_receptacle_views(np.array([.7,0.,.8]))
    assert modes[-1]=='pd_joint_pos' and obj.receptacle_view_prior is None
    assert not commands


@pytest.mark.parametrize('supported',[False,True])
def test_focused_detections_still_require_normal_support_grounding(supported):
    from dataclasses import asdict
    class Parent:
        def observe(self,sensor):
            self.last_obs=observation();self.current_detection=None
            return None
    tree=ast.parse((ROOT/'instruction_policy.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    node.bases=[ast.Name(id='Parent',ctx=ast.Load())]
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='observe']
    namespace={'Parent':Parent,'np':np,'asdict':asdict,'motion_detection_changed':lambda *a:False}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'policy','exec'),namespace)
    obj=namespace['InstructionSearchPilot']();obj.task_stage='placement_search'
    obj.cached=np.array([.7,0.,.8]);obj.cached_detection=None
    obj.io=SimpleNamespace(pose=lambda:np.zeros(3));obj.require_retained_payload=lambda:None
    obj.settle_head_view=lambda:None;obj.event=lambda *a,**k:None
    found=Detection(17,'plate',.3,[20,30,60,80],[.7,.01,.8])
    obj.focused_receptacle_detection=lambda *a:found
    obj._support_for=lambda *a:(supported,None);observed=[]
    obj.observation_listener=lambda obs,d,stage:observed.append(d)
    actual=obj.observe()
    assert actual is (found if supported else None)
    assert obj.current_detection is actual and observed==[actual]
    assert np.allclose(obj.cached,found.point_world if supported else [.7,0.,.8])


def test_visible_but_too_small_receptacle_view_does_not_end_recovery(monkeypatch):
    found=object();obj,commands,modes,retained,events=extension_pilot(monkeypatch,[found,found])
    fits=[]
    def fit(*args):
        fits.append(True)
        if len(fits)==1:raise ValueError('insufficient observed clearance')
    monkeypatch.setitem(obj.alternate_receptacle_views.__func__.__globals__,'receptacle_region',fit)
    assert obj.alternate_receptacle_views(np.array([.7,0.,.8])) is found
    assert len(commands)==2
    assert any(name=='receptacle_extension_view_incomplete' for name,_ in events)
