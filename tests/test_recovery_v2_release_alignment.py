import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'controllers/recovery_v2/experiments'
spec=importlib.util.spec_from_file_location('release_geometry',ROOT/'instruction_geometry.py')
geometry=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=geometry
spec.loader.exec_module(geometry)


def template():
    rng=np.random.default_rng(713)
    points=rng.normal(size=(700,3))
    points/=np.linalg.norm(points,axis=1)[:,None]
    points*=np.array([.028,.025,.022])
    colors=np.c_[rng.uniform(20,65,700),rng.uniform(15,65,700),rng.uniform(10,55,700)]
    return dict(points_tcp=points,colors_lab=colors,center_tcp=np.zeros(3),source_observation_id=np.asarray(1))


def test_partial_observed_surface_recovers_small_shift_in_rotated_tool_frame():
    model=template()
    pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[2.,3.,1.]
    ids=np.flatnonzero(model['points_tcp'][:,0]>0)
    shift=np.array([.002,-.001,.0015])
    live=model['points_tcp'][ids]@pose[:3,:3].T+pose[:3,3]+shift
    offset,fit=geometry.register_payload_points(model,live,model['colors_lab'][ids],pose)
    assert pose[:3,:3]@offset == pytest.approx(shift,abs=.001)
    assert fit['inliers']>=48
    assert fit['rms_m']<.004


def test_unrelated_appearance_cannot_supply_an_offset():
    model=template()
    with pytest.raises(ValueError,match='appearance'):
        geometry.register_payload_points(model,model['points_tcp'],np.tile([90,-90,-90],(700,1)),np.eye(4))


def test_invisible_or_remote_payload_does_not_get_a_correction():
    model=template()
    with pytest.raises(ValueError,match='Insufficient'):
        geometry.register_payload_points(model,model['points_tcp'][:20],model['colors_lab'][:20],np.eye(4))
    with pytest.raises(ValueError,match='Insufficient'):
        geometry.register_payload_points(model,model['points_tcp']+2,model['colors_lab'],np.eye(4))


def test_tiny_visible_patch_cannot_determine_object_center():
    model=template();model['points_tcp']*=.01
    with pytest.raises(ValueError,match='patch is too small'):
        geometry.register_payload_points(model,model['points_tcp'],model['colors_lab'],np.eye(4))


def stillness_pilot(positions):
    tree=ast.parse((ROOT/'instruction_policy.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    node.bases=[]
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='wait_for_release_stillness']
    namespace={'np':np,'array':np.asarray}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'policy','exec'),namespace)
    pilot=namespace['InstructionSearchPilot']()
    tcp=SimpleNamespace(p=np.array([[0.,0.,0.]]))
    pilot.io=SimpleNamespace(base=SimpleNamespace(control_freq=20),robot=SimpleNamespace(tcp_pose=tcp))
    pilot.calls=0;pilot.events=[]
    def hold(*args):
        tcp.p=np.array([positions[min(pilot.calls,len(positions)-1)]])
        pilot.calls+=1
    pilot.hold_release_pose_step=hold
    pilot.require_retained_payload=lambda:None
    pilot.event=lambda name,**kw:pilot.events.append(dict(event=name,**kw))
    return pilot


def test_release_waits_for_three_consecutive_stationary_measurements():
    pilot=stillness_pilot([[.003,0,0],[.004,0,0],[.004,0,0],[.004,0,0],[.004,0,0]])
    assert pilot.wait_for_release_stillness(np.zeros(3),np.eye(3))
    assert pilot.calls==5


def test_continuous_drift_cannot_pass_the_release_wait():
    pilot=stillness_pilot([[.003*i,0,0] for i in range(1,41)])
    assert not pilot.wait_for_release_stillness(np.zeros(3),np.eye(3))
    assert pilot.calls==40
