"""Transport the observed payload bound with the grasp and reject support leakage."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'controllers/compact_v1/experiments'


def load_function(file,name,**bindings):
    tree=ast.parse((ROOT/file).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=dict(np=np,math=math,cv2=cv2,**bindings)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/file),'exec'),namespace)
    return namespace[name]


def test_payload_bound_retains_the_observed_grasp_offset():
    vertices=load_function('dream_fetch_footprint.py','observed_payload_vertices')
    frame=dict(center_tcp_m=[.06,0,0],axis_tcp=[-1,0,0])
    points=vertices(np.eye(4),.03,.20,frame)
    assert points[:,0].min()==pytest.approx(-.06)
    assert points[:,0].max()==pytest.approx(.18)
    assert np.max(np.abs(points[:,1:]))<.051


def test_payload_bound_rotates_with_the_measured_tool_pose():
    vertices=load_function('dream_fetch_footprint.py','observed_payload_vertices')
    frame=dict(center_tcp_m=[.06,0,0],axis_tcp=[-1,0,0])
    transform=np.eye(4);transform[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];transform[:3,3]=[2,3,1]
    original=vertices(np.eye(4),.03,.20,frame)
    rotated=vertices(transform,.03,.20,frame)
    assert rotated.mean(0)==pytest.approx(transform[:3,3]+transform[:3,:3]@original.mean(0))
    assert rotated[:,1].min()==pytest.approx(2.94)
    assert rotated[:,1].max()==pytest.approx(3.18)


def test_payload_bound_rejects_an_invalid_axis():
    vertices=load_function('dream_fetch_footprint.py','observed_payload_vertices')
    with pytest.raises(ValueError,match='Invalid observed payload frame'):
        vertices(np.eye(4),.03,.20,dict(center_tcp_m=[0,0,0],axis_tcp=[0,0,0]))


def test_support_fragment_requests_a_new_view_instead_of_shrinking_the_object():
    angle=np.linspace(0,2*np.pi,64,endpoint=False)
    points=np.vstack([np.c_[.03*np.cos(angle),.03*np.sin(angle),np.full(64,z)] for z in np.linspace(.91,1.10,12)])
    shape=SimpleNamespace(points=points,support_height_m=.90,support_points=100,observation_id=1)
    grasp=load_function('instruction_geometry.py','tabletop_grasp',observed_shape=lambda *a,**k:shape)
    _,fit=grasp(SimpleNamespace(frame_id=1),None)
    assert fit['radius_m']<.04
    shape.points=np.vstack([points,[.25,0,.925]])
    with pytest.raises(ValueError,match='acquire another view'):
        grasp(SimpleNamespace(frame_id=1),None)
