"""Check release height using the observed object center and grasp offset."""
from pathlib import Path
import ast,json,math
from types import SimpleNamespace
import numpy as np
source=Path(__file__).resolve().parents[1]/'controllers/compact_v1/experiments/instruction_policy.py';tree=ast.parse(source.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InstructionSearchPilot');cls.bases=[]
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['planned_release_tcp','align_observed_release','wait_for_release_stillness']]
namespace=dict(np=np,math=math,array=np.asarray);exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),namespace)


def test_offset_release_command_and_height_verification():
    p=namespace['InstructionSearchPilot']();p.instruction=SimpleNamespace(placement_relation='on');p.perceived_payload_frame=dict(center_tcp_m=[.06,.005,0.],axis_tcp=[-1,0,0]);p.perceived_payload_radius=.02
    rotation=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]);release=np.array([1.,2.,1.])
    goal,offset=p.planned_release_tcp(release,rotation);assert np.allclose(goal+rotation@offset,release);assert abs(goal[2]-1.06)<1e-9
    matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=goal
    pose=SimpleNamespace(p=matrix[None,:3,3],to_transformation_matrix=lambda:matrix[None]);p.io=SimpleNamespace(robot=SimpleNamespace(tcp_pose=pose),base=SimpleNamespace(control_freq=20));p.hold_release_pose_step=lambda *a:None;p.require_retained_payload=lambda:None;p.event=lambda *a,**k:None
    assert p.wait_for_release_stillness(goal,rotation,release,dict(observed_safe_radius_m=.10),offset)
    matrix[2,3]-=.05
    assert not p.wait_for_release_stillness(goal,rotation,release,dict(observed_safe_radius_m=.10),offset)
