import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'controllers/recovery_v2/experiments'


def pilot(positions):
    tree=ast.parse((ROOT/'instruction_policy.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef));node.bases=[]
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='wait_for_release_stillness']
    namespace={'np':np,'array':np.asarray}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'policy','exec'),namespace)
    obj=namespace['InstructionSearchPilot']();matrix=np.eye(4);matrix[:3,3]=positions[0]
    obj.io=SimpleNamespace(robot=SimpleNamespace(tcp_pose=SimpleNamespace(p=matrix[None,:3,3],to_transformation_matrix=lambda:matrix[None])),base=SimpleNamespace(control_freq=20))
    obj.steps=0;obj.perceived_payload_radius=.05;obj.events=[]
    def control(*args):matrix[:3,3]=positions[min(obj.steps,len(positions)-1)];obj.steps+=1
    obj.hold_release_pose_step=control;obj.require_retained_payload=lambda:None
    obj.event=lambda name,**kw:obj.events.append((name,kw))
    return obj


def wait(obj,offset=np.zeros(3)):
    release=np.array([.7,0.,.9])
    return obj.wait_for_release_stillness(release-offset,np.eye(3),release=release,
        geometry={'observed_safe_radius_m':.06},offset=offset)


def test_stable_but_outside_release_height_does_not_end_wait_early():
    obj=pilot([[.7,0.,.927]])
    assert not wait(obj)
    assert obj.steps==40


def test_slow_convergence_uses_available_time_without_changing_height_tolerance():
    obj=pilot([[.7,0.,height] for height in [.927,.926,.925,.924,.923,.923,.923,.923]])
    assert wait(obj)
    assert 3<obj.steps<=40
    assert obj.events[-1][0]=='release_pose_settled'


def test_horizontal_guard_uses_the_observed_grasp_offset():
    obj=pilot([[.68,0.,.9]])
    assert wait(obj,np.array([.02,0.,0.]))
    assert obj.steps==3
    unaligned=pilot([[.68,0.,.9]])
    assert not wait(unaligned)
    assert unaligned.steps==40


def test_existing_stillness_only_call_keeps_its_original_behavior():
    obj=pilot([[.7,0.,.927]])
    assert obj.wait_for_release_stillness(np.array([.7,0.,.9]),np.eye(3))
    assert obj.steps==3
