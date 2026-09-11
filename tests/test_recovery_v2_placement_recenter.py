import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT=Path(__file__).resolve().parents[1]/'controllers/recovery_v2/experiments'


def pilot(monkeypatch, *, blocked=False):
    tree=ast.parse((ROOT/'instruction_policy.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    node.bases=[]
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef)
               and n.name=='align_base_before_placement']
    ns=dict(np=np,math=math)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT),'exec'),ns)
    obj=ns['InstructionSearchPilot']()
    obj.heading_navigation=True
    obj.instruction=SimpleNamespace(placement_relation='on')
    obj.perceived_payload_radius=.10;obj.perceived_payload_height=.09
    obj.position=np.array([0.,0.,0.]);obj.io=SimpleNamespace(pose=lambda:obj.position.copy())
    obj.occupancy=object();obj.events=[];obj.checks=0;obj.captures=[];obj.moves=[];obj.observations=0
    def retained():obj.checks+=1
    def observe():obj.observations+=1
    obj.require_retained_payload=retained;obj.geometry_observe=observe
    obj.event=lambda name,**data:obj.events.append(dict(event=name,**data))
    def footprint(io,**data):
        assert io is obj.io
        obj.captures.append(data)
        return data
    class Executor:
        def __init__(self,io,occupancy,footprint_provider,geometry_observe,event):
            assert occupancy is obj.occupancy  # Preserve all accumulated obstacle evidence.
            self.footprint=footprint_provider;self.observe=geometry_observe
            self.last_failure='fresh_depth_blocks_translation' if blocked else None
        def translate(self,goal,planned_heading):
            self.footprint();self.observe()
            if blocked:return False
            obj.moves.append((goal.copy(),planned_heading))
            obj.position[:2]=goal
            return True
    monkeypatch.setitem(sys.modules,'dream_fetch_footprint',SimpleNamespace(capture_robot_footprint=footprint))
    monkeypatch.setitem(sys.modules,'fetch_heading_control',SimpleNamespace(HeadingRouteExecutor=Executor))
    return obj


def align(obj, distance=.735, radius=.123):
    return obj.align_base_before_placement(np.array([distance,0.,.85]),{'observed_safe_radius_m':radius})


def test_alignment_includes_held_payload_and_existing_map(monkeypatch):
    obj=pilot(monkeypatch)
    assert align(obj)
    assert obj.captures==[dict(payload_radius=.10,payload_height=.09)]
    assert obj.checks==2 and obj.observations==1
    assert obj.moves[0][0]==pytest.approx([.045,0.])


def test_observed_collision_blocks_motion(monkeypatch):
    obj=pilot(monkeypatch,blocked=True)
    assert align(obj) is False
    assert not obj.moves
    assert obj.position==pytest.approx([0.,0.,0.])
    assert obj.events[-1]['last_failure']=='fresh_depth_blocks_translation'


def test_farther_receptacle_cannot_expand_the_five_centimetre_motion_limit(monkeypatch):
    obj=pilot(monkeypatch)
    assert align(obj,distance=.90)
    assert np.linalg.norm(obj.moves[0][0])==pytest.approx(.05)


@pytest.mark.parametrize('change',[
    dict(heading_navigation=False),dict(perceived_payload_radius=.08),
    dict(instruction=SimpleNamespace(placement_relation='in')),
    dict(position=np.array([0.,0.,math.pi/2])),
])
def test_inapplicable_pose_or_task_keeps_existing_placement(monkeypatch,change):
    obj=pilot(monkeypatch)
    for key,value in change.items():setattr(obj,key,value)
    assert align(obj) is None
    assert not obj.moves and not obj.captures


@pytest.mark.parametrize('distance,radius',[(.70,.123),(.735,.104),(.735,.16),(float('nan'),.123)])
def test_invalid_or_unnecessary_alignment_cannot_move_base(monkeypatch,distance,radius):
    obj=pilot(monkeypatch)
    assert align(obj,distance=distance,radius=radius) is None
    assert not obj.moves and not obj.captures
