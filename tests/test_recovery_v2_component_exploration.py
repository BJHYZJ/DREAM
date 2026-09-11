import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'controllers/recovery_v2/experiments'


def pilot(route_found=True, lose_target=False):
    tree=ast.parse((ROOT/'instruction_heading_navigation.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InstructionHeadingNavigation')
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='try_exploration_expansion']
    state=dict(planners=0,executed=[],route_candidates=[])
    union=np.ones((31,31),bool);safe=union.copy();safe[:,15:]=False
    def planner(occupancy,footprint,base,allow_reverse):
        assert allow_reverse is False
        state['planners']+=1
        return SimpleNamespace(pose_masks=union[None].copy())
    def distances(mask,start):
        rr,cc=np.indices(mask.shape)
        return np.where(mask,abs(rr-start[0])+abs(cc-start[1]),np.inf),None
    def route(planner,candidates,base,**kw):
        state['route_candidates']=candidates
        return ([[0,0,0],[2,0,0]] if route_found and candidates else []),[]
    namespace=dict(np=np,FetchHeadingAStar=planner,weighted_distances=distances,candidate_route=route)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(ROOT),'exec'),namespace)
    p=namespace['InstructionHeadingNavigation']()
    p.heading_navigation=True;p.task_stage='placement_search';p.cached=None;p.exploration_goal=None;p.search_anchor=None
    p.search_region_center=None;p.regional_frontier_queries=0;p.regional_frontier_limit=48
    p.frontier_visits=np.full((31,31),2);p.io=SimpleNamespace(step_id=1000,pose=lambda:np.zeros(3))
    p.occupancy=SimpleNamespace(resolution=.1,known=np.ones((31,31)))
    p.measured_navigation_footprint=lambda:dict(vertices_base_xy=[[-.3,-.2],[.3,-.2],[.3,.2],[-.3,.2]],padding_m=.04)
    p.event=lambda *a,**k:None
    p.pick_frontier=lambda start,distances,safe_override:tuple(np.argwhere(safe_override)[0]) if safe_override.any() else None
    def execute(base,route,exploring):
        state['executed'].append(route);p.exploration_goal=None if lose_target else np.array([2.,0.]);return True
    p.execute_heading_route=execute
    p.try_heading_navigation=lambda reason:state.setdefault('continued',reason) and True
    return p,safe,distances(safe,(15,5))[0],state


@pytest.mark.parametrize('change',[
    dict(heading_navigation=False),dict(task_stage='pickup_search'),dict(cached=np.ones(3)),
    dict(exploration_goal=np.ones(2)),dict(search_anchor=np.ones(2)),dict(search_region_center=np.ones(2)),
    dict(heading_expansion_attempts=4),dict(heading_expansion_last_step=900),dict(frontier_visits=np.zeros((31,31))),
])
def test_expansion_does_not_interrupt_a_target_or_first_search(change):
    p,safe,distances,state=pilot()
    p.__dict__.update(change)
    assert p.try_exploration_expansion((15,5),distances,safe) is None
    assert state['planners']==0 and not state['executed']


def test_only_validated_route_beyond_circle_component_can_execute():
    p,safe,distances,state=pilot();before=safe.copy();known=p.occupancy.known.copy()
    assert p.try_exploration_expansion((15,5),distances,safe) is True
    assert len(state['executed'])==1
    assert all(cell[1]>=20 for cell,_ in state['route_candidates'])
    assert np.array_equal(safe,before) and np.array_equal(p.occupancy.known,known)
    assert p.heading_expansion_goal==pytest.approx([2.,0.])


def test_no_validated_route_leaves_normal_search_available():
    p,safe,distances,state=pilot(route_found=False)
    assert p.try_exploration_expansion((15,5),distances,safe) is None
    assert not state['executed']
    assert p.heading_expansion_attempts==1


def test_expansion_goal_continues_across_circle_component_boundary():
    p,safe,distances,state=pilot();p.heading_expansion_goal=np.array([2.,0.]);p.exploration_goal=np.array([2.,0.])
    assert p.try_exploration_expansion((15,5),distances,safe) is True
    assert state['continued']=='continue_observed_component_expansion'
    assert state['planners']==0


def test_fresh_target_can_clear_the_expansion_during_execution():
    p,safe,distances,state=pilot(lose_target=True)
    assert p.try_exploration_expansion((15,5),distances,safe) is True
    assert p.heading_expansion_goal is None


def test_fresh_target_supersedes_a_pending_expansion():
    p,safe,distances,state=pilot();p.heading_expansion_goal=np.array([2.,0.]);p.exploration_goal=np.array([2.,0.]);p.cached=np.ones(3)
    assert p.try_exploration_expansion((15,5),distances,safe) is None
    assert p.heading_expansion_goal is None and state['planners']==0


@pytest.mark.parametrize('active', [None, [.8, 0.]])
def test_new_navigation_hook_preserves_ordinary_docking_and_detours(monkeypatch, active):
    import test_recovery_v2_search_progress
    monkeypatch.setattr(test_recovery_v2_search_progress, 'ROOT', ROOT)
    p=test_recovery_v2_search_progress.navigation(active)
    p.navigate(budget=1)
    if active is None:assert p.destinations[0][0]<-1
    else:assert p.destinations[0]==pytest.approx(active)
