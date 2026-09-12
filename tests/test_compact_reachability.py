"""A connected position map need not permit the robot's turn/translation edges."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'controllers/compact_v1/experiments/dream_fetch_heading_astar.py'


def corridor():
    tree=ast.parse(SOURCE.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FetchHeadingAStar')
    node.bases=[]
    node.body=[n for n in node.body if isinstance(n,ast.FunctionDef) and n.name in
               {'reachable_pose_masks','edge_is_valid','point_is_occupied'}]
    namespace=dict(np=np,math=math,navigation_swept_clear=lambda occ,fp,start,end,views:abs(end[2])<1e-8)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),namespace)
    p=namespace['FetchHeadingAStar']()
    p.pose_masks=np.ones((8,3,5),bool)
    p.forward_masks=np.zeros_like(p.pose_masks);p.forward_masks[0,1,0]=True
    p.turn_ccw_masks=np.zeros_like(p.pose_masks)
    p.steps=[(round(math.sin(h*math.pi/4)),round(math.cos(h*math.pi/4))) for h in range(8)]
    p.allow_reverse=False;p.start_pose=np.zeros(3);p.to_pt=lambda pose:(1,0,0)
    p.occupancy=SimpleNamespace(known=np.ones((3,5),np.int8));p.footprint={};p.layer_views=[]
    return p


def test_position_connectivity_does_not_cross_a_blocked_swept_edge():
    p=corridor();before=p.pose_masks.copy()
    reached=p.reachable_pose_masks()
    assert reached.sum()==2
    assert reached[0,1,0] and reached[0,1,1]
    assert not reached[:,:,2:].any()
    assert np.array_equal(p.pose_masks,before)


def test_search_budget_returns_only_proven_reachable_states():
    p=corridor();reached=p.reachable_pose_masks(maximum_states=1)
    assert reached.sum()==1 and reached[0,1,0]


def test_no_initial_swept_turn_leaves_no_reachable_candidate():
    p=corridor();p.reachable_pose_masks.__func__.__globals__['navigation_swept_clear']=lambda *a,**k:False
    assert not p.reachable_pose_masks().any()
