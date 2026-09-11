import ast
from dataclasses import asdict
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'controllers/recovery_v2/experiments'


def method_class(filename, class_name, methods, bases, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    node.bases = [ast.Name(id=name, ctx=ast.Load()) for name in bases]
    namespace.update(np=np, math=math, asdict=asdict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), filename, 'exec'), namespace)
    return namespace[class_name]


def navigation(active):
    def distances(safe, start):
        rr, cc = np.indices(safe.shape)
        return abs(rr-start[0]) + abs(cc-start[1]), None
    cls = method_class('maniskill_crossroom_policy.py', 'CrossRoomSearchPilot',
                       ['navigate', 'docking_limits', 'try_exploration_expansion'], [], {'weighted_distances': distances})
    pilot = cls()
    pose = np.zeros(3)
    safe = np.ones((31, 31), bool)
    pilot.io = SimpleNamespace(pose=lambda: pose.copy())
    pilot.occupancy = SimpleNamespace(resolution=.2, traversable=lambda: safe,
        cells=lambda xy: np.rint((np.asarray(xy)+3)/.2).astype(int)[..., ::-1],
        world=lambda cells: np.asarray(cells)[..., ::-1]*.2-3)
    pilot.exploration_goal = None if active is None else np.asarray(active, dtype=float)
    pilot.search_anchor = None
    pilot.cached = np.array([-2., 0., .9])
    pilot.current_detection = None
    pilot.frontier_visits = np.zeros_like(safe, dtype=int)
    pilot.event = lambda *a, **k: None
    pilot.compact_navigation_arm = lambda: None
    pilot.retrieve = lambda **k: None
    pilot.geometry_observe = lambda: None
    pilot.scan_sweep = lambda: None
    pilot.look_around = lambda: None
    pilot.planner = SimpleNamespace(path=lambda start, goal: [np.asarray(goal)])
    pilot.destinations = []
    def move(path):
        pilot.destinations.append(path[-1].copy())
        pose[:2] = path[-1]
        return True
    pilot.move_chunk = move
    return pilot


def test_active_detour_is_executed_despite_a_remembered_dock():
    pilot = navigation([.8, 0])
    pilot.navigate(budget=1)
    assert pilot.destinations[0] == pytest.approx([.8, 0])
    assert pilot.frontier_visits.max() == 0  # Planning alone does not count as a visit.
    pilot.navigate(budget=1)
    assert pilot.frontier_visits.max() == 1  # Arrival does count.


def test_without_detour_a_remembered_target_keeps_normal_docking():
    pilot = navigation(None)
    pilot.navigate(budget=1)
    assert pilot.destinations[0][0] < -1


@pytest.mark.parametrize('used, expected_cells', [(47, 1), (48, 2), (60, 2)])
def test_fallback_observation_points_exit_local_prior(monkeypatch, used, expected_cells):
    mask = np.array([[True, False], [False, True]])
    monkeypatch.setitem(sys.modules, 'instruction_observation_frontier',
                        SimpleNamespace(observation_standoffs=lambda *a: mask.copy()))
    seen = []
    def choose(age, semantic, frontier, start, **kwargs):
        seen.append(frontier.copy())
        return SimpleNamespace(index=tuple(np.argwhere(frontier)[0]))
    monkeypatch.setitem(sys.modules, 'dream_learned_core', SimpleNamespace(select_frontier_goal=choose))
    class Parent:
        def pick_frontier(self, *args, **kwargs):
            if self.regional_frontier_queries < self.regional_frontier_limit:
                self.regional_frontier_queries += 1
            return None
    cls = method_class('instruction_policy.py', 'InstructionSearchPilot', ['pick_frontier'],
                       ['Parent'], {'Parent': Parent})
    pilot = cls()
    pilot.search_region_center = np.zeros(2)
    pilot.regional_frontier_queries = used
    pilot.regional_frontier_limit = 48
    pilot.frontier_visits = np.zeros((2, 2))
    pilot.memory = None
    pilot.query = 'object'
    pilot.event = lambda *a, **k: None
    pilot.occupancy = SimpleNamespace(known=np.ones((2, 2)), last_seen=np.zeros((2, 2)),
        radius=.4, resolution=.04, traversable=lambda: np.ones((2, 2), bool),
        world=lambda cells: np.asarray(cells)*5., semantic_field=lambda *a: np.zeros((2, 2)))
    pilot.pick_frontier((0, 0), np.ones((2, 2)))
    assert seen[0].sum() == expected_cells
