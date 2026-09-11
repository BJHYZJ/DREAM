import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1] / 'controllers/recovery_v2/experiments'


def extract_class(path, class_name, methods, base=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node.bases = [ast.Name(id='Base', ctx=ast.Load())] if base else []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    namespace = dict(np=np, asdict=asdict, Base=base)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), path, 'exec'), namespace)
    return namespace[class_name]


@dataclass(frozen=True)
class Detection:
    observation_id: int
    query: str = 'plate'
    point_world: tuple = (1., 2., .8)


def memory(dynamic):
    tree = ast.parse((ROOT / 'dream_learned_core.py').read_text())
    name = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
                and any(isinstance(f, ast.FunctionDef) and f.name == 'detection_rejected' for f in n.body))
    cls = extract_class('dream_learned_core.py', name, ['reject', 'detection_rejected', 'rejections_for'])
    obj = cls()
    obj.dynamic = dynamic
    obj.rejected_regions = []
    obj.rejected_observations = set()
    obj.query_rejections = {}
    obj.rejection_version = 0
    obj.observations = {i: SimpleNamespace(frame_id=i) for i in (10, 20)}
    return obj


def pilot(dynamic, previous=Detection(10)):
    class Base:
        def retrieve(self, use_live=False):
            self.cached_detection = self.retrieved
            self.cached = np.asarray(self.retrieved.point_world)
            return self.retrieved

    cls = extract_class('instruction_policy.py', 'InstructionSearchPilot', ['retrieve'], Base)
    obj = cls()
    obj.task_stage = 'placement_search'
    obj.query = 'plate'
    obj.cached_detection = previous
    obj.cached = None if previous is None else np.asarray(previous.point_world)
    obj.retrieved = Detection(20)
    obj.memory = memory(dynamic)
    obj._support_for = lambda obs, det: (obs.frame_id == 10, None)
    obj.event = lambda *args, **kwargs: None
    return obj


@pytest.mark.parametrize('dynamic', [True, False])
def test_ungrounded_image_preserves_another_supported_view_at_same_position(dynamic):
    obj = pilot(dynamic)
    assert obj.retrieve() == Detection(10)
    assert obj.cached_detection == Detection(10)
    assert np.array_equal(obj.cached, [1., 2., .8])
    assert not obj.memory.detection_rejected(Detection(10))
    assert obj.memory.detection_rejected(Detection(20)) is dynamic
    assert not obj.memory.detection_rejected(Detection(20, query='mug'))
    assert not obj.memory.rejections_for('plate')[0]


@pytest.mark.parametrize('dynamic', [True, False])
def test_rejected_view_alone_does_not_create_a_destination(dynamic):
    obj = pilot(dynamic, previous=None)
    assert obj.retrieve() is None
    assert obj.cached is None and obj.cached_detection is None


def test_free_depth_rejection_is_not_resurrected_by_view_fallback():
    obj = pilot(True)
    obj.memory.reject([1., 2.], 10, observation_id=10, query='plate')
    assert obj.retrieve() is None
    assert obj.memory.detection_rejected(Detection(10))


@pytest.mark.parametrize('dynamic', [True, False])
def test_old_image_must_still_ground_the_support(dynamic):
    obj = pilot(dynamic)
    obj._support_for = lambda obs, det: (False, None)
    assert obj.retrieve() is None


@pytest.mark.parametrize('dynamic', [True, False])
def test_fresh_supported_retrieval_supersedes_the_prior(dynamic):
    obj = pilot(dynamic)
    obj._support_for = lambda obs, det: (True, None)
    assert obj.retrieve() == Detection(20)
    assert obj.cached_detection == Detection(20)


def test_existing_spatial_rejection_still_invalidates_old_nearby_detections():
    obj = memory(True)
    obj.reject([1., 2.], 20, query='plate')
    assert obj.detection_rejected(Detection(10))
    assert obj.detection_rejected(Detection(20))
    assert not obj.detection_rejected(Detection(21))
    assert not obj.detection_rejected(Detection(10, query='mug'))


def test_static_memory_keeps_its_existing_rejection_behavior():
    obj = memory(False)
    obj.reject([1., 2.], 20, radius=None, observation_id=20, query='plate')
    assert obj.rejection_version == 0 and obj.query_rejections == {}
