import json
from pathlib import Path

import pytest

from dream_sim.study_report import analyze, sha


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def declared_batch(root: Path, success):
    jobs=[dict(name=f'test_attempt_{i:02d}',scene=f'test_house_{i//6}',seed=i%3,
               variant='dynamic' if i%2 else 'static',task='frozen_tasks/task.json') for i in range(60)]
    write(root/'frozen_tasks/task.json',{'test_fixture':True})
    (root/'frozen_tasks/evaluator_room_map.npz').write_bytes(b'test map bytes')
    for job in jobs:
        job.update(task_sha256=sha(root/job['task']),room_map_sha256=sha(root/'frozen_tasks/evaluator_room_map.npz'))
    protocol=dict(planned_attempts=60,attempts=jobs,source_sha256={})
    write(root/'protocol.json',protocol)
    first=root/jobs[0]['name']
    write(first/'result.json',dict(evaluator_task_success=success,source_files_unchanged=True))
    write(first/'source_hashes_before.json',{})
    write(first/'source_hashes_after.json',{})
    records=[dict(name=job['name'],task_success=success,result_sha256=sha(first/'result.json'),
                  source_verified_before=True,source_verified_after=True,
                  task_verified_before=True,task_verified_after=True,status='completed') for job in jobs]
    write(root/'batch_result.json',dict(all_planned_attempts_recorded=True,attempts=records,
                                      protocol_sha256=sha(root/'protocol.json')))
    return root


def test_unfinished_comparison_does_not_write_statistics(tmp_path):
    with pytest.raises(RuntimeError,match='has not finished'):
        analyze(tmp_path/'run',tmp_path/'audits',tmp_path/'summary')
    assert not (tmp_path/'summary').exists()


def test_missing_outcome_is_not_imputed_as_a_failure(tmp_path):
    root=declared_batch(tmp_path/'run',None)
    with pytest.raises(ValueError,match='missing or inconsistent task outcome'):
        analyze(root,tmp_path/'audits',tmp_path/'summary')
    assert not (tmp_path/'summary').exists()


def test_changed_task_cannot_enter_a_comparison(tmp_path):
    root=declared_batch(tmp_path/'run',False)
    write(root/'frozen_tasks/task.json',{'changed':True})
    with pytest.raises(ValueError,match='task input changed'):
        analyze(root,tmp_path/'audits',tmp_path/'summary')


def test_reported_success_needs_an_independent_replay_record(tmp_path):
    root=declared_batch(tmp_path/'run',True)
    with pytest.raises(FileNotFoundError):
        analyze(root,tmp_path/'audits',tmp_path/'summary')
    assert not (tmp_path/'summary').exists()
