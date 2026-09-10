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


def batch_with_contact_rejection(tmp_path):
    root=declared_batch(tmp_path/'run',True)
    protocol=json.loads((root/'protocol.json').read_text())
    batch=json.loads((root/'batch_result.json').read_text())
    for index,(job,record) in enumerate(zip(protocol['attempts'],batch['attempts'])):
        folder=root/job['name']
        write(folder/'result.json',dict(evaluator_task_success=index==0,source_files_unchanged=True))
        write(folder/'source_hashes_before.json',{})
        write(folder/'source_hashes_after.json',{})
        record.update(task_success=index==0,result_sha256=sha(folder/'result.json'))
    write(root/'batch_result.json',batch)
    name=protocol['attempts'][0]['name'];folder=root/name
    (folder/'events.jsonl').write_text('{"event":"synthetic_test_fixture"}\n')
    audits=tmp_path/'audits';physical_path=audits/name/'physical/audit.json'
    physical=dict(physical_reexecution_passed=True,source_recording_unchanged=True,
        replay_environment_source_matches_recording=True,replay_source_unchanged=True,
        evaluation={'evaluator_task_success':True},contact_audit={
            'native_environment_contact_control_steps':1,
            'native_environment_contact_rows':[{'control_step':10,'force_n':0.75}]})
    write(physical_path,physical)
    names='''source_hashes_match saved_source_matches_hashes instruction_first
        actual_control_reexecution replay_sources_verified replay_refers_to_this_run
        no_native_contact_above_threshold complete_aligned_traces
        identity_records_consistent_with_saved_masks_and_replayed_positions
        fresh_visual_discovery_at_standoff_during_translation continuous_1x_frame_steps
        all_source_videos_decode_completely matching_5fps_playback
        current_observation_panel_matches_raw_camera external_relocation_caption_visible
        annotated_video_retains_all_frames original_task_scoring reexecuted_task_success
        correct_visual_moved_target_observation all_logged_updates_have_observed_free_depth
        logged_updates_have_variant_correct_caption'''.split()
    checks=dict.fromkeys(names,True);checks['no_native_contact_above_threshold']=False
    review=dict(primary_task_checks=checks,primary_task_record_review_passed=False,
        physical_audit_sha256=sha(physical_path),events_sha256=sha(folder/'events.jsonl'))
    review_path=audits/name/'record/record_review.json';write(review_path,review)
    return root,audits,review_path


def test_contact_rejection_preserves_raw_outcome_and_full_denominator(tmp_path):
    root,audits,_=batch_with_contact_rejection(tmp_path)
    before=sha(root/'test_attempt_00/result.json')
    report=analyze(root,audits,tmp_path/'summary',include_contact_rejections=True)
    assert report['statistics']['variants']['static']['successes']==0
    assert report['reported_task_statistics']['variants']['static']['successes']==1
    assert all(row['attempts']==30 for row in report['statistics']['variants'].values())
    assert len(report['attempts'])==60 and all(row['eligible_for_summary'] for row in report['attempts'])
    assert report['attempts'][0]['task_success'] is True
    assert report['attempts'][0]['qualified_task_success'] is False
    assert report['all_counted_successes_passed_audits'] is True
    assert report['successful_task_physics_and_record_checks_passed'] is False
    assert report['audit_rejections']==[dict(name='test_attempt_00',reason='native_contact_rejected',native_contact_steps=1)]
    assert sha(root/'test_attempt_00/result.json')==before


def test_contact_rejection_requires_explicit_reporting_option(tmp_path):
    root,audits,_=batch_with_contact_rejection(tmp_path)
    with pytest.raises(ValueError,match='primary task record audit did not pass'):
        analyze(root,audits,tmp_path/'summary')
    assert not (tmp_path/'summary').exists()


@pytest.mark.parametrize('failure',['source_hashes_match','reexecuted_task_success',
                                   'identity_records_consistent_with_saved_masks_and_replayed_positions'])
def test_contact_reporting_never_swallows_other_audit_failures(tmp_path,failure):
    root,audits,path=batch_with_contact_rejection(tmp_path)
    review=json.loads(path.read_text());review['primary_task_checks'][failure]=False;write(path,review)
    with pytest.raises(ValueError,match='primary task record audit did not pass'):
        analyze(root,audits,tmp_path/'summary',include_contact_rejections=True)
    assert not (tmp_path/'summary').exists()


def test_contact_reporting_requires_the_complete_audit(tmp_path):
    root,audits,path=batch_with_contact_rejection(tmp_path)
    review=json.loads(path.read_text());del review['primary_task_checks']['source_hashes_match'];write(path,review)
    with pytest.raises(ValueError,match='primary task record audit did not pass'):
        analyze(root,audits,tmp_path/'summary',include_contact_rejections=True)
