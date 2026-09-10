"""Validate the complete comparison before computing its final statistics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from dream_sim.sources import engine_root, safe_member


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read(path):return json.loads(path.read_text())


def contact_rejection(physical, review):
    """Recognize a completed audit that rejects only native-environment contact."""
    required={
        'source_hashes_match','saved_source_matches_hashes','instruction_first',
        'actual_control_reexecution','replay_sources_verified','replay_refers_to_this_run',
        'no_native_contact_above_threshold','complete_aligned_traces',
        'identity_records_consistent_with_saved_masks_and_replayed_positions',
        'fresh_visual_discovery_at_standoff_during_translation','continuous_1x_frame_steps',
        'all_source_videos_decode_completely','matching_5fps_playback',
        'current_observation_panel_matches_raw_camera','external_relocation_caption_visible',
        'annotated_video_retains_all_frames','original_task_scoring','reexecuted_task_success',
        'correct_visual_moved_target_observation','all_logged_updates_have_observed_free_depth',
        'logged_updates_have_variant_correct_caption'}
    checks=review.get('primary_task_checks',{})
    contact=physical.get('contact_audit',{})
    rows=contact.get('native_environment_contact_rows',[])
    steps=contact.get('native_environment_contact_control_steps')
    return (required<=set(checks) and all(type(value) is bool for value in checks.values())
        and {key for key,value in checks.items() if not value}=={'no_native_contact_above_threshold'}
        and review.get('primary_task_record_review_passed') is False
        and physical.get('evaluation',{}).get('evaluator_task_success') is True
        and type(steps) is int and steps>0 and bool(rows)
        and steps==len({row['control_step'] for row in rows})
        and all(row['force_n']>=0.5 for row in rows))


def analyze(root: Path,audits: Path,output: Path,baseline_path: Path | None=None,
            include_contact_rejections: bool=False):
    root=root.resolve();audits=audits.resolve();output=output.resolve()
    if not (root/'batch_result.json').exists():
        raise RuntimeError('The 60-attempt comparison has not finished')
    protocol=read(root/'protocol.json');batch=read(root/'batch_result.json')
    expected={job['name']:job for job in protocol['attempts']}
    if len(expected)!=60 or protocol.get('planned_attempts')!=60 or batch.get('all_planned_attempts_recorded') is not True:
        raise ValueError('Incomplete declared comparison')
    if batch['protocol_sha256']!=sha(root/'protocol.json'):
        raise ValueError('Protocol checksum mismatch')
    records={row['name']:row for row in batch['attempts']}
    if len(batch['attempts'])!=60 or set(records)!=set(expected):
        raise ValueError('Missing or duplicated attempts')
    rows=[]
    for name,job in expected.items():
        if len(safe_member(name).parts)!=1:raise ValueError('Invalid attempt directory')
        record=records[name];folder=root/name
        for flag in ('source_verified_before','source_verified_after','task_verified_before','task_verified_after'):
            if record.get(flag) is not True:raise ValueError(f'{name}: {flag}')
        task=root/safe_member(job['task'])
        if sha(task)!=job['task_sha256'] or sha(task.parent/'evaluator_room_map.npz')!=job['room_map_sha256']:
            raise ValueError(f'{name}: task input changed')
        if sha(folder/'result.json')!=record['result_sha256']:
            raise ValueError(f'{name}: result checksum mismatch')
        result=read(folder/'result.json')
        success=result.get('evaluator_task_success')
        if type(success) is not bool or record.get('task_success') is not success:
            raise ValueError(f'{name}: missing or inconsistent task outcome')
        if result.get('source_files_unchanged') is not True:
            raise ValueError(f'{name}: source changed during execution')
        for filename in ('source_hashes_before.json','source_hashes_after.json'):
            if read(folder/filename)!=protocol['source_sha256']:
                raise ValueError(f'{name}: controller differs from declared source')
        row=dict(name=name,scene=job['scene'],seed=job['seed'],variant=job['variant'],
                 task_success=success,qualified_task_success=success,
                 strict_success=result.get('evaluator_protocol_success'),
                 runner_status=record['status'],error=result.get('error'),eligible_for_summary=True,
                 result_sha256=record['result_sha256'],wall_time_s=result.get('wall_time_s'),
                 criteria=result.get('criteria',{}))
        if success:
            physical_path=audits/name/'physical/audit.json'
            review_path=audits/name/'record/record_review.json'
            physical=read(physical_path);review=read(review_path)
            for flag in ('physical_reexecution_passed','source_recording_unchanged',
                         'replay_environment_source_matches_recording','replay_source_unchanged'):
                if physical.get(flag) is not True:raise ValueError(f'{name}: {flag}')
            if review['physical_audit_sha256']!=sha(physical_path):
                raise ValueError(f'{name}: record audit refers to a different physics replay')
            if review['events_sha256']!=sha(folder/'events.jsonl'):
                raise ValueError(f'{name}: event record mismatch')
            passed=review.get('primary_task_record_review_passed') is True
            if not passed:
                if not include_contact_rejections or not contact_rejection(physical,review):
                    raise ValueError(f'{name}: primary task record audit did not pass')
                row.update(qualified_task_success=False,audit_qualification='native_contact_rejected')
            else:
                row['audit_qualification']='passed'
            row.update(physical_reexecution_passed=True,primary_task_record_review_passed=passed,
                physical_audit_sha256=sha(physical_path),record_review_sha256=sha(review_path),
                native_contact_steps=physical['contact_audit']['native_environment_contact_control_steps'])
        rows.append(row)

    sys.path.insert(0,str(engine_root()/'experiments'))
    from instruction_study_statistics import summarize_paired_outcomes
    reported_statistics=summarize_paired_outcomes(rows)
    qualified_rows=[dict(row,task_success=row['qualified_task_success']) for row in rows]
    statistics=summarize_paired_outcomes(qualified_rows)
    statistics['analysis_plan']=dict(statistics['analysis_plan'],
        primary_endpoint='Task completion with passing independent physical and primary-record audits',
        audit_rejections='Remain in the denominator as unsuccessful qualified outcomes; raw task outcomes retained separately')
    rejections=[dict(name=row['name'],reason=row['audit_qualification'],
                    native_contact_steps=row['native_contact_steps'])
                for row in rows if row.get('audit_qualification')=='native_contact_rejected']
    import numpy as np
    comparison={}
    if baseline_path is not None:
        baseline=list(csv.DictReader(baseline_path.open()))
        if len(baseline)!=60 or any(row['eligible_for_summary']!='True' or row['task_success'] not in ('True','False') for row in baseline):
            raise ValueError('The reference comparison is incomplete')
        if any(row.get('qualified_task_success',row['task_success']) not in ('True','False') for row in baseline):
            raise ValueError('The reference comparison has missing audit-qualified outcomes')
        before={(row['scene'],int(row['seed']),row['variant']):
                row.get('qualified_task_success',row['task_success'])=='True' for row in baseline}
        after={(row['scene'],row['seed'],row['variant']):row['qualified_task_success'] for row in rows}
        if set(before)!=set(after):raise ValueError('Before/after task sets differ')
        houses=statistics['house_order'];seeds=statistics['seed_order']
        rng=np.random.default_rng(20260908)
        draws=rng.integers(0,len(houses),size=(20000,len(houses)))
        for variant in ('dynamic','static'):
            a=np.asarray([[after[house,seed,variant] for seed in seeds] for house in houses])
            b=np.asarray([[before[house,seed,variant] for seed in seeds] for house in houses])
            differences=(a.astype(float)-b.astype(float)).mean(axis=1)
            comparison[variant]=dict(baseline_successes=int(b.sum()),candidate_successes=int(a.sum()),attempts=30,
                improved_attempts=int((a&~b).sum()),regressed_attempts=int((~a&b).sum()),
                difference_percentage_points=float(100*differences.mean()),
                paired_house_interval_percentage_points=(100*np.quantile(differences[draws].mean(axis=1),[.025,.975])).tolist())
    output.mkdir(parents=True,exist_ok=True)
    report=dict(protocol_sha256=sha(root/'protocol.json'),all_60_outcomes_bound=True,
        successful_task_physics_and_record_checks_passed=not rejections,
        all_reported_successes_audited=True,all_counted_successes_passed_audits=True,
        statistics_endpoint='audit-qualified task completion',audit_rejections=rejections,
        declared_analysis_plan=protocol.get('analysis_plan'),
        statistics=statistics,reported_task_statistics=reported_statistics,comparison_to_baseline=comparison,
        baseline_csv_sha256=sha(baseline_path) if baseline_path is not None else None,attempts=rows,
        scope='Same development-selected houses and seeds; no held-out or direct hardware comparison is implied.')
    (output/'comparison_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['name','scene','seed','variant','task_success','qualified_task_success','audit_qualification',
             'strict_success','eligible_for_summary','runner_status','wall_time_s','error','native_contact_steps']
    with (output/'attempts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--audits',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-csv',type=Path,help='Optional matched baseline attempts.csv for a before/after comparison')
    parser.add_argument('--include-contact-rejections',action='store_true',
        help='Report fully documented native-contact audit rejections as unsuccessful qualified outcomes, retaining raw task outcomes')
    args=parser.parse_args()
    report=analyze(args.run,args.audits,args.output,args.baseline_csv,args.include_contact_rejections)
    print(json.dumps(dict(variants=report['statistics']['variants'],
        memory_contrast=report['statistics']['contrast'],
        comparison_to_baseline=report['comparison_to_baseline'],
        reported_task_variants=report['reported_task_statistics']['variants'],
        audit_rejections=report['audit_rejections']),indent=2))


if __name__=='__main__':main()
