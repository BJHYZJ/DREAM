import hashlib
import json
import zipfile

import pytest

from dream_sim.verify_evidence import verify_residential


def evidence(tmp_path):
    (tmp_path / 'attempts').mkdir()
    (tmp_path / 'analysis').mkdir()
    rows=[]
    for i in range(50):
        result=json.dumps(dict(evaluator_task_success=i==0)).encode()
        path=tmp_path / 'attempts' / f'run{i}.zip'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('result.json', result)
            if i==0:
                archive.writestr('audit/record/record_review.json',json.dumps(dict(primary_task_record_review_passed=True)))
        rows.append(dict(name=f'run{i}',scene=f'house{i}',seed=42,variant='dynamic',task_success=i==0,
            qualified_task_success=i==0,result_sha256=hashlib.sha256(result).hexdigest()))
    report=dict(attempts=rows,all_planned_outcomes_bound=True,all_counted_successes_passed_audits=True,
                statistics=dict(variants=dict(dynamic=dict(attempts=50,successes=1,success_rate=.02))))
    (tmp_path / 'analysis/study_analysis.json').write_text(json.dumps(report))
    checksums={p.relative_to(tmp_path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path / 'checksums.json').write_text(json.dumps(checksums))
    return report,checksums


def test_residential_evidence_retains_every_failure(tmp_path):
    evidence(tmp_path)
    assert verify_residential(tmp_path)==dict(attempts=50,successes=1,compact_records_verified=True)


def test_residential_evidence_rejects_changed_archive(tmp_path):
    evidence(tmp_path)
    (tmp_path / 'attempts/run49.zip').write_bytes(b'changed')
    with pytest.raises(ValueError,match='evidence changed'):
        verify_residential(tmp_path)


def test_residential_evidence_rejects_rehashed_inflated_summary(tmp_path):
    report,checksums=evidence(tmp_path)
    report['statistics']['variants']['dynamic'].update(successes=40,success_rate=.8)
    path=tmp_path / 'analysis/study_analysis.json';path.write_text(json.dumps(report))
    checksums['analysis/study_analysis.json']=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'checksums.json').write_text(json.dumps(checksums))
    with pytest.raises(ValueError,match='counts differ'):
        verify_residential(tmp_path)
