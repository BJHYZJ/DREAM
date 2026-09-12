"""A continuously tracked task can pass the task audit without lost-and-found."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('task_endpoint', [False, True])
def test_audit_status_uses_requested_endpoint(tmp_path, monkeypatch, task_endpoint):
    source = Path(__file__).resolve().parents[1] / 'controllers/recovery_v3/experiments/audit_instruction_batch.py'
    spec = importlib.util.spec_from_file_location('batch_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = tmp_path / 'batch'
    frozen = batch / 'frozen_workspace/DREAM_code/experiments'
    frozen.mkdir(parents=True)
    for name in ('replay_instruction_actions.py', 'review_instruction_record.py'):
        (frozen / name).touch()
    (batch / 'attempts.jsonl').write_text(json.dumps(dict(name='task', protocol_success=True,
        task_success=True, source_verified_after=True)) + '\n')
    (batch / 'batch_result.json').write_text('{}')
    output = tmp_path / 'audits'

    def run(command, **kwargs):
        destination = Path(command[command.index('--output') + 1])
        destination.mkdir(parents=True)
        if 'replay_instruction_actions.py' in command[1]:
            (destination / 'audit.json').write_text(json.dumps(dict(physical_reexecution_passed=True,
                contact_audit=dict(native_environment_contact_control_steps=0))))
        else:
            (destination / 'record_review.json').write_text(json.dumps(dict(record_review_passed=False,
                primary_task_record_review_passed=True, review_definition_version=4)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    monkeypatch.setattr(module.sys, 'argv', ['audit', '--batch', str(batch), '--output', str(output)]
                        + (['--all-task-successes'] if task_endpoint else []))
    module.main()
    status = json.loads((output / 'task/status.json').read_text())
    assert status['automatic_checks_passed'] is task_endpoint
    assert status['strict_checks_passed'] is False
    assert status['primary_task_checks_passed'] is True
    assert status['release_ready'] is False
