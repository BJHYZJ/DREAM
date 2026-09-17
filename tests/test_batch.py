import subprocess
import sys
import time

import pytest

from dream_sim.batch import action_time_success, classify_failure, wait_for_execution


def test_whole_task_deadline_excludes_queue_and_keeps_timeout_a_failure(tmp_path):
    path = tmp_path / "worker.started.json"
    script = tmp_path / "fake_worker.py"
    script.write_text(
        'import json,time,sys\nfrom pathlib import Path\ntime.sleep(.3)\nPath(sys.argv[1]).write_text(json.dumps({"started_unix_s":time.time()}))\ntime.sleep(5)\n'
    )
    start = time.time()
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), str(path)], start_new_session=True, stdout=log, stderr=log
        )
        code, timed_out, elapsed = wait_for_execution(
            process, path, 0.2, grace=0.2, poll_seconds=0.02
        )
    assert timed_out and code != 0 and elapsed >= 0.2
    assert 0.5 <= time.time() - start < 3
    assert (
        classify_failure({"evaluator_task_success": True}, True, "placement_search")
        == "runtime_watchdog_timeout"
    )


def test_clean_completion_uses_recorded_finish_time(tmp_path):
    path = tmp_path / "worker.started.json"
    finish = tmp_path / "worker.finished.json"
    script = tmp_path / "fake_worker.py"
    script.write_text(
        'import json,time,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps({"started_unix_s":time.time()}))\ntime.sleep(.05)\nPath(sys.argv[2]).write_text(json.dumps({"finished_unix_s":time.time()}))\n'
    )
    process = subprocess.Popen(
        [sys.executable, str(script), str(path), str(finish)], start_new_session=True
    )
    code, timed_out, elapsed = wait_for_execution(process, path, 1, grace=0.2, poll_seconds=0.02)
    assert code == 0 and not timed_out and 0.04 < elapsed < 1


def test_server_duration_is_separate_from_robot_action_duration():
    result = dict(
        evaluator_task_success=True,
        wall_time_s=1600,
        robot_action_seconds=700,
        robot_action_limit_seconds=900,
        robot_action_timeout=False,
    )
    assert action_time_success(result, 0, False, 900)
    assert not action_time_success(result, 0, True, 900)
    assert not action_time_success(dict(result, robot_action_seconds=901), 0, False, 900)
    assert not action_time_success(dict(result, robot_action_timeout=True), 0, False, 900)
    assert not action_time_success(dict(result, robot_action_seconds=None), 0, False, 900)
    assert not action_time_success(dict(result, robot_action_limit_seconds=1800), 0, False, 900)
    assert (
        classify_failure(dict(result, robot_action_timeout=True), False, "pickup_search")
        == "pickup_search_action_timeout"
    )


def test_extended_budget_cannot_rescore_a_previous_shorter_trial():
    completed = dict(
        evaluator_task_success=True,
        robot_action_seconds=1050,
        robot_action_limit_seconds=1200,
        robot_action_timeout=False,
    )
    assert action_time_success(completed, 0, False, 1200)
    assert not action_time_success(completed, 0, False, 900)
    previous = dict(completed, robot_action_seconds=900, robot_action_limit_seconds=900)
    assert not action_time_success(previous, 0, False, 1200)
    assert not action_time_success(dict(completed, robot_action_seconds=1200.05), 0, False, 1200)


@pytest.mark.parametrize("already_finished", [False, True])
def test_disabled_server_watchdog_never_rejects_long_execution(tmp_path, already_finished):
    started = tmp_path / "worker.started.json"
    finished = tmp_path / "worker.finished.json"
    script = tmp_path / "worker.py"
    script.write_text(
        "import json,sys,time\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps({'started_unix_s':time.time()-10000}))\n"
        "time.sleep(.05)\n"
        "Path(sys.argv[2]).write_text(json.dumps({'finished_unix_s':time.time()}))\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(script), str(started), str(finished)], start_new_session=True
    )
    if already_finished:
        process.wait(timeout=5)
    code, timed_out, elapsed = wait_for_execution(process, started, 0, poll_seconds=0.01)
    assert code == 0 and timed_out is False and elapsed >= 10000
    result = dict(
        evaluator_task_success=True,
        robot_action_seconds=1800,
        robot_action_limit_seconds=1800,
        robot_action_timeout=False,
    )
    assert action_time_success(result, code, timed_out, 1800)
    assert not action_time_success(
        dict(result, robot_action_seconds=1800.05), code, timed_out, 1800
    )
