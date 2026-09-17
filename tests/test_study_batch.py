import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dream_sim import study


def setup_study(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    source = engine / "base"
    (engine / "experiments").mkdir(parents=True)
    (source / "experiments").mkdir(parents=True)
    (source / "experiments/policy.py").write_bytes(b"baseline\n")
    (source / "experiments/evaluator.py").write_bytes(b"evaluator unchanged\n")
    revision = tmp_path / "controllers/compact_v1"
    (revision / "experiments").mkdir(parents=True)
    (revision / "experiments/policy.py").write_bytes(b"candidate\n")
    (revision / "controller.json").write_text(
        json.dumps(
            dict(
                base_study_source_id="base",
                overrides={"experiments/policy.py": study.sha(b"candidate\n")},
            )
        )
    )
    (engine / "lock.json").write_text("{}")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "dream_models.lock.json").write_text("{}")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    tasks = [tmp_path / "first-task.json", tmp_path / "second-task.json"]
    manifest = dict(
        id="locked-cohort",
        seed=42,
        variants=["dynamic"],
        tasks=[dict(scene="first-house"), dict(scene="second-house")],
    )
    catalog = dict(
        study={"source_id": "base"},
        sources={"base": {}},
        cases=[],
        model_lock={"file": "lock.json"},
    )
    monkeypatch.setattr(study, "PROJECT", tmp_path)
    monkeypatch.setattr(study, "engine_root", lambda: engine)
    monkeypatch.setattr(study, "verify_public_configs", lambda: None)
    monkeypatch.setattr(study, "load_task_manifest", lambda _: (manifest, tasks))
    monkeypatch.setitem(
        sys.modules,
        "run_instruction_profile",
        SimpleNamespace(
            load_catalog=lambda _: (engine, catalog),
            verify_source=lambda *_: source,
            verify_case=lambda *_: None,
        ),
    )
    args = [
        "study",
        "--controller",
        "compact_v1",
        "--task-manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "screen"),
        "--model-cache",
        str(cache),
        "--gpus",
        "0",
        "1",
        "--parallel",
        "--raster-threads",
        "4",
        "--execute",
    ]
    monkeypatch.setattr(sys, "argv", args)
    return tasks


@pytest.mark.parametrize(
    "workers_per_gpu,from_profile", [(1, False), (1, True), (2, False), (4, False), (4, True)]
)
@pytest.mark.parametrize(
    "robot_time_limit,wall_time_limit",
    [(900, 2700), (1200, 2700), (1200, 3600), (1500, 4500), (1800, 5400), (1800, 0)],
)
def test_parallel_prepares_frozen_cohort_before_bounded_execution(
    tmp_path, monkeypatch, workers_per_gpu, from_profile, robot_time_limit, wall_time_limit
):
    tasks = setup_study(tmp_path, monkeypatch)
    if robot_time_limit != 900:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                *sys.argv,
                "--robot-time-limit-seconds",
                str(robot_time_limit),
                "--wait-for-slot-seconds",
                "1200",
            ],
        )
    if wall_time_limit != 2700:
        monkeypatch.setattr(
            sys, "argv", [*sys.argv, "--wall-timeout-seconds", str(wall_time_limit)]
        )
    if from_profile:
        profile = tmp_path / ".runtime/worker_resource_limits.json"
        profile.parent.mkdir()
        profile.write_text(json.dumps(dict(workers_per_gpu=workers_per_gpu)))
    elif workers_per_gpu != 2:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--workers-per-gpu", str(workers_per_gpu)])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            assert "--prepare-only" in command
            assert command[command.index("--wall-timeout") + 1] == str(wall_time_limit)
            assert command[command.index("--simulation-time-limit-seconds") + 1] == str(
                robot_time_limit
            )
            assert command[command.index("--tasks") + 1 : command.index("--output")] == list(
                map(str, tasks)
            )
            frozen = Path(command[command.index("--source-repo") + 1])
            assert (frozen / "experiments/evaluator.py").read_bytes() == b"evaluator unchanged\n"
            assert (frozen / "experiments/policy.py").read_bytes() == b"candidate\n"
        else:
            assert command[1:3] == ["-m", "dream_sim.batch"]
            for flag, value in [
                ("--time-limit-seconds", str(wall_time_limit)),
                ("--robot-time-limit-seconds", str(robot_time_limit)),
                ("--wait-for-slot-seconds", "1200" if robot_time_limit != 900 else "120"),
                ("--workers-per-gpu", str(workers_per_gpu)),
                ("--raster-threads", "4"),
            ]:
                assert command[command.index(flag) + 1] == value
            assert command[command.index("--run") + 1] == str(tmp_path / "screen")
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(study.subprocess, "run", run)
    study.main()
    assert len(calls) == 2


@pytest.mark.parametrize("seconds", [0, -1, 1801])
def test_robot_time_limit_rejects_unsupported_budgets(tmp_path, monkeypatch, seconds):
    setup_study(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            *sys.argv,
            "--robot-time-limit-seconds",
            str(seconds),
        ],
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2
    assert not (tmp_path / "screen").exists()


def test_failed_preparation_never_starts_workers(tmp_path, monkeypatch):
    setup_study(tmp_path, monkeypatch)
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(study.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        study.main()
    assert len(calls) == 1 and "--prepare-only" in calls[0]


def test_parallel_requires_locked_cohort(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["study", "--parallel", "--output", str(tmp_path / "screen")])
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2


def test_default_staged_return_is_materialized_and_launched_for_locked_cohort(
    tmp_path, monkeypatch
):
    setup_study(tmp_path, monkeypatch)
    directory = tmp_path / "controllers/staged_return"
    (directory / "experiments").mkdir(parents=True)
    payload = b"arm return implementation\n"
    (directory / "experiments/arm_return.py").write_bytes(payload)
    (directory / "controller.json").write_text(
        json.dumps(
            dict(
                base_controller="compact_v1",
                base_study_source_id="base",
                overrides={"experiments/arm_return.py": study.sha(payload)},
            )
        )
    )
    monkeypatch.setattr(sys, "argv", [sys.argv[0], *sys.argv[3:]])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            frozen = Path(command[command.index("--source-repo") + 1])
            assert (frozen / "experiments/arm_return.py").read_bytes() == payload
            assert (frozen / "experiments/policy.py").read_bytes() == b"candidate\n"
            assert command[command.index("--cohort-id") + 1] == "locked-cohort"
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(study.subprocess, "run", run)
    study.main()
    assert len(calls) == 2


@pytest.mark.parametrize("seconds", [-1, 5401])
def test_wall_time_limit_rejects_unsupported_watchdogs(tmp_path, monkeypatch, seconds):
    setup_study(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--wall-timeout-seconds", str(seconds)])
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2
    assert not (tmp_path / "screen").exists()


def test_custom_wall_time_limit_requires_parallel_batch(tmp_path, monkeypatch):
    setup_study(tmp_path, monkeypatch)
    argv = [arg for arg in sys.argv if arg != "--parallel"]
    monkeypatch.setattr(sys, "argv", [*argv, "--wall-timeout-seconds", "3600"])
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2
    assert not (tmp_path / "screen").exists()
