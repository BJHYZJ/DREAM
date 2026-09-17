"""Portable allocations must fit the caller's resources and preserve existing profiles."""

import json
import sys

import pytest

from dream_sim import configure


def test_single_worker_uses_available_cores_not_authors_cpu_ids(monkeypatch):
    monkeypatch.setattr(configure.os, "sched_getaffinity", lambda _: {17, 21, 35, 39})
    config = configure.worker_config(["0"], 1, 2)
    assert config["allowed_gpu_ids"] == ["0"]
    assert config["cpu_affinity_by_worker"] == {"0:0": [17, 21]}
    assert config["max_concurrent_workers"] == 1
    assert config["worker_memory_gib"] is None
    assert "memory_cgroup_parent" not in config


def test_rejects_oversubscribed_allocation(monkeypatch):
    monkeypatch.setattr(configure.os, "sched_getaffinity", lambda _: {1, 2})
    with pytest.raises(ValueError, match="distinct available CPU"):
        configure.worker_config(["0", "1"], 1, 2)


def test_existing_machine_allocation_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "worker_resource_limits.json"
    original = json.dumps({"allowed_gpu_ids": ["3"]})
    path.write_text(original)
    monkeypatch.setattr(configure.os, "sched_getaffinity", lambda _: {1, 2})
    monkeypatch.setattr(sys, "argv", ["configure", "--gpus", "0", "--output", str(path)])
    with pytest.raises(SystemExit) as error:
        configure.main()
    assert error.value.code == 2
    assert path.read_text() == original
