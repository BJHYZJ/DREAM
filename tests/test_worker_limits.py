import pytest

from dream_sim.worker_limits import WorkerLimits, WorkerSlotBusy, available_worker


def test_worker_slots_bound_total_and_prevent_duplicate_gpu(tmp_path):
    def worker(gpu):
        return WorkerLimits(
            gpu,
            tmp_path / (gpu + ".json"),
            lock_root=tmp_path / "locks",
            config_path=tmp_path / "no_kernel_config",
        )

    with worker("4"):
        with pytest.raises(RuntimeError, match="already holds"):
            with worker("4"):
                pass
        with worker("5"):
            with pytest.raises(RuntimeError, match="two-worker"):
                with worker("6"):
                    pass
        with worker("6"):
            pass
    with worker("7"):
        pass


def test_queued_launcher_acquires_only_after_existing_gpu_worker_exits(tmp_path):
    import threading

    kwargs = dict(lock_root=tmp_path / "locks", config_path=tmp_path / "no_kernel_config")
    acquired = threading.Event()

    def queued():
        with available_worker("0", tmp_path / "queued.json", wait_seconds=2, **kwargs):
            acquired.set()

    with WorkerLimits("0", tmp_path / "active.json", **kwargs):
        thread = threading.Thread(target=queued)
        thread.start()
        assert not acquired.wait(0.1)
        with pytest.raises(WorkerSlotBusy):
            with available_worker("0", tmp_path / "impatient.json", wait_seconds=0.05, **kwargs):
                pass
    thread.join(timeout=3)
    assert not thread.is_alive()
    assert acquired.is_set()


def test_two_per_gpu_and_four_global_slots_preserve_legacy_exclusion(tmp_path):
    import json

    config = tmp_path / "parallel.json"
    config.write_text(json.dumps({"workers_per_gpu": 2, "max_concurrent_workers": 4}))

    def worker(gpu, label, configured=True):
        return WorkerLimits(
            gpu,
            tmp_path / (label + ".json"),
            lock_root=tmp_path / "locks",
            config_path=config if configured else tmp_path / "no_config",
        )

    with worker("0", "legacy", False):
        with pytest.raises(WorkerSlotBusy):
            with worker("0", "blocked"):
                pass
    with worker("0", "a") as a, worker("0", "b") as b:
        assert {a.gpu_slot, b.gpu_slot} == {0, 1}
        with pytest.raises(WorkerSlotBusy):
            with worker("0", "third"):
                pass
        with worker("1", "c"), worker("1", "d"):
            with pytest.raises(WorkerSlotBusy):
                with worker("2", "fifth"):
                    pass
        with pytest.raises(WorkerSlotBusy):
            with worker("0", "legacy_blocked", False):
                pass
    with worker("0", "released"):
        assert (tmp_path / "released.started.json").exists()


@pytest.mark.parametrize("count", [2, 4])
def test_bounded_cpu_profiles_are_applied_and_recorded(tmp_path, monkeypatch, count):
    import json

    import dream_sim.worker_limits as module

    affinity = list(range(20, 20 + count))
    current = set(range(64))
    applied = []

    def apply(pid, cores):
        nonlocal current
        current = set(cores)
        applied.append((pid, list(cores)))

    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: current)
    monkeypatch.setattr(module.os, "sched_setaffinity", apply)
    monkeypatch.setattr(module.os, "nice", lambda increment: applied.append(("nice", increment)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(cpu_affinity_by_worker={"0:0": affinity})))
    receipt = tmp_path / "receipt.json"
    with WorkerLimits("0", receipt, lock_root=tmp_path / "locks", config_path=config):
        assert current == set(affinity)
        assert (
            json.loads(receipt.with_suffix(".started.json").read_text())["cpu_affinity"] == affinity
        )
    assert applied == [(0, affinity), ("nice", 10)]


@pytest.mark.parametrize("affinity", [[1], [1, 2, 3], [1, 2, 3, 4, 5], [1, 1], [1, 2, 3, 3]])
def test_invalid_cpu_profiles_are_rejected_before_affinity_change(tmp_path, monkeypatch, affinity):
    import json

    import dream_sim.worker_limits as module

    monkeypatch.setattr(module.os, "sched_getaffinity", lambda pid: set(range(64)))

    def unexpected(*args):
        raise AssertionError("Invalid profile must not be applied")

    monkeypatch.setattr(module.os, "sched_setaffinity", unexpected)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(cpu_affinity_by_worker={"0:0": affinity})))
    with pytest.raises(ValueError, match="distinct"):
        with WorkerLimits(
            "0", tmp_path / "receipt.json", lock_root=tmp_path / "locks", config_path=config
        ):
            pass
