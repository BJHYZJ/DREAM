"""Larger bounded concurrency keeps per-card, global, and RAM exclusions."""

import json
from contextlib import ExitStack

import pytest

from dream_sim.worker_limits import WorkerLimits, WorkerSlotBusy


def test_four_per_card_eight_global_and_legacy_exclusion(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(workers_per_gpu=4, max_concurrent_workers=8)))

    def worker(gpu, name, legacy=False):
        return WorkerLimits(
            gpu,
            tmp_path / (name + ".json"),
            lock_root=tmp_path / "locks",
            config_path=tmp_path / "absent.json" if legacy else config,
        )

    with ExitStack() as stack:
        workers = [stack.enter_context(worker(str(i // 4), str(i))) for i in range(8)]
        assert [w.gpu_slot for w in workers] == [0, 1, 2, 3, 0, 1, 2, 3]
        with pytest.raises(WorkerSlotBusy):
            with worker("0", "fifth_on_card"):
                pass
        with pytest.raises(WorkerSlotBusy, match="All deployment"):
            with worker("2", "ninth_total"):
                pass
        with pytest.raises(WorkerSlotBusy):
            with worker("0", "legacy", True):
                pass
    with worker("0", "released", True):
        pass


@pytest.mark.parametrize(
    "parent_gib,declared_gib,accepted",
    [(80, 80, True), (81, 80, False), (80, 32, False), (96, 96, False)],
)
def test_memory_limit_must_fit_declared_budget_and_absolute_cap(
    tmp_path, parent_gib, declared_gib, accepted
):
    parent = tmp_path / "kernel"
    group = parent / "gpu0/slot0"
    group.mkdir(parents=True)
    (parent / "memory.limit_in_bytes").write_text(str(parent_gib * 1024**3))
    (group / "memory.limit_in_bytes").write_text(str(10 * 1024**3))
    (group / "memory.usage_in_bytes").write_text("0")
    (group / "cgroup.procs").write_text("")
    config = tmp_path / "config.json"
    receipt = tmp_path / "receipt.json"
    config.write_text(
        json.dumps(
            dict(
                workers_per_gpu=4,
                max_concurrent_workers=8,
                memory_cgroup_parent=str(parent),
                worker_memory_gib=10,
                total_memory_gib=declared_gib,
            )
        )
    )
    worker = WorkerLimits("0", receipt, lock_root=tmp_path / "locks", config_path=config)
    if accepted:
        with worker:
            assert worker.limit_bytes == 10 * 1024**3
        assert (
            json.loads(receipt.with_suffix(".started.json").read_text())[
                "kernel_memory_limit_bytes"
            ]
            == 10 * 1024**3
        )
    else:
        with pytest.raises(ValueError, match="budget"):
            with worker:
                pass
        assert not receipt.with_suffix(".started.json").exists()
