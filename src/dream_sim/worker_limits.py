"""Shared worker slots and optional kernel memory limits for this deployment."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

MAX_WORKERS = 2


class WorkerSlotBusy(RuntimeError):
    pass


@contextmanager
def available_worker(gpu, receipt, *, wait_seconds=0, **kwargs):
    """Queue a lightweight launcher until a real worker slot is released."""
    if wait_seconds < 0:
        raise ValueError("Worker slot timeout must be nonnegative")
    deadline = time.monotonic() + wait_seconds
    with ExitStack() as stack:
        while True:
            try:
                worker = stack.enter_context(WorkerLimits(gpu, receipt, **kwargs))
                break
            except WorkerSlotBusy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(0.5, remaining))
        yield worker


class WorkerLimits:
    def __init__(self, gpu, receipt, *, lock_root=None, config_path=None):
        self.gpu = gpu
        self.receipt = Path(receipt)
        self.lock_root = Path(lock_root or f"/tmp/dream_worker_slots_{os.getuid()}")
        self.config_path = Path(
            config_path
            or Path(__file__).resolve().parents[2] / ".runtime/worker_resource_limits.json"
        )
        self.locks = []
        self.stop = threading.Event()
        self.thread = None
        self.peak_bytes = 0
        self.limit_bytes = None
        self.usage_path = None
        self.soft_limit_triggered = False
        self.max_workers = MAX_WORKERS
        self.gpu_slot = 0
        self.cpu_affinity = None

    def lock(self, name, *, shared=False):
        stream = (self.lock_root / name).open("a")
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            return False
        self.locks.append(stream)
        return True

    def __enter__(self):
        self.lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            config = json.loads(self.config_path.read_text()) if self.config_path.exists() else {}
            per_gpu = config.get("workers_per_gpu", 1)
            self.max_workers = config.get("max_concurrent_workers", MAX_WORKERS)
            if per_gpu not in (1, 2, 4) or not 1 <= self.max_workers <= 8:
                raise ValueError("Deployment supports at most four workers per GPU and eight total")
            if not self.lock("gpu" + self.gpu, shared=per_gpu > 1):
                raise WorkerSlotBusy("An inference/audit worker already holds this physical GPU")
            if per_gpu > 1:
                self.gpu_slot = next(
                    (i for i in range(per_gpu) if self.lock(f"gpu{self.gpu}_slot{i}")), None
                )
                if self.gpu_slot is None:
                    raise WorkerSlotBusy("All worker slots on this GPU are occupied")
            if not any(self.lock("slot" + str(i)) for i in range(self.max_workers)):
                raise WorkerSlotBusy(
                    "The two-worker memory limit is already in use"
                    if self.max_workers == 2
                    else "All deployment worker slots are occupied"
                )
            affinity = config.get("cpu_affinity_by_worker", {}).get(f"{self.gpu}:{self.gpu_slot}")
            if affinity is not None:
                if (
                    len(affinity) not in (2, 4)
                    or len(set(affinity)) != len(affinity)
                    or not set(affinity) <= os.sched_getaffinity(0)
                ):
                    raise ValueError(
                        "Each configured worker needs two or four distinct available CPU cores"
                    )
                os.sched_setaffinity(0, affinity)
                self.cpu_affinity = sorted(os.sched_getaffinity(0))
                os.nice(10)
            if config.get("memory_cgroup_parent"):
                parent = Path(config["memory_cgroup_parent"])
                group = parent / ("gpu" + self.gpu)
                if per_gpu > 1:
                    group = group / ("slot" + str(self.gpu_slot))
                self.limit_bytes = int((group / "memory.limit_in_bytes").read_text())
                if (
                    self.limit_bytes > config.get("worker_memory_gib", 16) * 1024**3
                    or self.limit_bytes > 16 * 1024**3
                    or not 0 < config.get("total_memory_gib", 32) <= 80
                    or int((parent / "memory.limit_in_bytes").read_text())
                    > config.get("total_memory_gib", 32) * 1024**3
                ):
                    raise ValueError("Configured worker memory limits exceed the deployment budget")
                (group / "cgroup.procs").write_text(str(os.getpid()))
                self.usage_path = group / "memory.usage_in_bytes"
                self.thread = threading.Thread(target=self.monitor, daemon=True)
                self.thread.start()
            self.receipt.parent.mkdir(parents=True, exist_ok=True)
            started = self.receipt.with_suffix(".started.json")
            temporary = started.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    dict(
                        pid=os.getpid(),
                        started_unix_s=time.time(),
                        physical_gpu=self.gpu,
                        gpu_slot=self.gpu_slot,
                        cpu_affinity=self.cpu_affinity,
                        kernel_memory_limit_bytes=self.limit_bytes,
                    ),
                    indent=2,
                )
                + "\n"
            )
            temporary.replace(started)
            return self
        except BaseException:
            self.close()
            raise

    def monitor(self):
        triggered_at = None
        while not self.stop.wait(1.0):
            try:
                usage = int(self.usage_path.read_text())
            except OSError:
                continue
            self.peak_bytes = max(self.peak_bytes, usage)
            if usage >= 0.90 * self.limit_bytes and triggered_at is None:
                self.soft_limit_triggered = True
                triggered_at = time.monotonic()
                os.kill(os.getpid(), signal.SIGINT)
            elif triggered_at is not None and time.monotonic() - triggered_at > 30:
                os.kill(os.getpid(), signal.SIGTERM)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        for stream in self.locks:
            stream.close()
        self.locks.clear()

    def __exit__(self, *exception):
        finished_unix_s = time.time()
        self.close()
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        self.receipt.with_suffix(".memory.json").write_text(
            json.dumps(
                dict(
                    max_concurrent_workers=self.max_workers,
                    physical_gpu=self.gpu,
                    gpu_slot=self.gpu_slot,
                    cpu_affinity=self.cpu_affinity,
                    kernel_memory_limit_bytes=self.limit_bytes,
                    peak_accounted_memory_bytes=self.peak_bytes,
                    memory_usage_file=str(self.usage_path) if self.usage_path else None,
                    graceful_stop_threshold_fraction=0.90,
                    soft_limit_triggered=self.soft_limit_triggered,
                ),
                indent=2,
            )
            + "\n"
        )
        self.receipt.with_suffix(".finished.json").write_text(
            json.dumps(dict(pid=os.getpid(), finished_unix_s=finished_unix_s), indent=2) + "\n"
        )
