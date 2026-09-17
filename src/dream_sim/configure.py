"""Create a worker allocation for a new machine without changing an existing one."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dream_sim.sources import PROJECT


def worker_config(gpus: list[str], workers_per_gpu: int, cpu_threads: int) -> dict:
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu.isdecimal() for gpu in gpus):
        raise ValueError("Choose unique physical GPU indices from nvidia-smi")
    workers = len(gpus) * workers_per_gpu
    if workers_per_gpu not in (1, 2, 4) or workers > 8:
        raise ValueError("Use one, two, or four workers per GPU and at most eight workers")
    cpus = sorted(os.sched_getaffinity(0))
    if cpu_threads not in (2, 4) or len(cpus) < workers * cpu_threads:
        raise ValueError("Each worker needs two or four distinct available CPU cores")
    affinity = {}
    for index, (gpu, slot) in enumerate(
        (gpu, slot) for gpu in gpus for slot in range(workers_per_gpu)
    ):
        affinity[f"{gpu}:{slot}"] = cpus[index * cpu_threads : (index + 1) * cpu_threads]
    return {
        "allowed_gpu_ids": gpus,
        "workers_per_gpu": workers_per_gpu,
        "max_concurrent_workers": workers,
        "cpu_affinity_by_worker": affinity,
        "worker_memory_gib": None,
        "total_memory_gib": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", required=True, help="Allocated physical GPU indices")
    parser.add_argument("--workers-per-gpu", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--cpu-threads", type=int, choices=(2, 4), default=2)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / ".runtime/worker_resource_limits.json"
    )
    args = parser.parse_args()
    try:
        config = worker_config(args.gpus, args.workers_per_gpu, args.cpu_threads)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(config, indent=2) + "\n")
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Worker allocation saved to {args.output}")
    print("CPU/GPU slots are configured. No kernel RAM or CUDA allocator cap was installed.")


if __name__ == "__main__":
    main()
