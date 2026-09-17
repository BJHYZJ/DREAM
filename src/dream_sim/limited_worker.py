"""Run a frozen entry point with explicit GPU and CPU thread limits.

The controller, observations, physics and evaluation code remain unchanged.
The adapter only bounds CPU pools and selects the video encoder speed preset;
its settings and hash are recorded separately from the frozen controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path


def configured_gpus(config_path: Path | None = None) -> tuple[str, ...]:
    """Use an explicit deployment allocation, never the caller's CUDA mask."""
    path = (
        config_path or Path(__file__).resolve().parents[2] / ".runtime/worker_resource_limits.json"
    )
    config = json.loads(path.read_text()) if path.exists() else {}
    values = config.get("allowed_gpu_ids", ["4", "5", "6", "7"])
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(gpu, str) or not gpu.isdecimal() for gpu in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("Deployment GPU allocation must contain unique device indices")
    return tuple(values)


ALLOWED_GPUS = configured_gpus()


def worker_environment(gpu: str, raster_threads: int = 8) -> dict[str, str]:
    if gpu not in ALLOWED_GPUS:
        raise ValueError(
            f"Inference is restricted to configured physical GPUs {', '.join(ALLOWED_GPUS)}"
        )
    if not 1 <= raster_threads <= 8:
        raise ValueError("Raster thread count must be between 1 and 8")
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=gpu,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
        OPENCV_FOR_THREADS_NUM="2",
        LP_NUM_THREADS=str(raster_threads),
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
    )
    for name in ("lvp_icd.json", "lvp_icd.x86_64.json"):
        icd = Path("/usr/share/vulkan/icd.d") / name
        if not environment.get("VK_ICD_FILENAMES") and icd.is_file():
            environment["VK_ICD_FILENAMES"] = str(icd)
            break
    return environment


def configure_video_threads() -> None:
    import cv2
    import imageio.v2 as imageio

    cv2.setNumThreads(2)
    original = imageio.get_writer

    def bounded_writer(*args, **kwargs):
        if kwargs.get("codec") == "libx264":
            params = list(kwargs.get("ffmpeg_params") or [])
            kwargs["ffmpeg_params"] = params + ["-threads", "2", "-preset", "veryfast"]
        return original(*args, **kwargs)

    imageio.get_writer = bounded_writer


def configure_cuda_memory_budget(receipt: Path, config_path: Path | None = None) -> None:
    """Cap this task's Torch allocator without reserving GPU memory."""
    path = (
        config_path or Path(__file__).resolve().parents[2] / ".runtime/worker_resource_limits.json"
    )
    config = json.loads(path.read_text()) if path.exists() else {}
    fraction = config.get("torch_cuda_memory_fraction")
    if fraction is None:
        return
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 0.10:
        raise ValueError("Per-task CUDA allocator budget must be positive and at most ten percent")
    import torch

    torch.cuda.set_per_process_memory_fraction(float(fraction), device=0)
    total = torch.cuda.get_device_properties(0).total_memory
    receipt.with_suffix(".cuda.json").write_text(
        json.dumps(
            dict(
                torch_allocator_fraction=fraction,
                torch_allocator_limit_bytes=int(total * fraction),
                device_total_bytes=total,
                memory_reserved_by_setting=False,
                boundary="Torch allocations only; driver/context overhead is additional. Actual per-card use must be monitored.",
            ),
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=ALLOWED_GPUS, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--raster-threads", type=int, default=8)
    parser.add_argument("--wait-for-slot-seconds", type=float, default=0)
    args, remaining = parser.parse_known_args()
    os.environ.update(worker_environment(args.gpu, args.raster_threads))
    # File invocation puts dream_sim/ itself first, where profile.py would
    # shadow Python's stdlib profile when Torch imports cProfile.
    package_directory = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != package_directory]
    entrypoint = args.entrypoint.resolve(strict=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as stream:
        json.dump(
            dict(
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                resource_guard_sha256=hashlib.sha256(
                    Path(__file__).with_name("worker_limits.py").read_bytes()
                ).hexdigest(),
                physical_gpu=args.gpu,
                raster_threads=args.raster_threads,
                worker_slot_wait_limit_s=args.wait_for_slot_seconds,
                opencv_threads=2,
                encoder_threads=2,
                encoder_preset="veryfast",
                entrypoint=str(entrypoint),
                arguments=remaining,
                controller_source_modified=False,
                simulation_settings_modified=False,
            ),
            stream,
            indent=2,
        )
    from dream_sim.worker_limits import available_worker

    with available_worker(args.gpu, args.receipt, wait_seconds=args.wait_for_slot_seconds):
        configure_video_threads()
        configure_cuda_memory_budget(args.receipt)
        sys.path.insert(0, str(entrypoint.parent))
        sys.argv = [str(entrypoint), *remaining]
        runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
