#!/usr/bin/env python3
"""Download and read the public RGB-D sequences used by DREAM benchmarks."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATASETS = {
    "tum_fr1_room": (
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/"
        "rgbd_dataset_freiburg1_room.tgz"
    ),
    "tum_fr2_pioneer_slam2": (
        "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/"
        "rgbd_dataset_freiburg2_pioneer_slam2.tgz"
    ),
    "bonn_moving_nonobstructing_box": (
        "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/"
        "rgbd_bonn_moving_nonobstructing_box.zip"
    ),
    "bonn_moving_obstructing_box": (
        "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/"
        "rgbd_bonn_moving_obstructing_box.zip"
    ),
    "bonn_removing_nonobstructing_box": (
        "https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/"
        "rgbd_bonn_removing_nonobstructing_box.zip"
    ),
}


@dataclass(frozen=True)
class RgbdRecord:
    timestamp: float
    rgb_path: Path
    depth_path: Path
    pose: tuple[float, float, float, float, float, float, float]


def _read_index(path: Path, values: int) -> list[tuple[float, tuple[str, ...]]]:
    rows: list[tuple[float, tuple[str, ...]]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < values + 1:
                continue
            rows.append((float(fields[0]), tuple(fields[1 : values + 1])))
    return rows


def _nearest(
    timestamp: float,
    rows: list[tuple[float, tuple[str, ...]]],
    start: int,
    max_difference: float,
) -> tuple[int, tuple[float, tuple[str, ...]]] | None:
    while start + 1 < len(rows) and abs(rows[start + 1][0] - timestamp) <= abs(
        rows[start][0] - timestamp
    ):
        start += 1
    if abs(rows[start][0] - timestamp) > max_difference:
        return None
    return start, rows[start]


def associate_sequence(
    sequence_dir: Path,
    *,
    target_hz: float = 5.0,
    max_difference: float = 0.03,
) -> list[RgbdRecord]:
    """Associate RGB, depth, and ground-truth poses in TUM text format."""

    rgb_rows = _read_index(sequence_dir / "rgb.txt", 1)
    depth_rows = _read_index(sequence_dir / "depth.txt", 1)
    pose_rows = _read_index(sequence_dir / "groundtruth.txt", 7)
    records: list[RgbdRecord] = []
    depth_idx = pose_idx = 0
    last_timestamp = float("-inf")
    min_period = 1.0 / target_hz

    for rgb_timestamp, (rgb_relpath,) in rgb_rows:
        if rgb_timestamp - last_timestamp < min_period:
            continue
        depth_match = _nearest(
            rgb_timestamp, depth_rows, depth_idx, max_difference=max_difference
        )
        pose_match = _nearest(
            rgb_timestamp, pose_rows, pose_idx, max_difference=max_difference
        )
        if depth_match is None or pose_match is None:
            continue
        depth_idx, (depth_timestamp, (depth_relpath,)) = depth_match
        pose_idx, (_, pose_fields) = pose_match
        records.append(
            RgbdRecord(
                timestamp=(rgb_timestamp + depth_timestamp) / 2.0,
                rgb_path=sequence_dir / rgb_relpath,
                depth_path=sequence_dir / depth_relpath,
                pose=tuple(float(value) for value in pose_fields),
            )
        )
        last_timestamp = rgb_timestamp
    if not records:
        raise RuntimeError(f"No associated RGB-D/pose records found in {sequence_dir}")
    return records


def locate_sequence(dataset_root: Path, dataset_name: str) -> Path:
    """Find the extracted directory containing TUM-format index files."""

    base = dataset_root / dataset_name
    candidates = [base, *base.glob("**/*")]
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / filename).is_file()
            for filename in ("rgb.txt", "depth.txt", "groundtruth.txt")
        ):
            return candidate
    raise FileNotFoundError(f"Could not locate an extracted sequence below {base}")


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    with urllib.request.urlopen(request) as response:
        mode = "ab" if existing and response.status == 206 else "wb"
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(destination)


def download_dataset(dataset_root: Path, dataset_name: str) -> Path:
    if dataset_name not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset_name!r}")
    try:
        return locate_sequence(dataset_root, dataset_name)
    except FileNotFoundError:
        pass

    url = DATASETS[dataset_name]
    archive = dataset_root / "archives" / Path(url).name
    if not archive.is_file():
        print(f"Downloading {dataset_name} from {url}", flush=True)
        _download(url, archive)

    destination = dataset_root / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive.name}", flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    elif archive.suffix in {".tgz", ".gz"}:
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(destination, filter="data")
    else:
        raise ValueError(f"Unsupported archive type: {archive}")
    return locate_sequence(dataset_root, dataset_name)


def ensure_datasets(dataset_root: Path, names: Iterable[str]) -> dict[str, Path]:
    return {name: download_dataset(dataset_root, name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "names", nargs="*", choices=sorted(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--root", type=Path, default=Path("experiments/data"))
    args = parser.parse_args()
    for name, path in ensure_datasets(args.root, args.names).items():
        records = associate_sequence(path)
        print(f"{name}: {path} ({len(records)} associated frames at 5 Hz)")


if __name__ == "__main__":
    main()
