#!/usr/bin/env python3
"""Create a timestamped qualitative timeline from DREAM's public robot videos."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


VIDEOS = {
    "combination_01": "https://bjhyzj.github.io/dream-web/media/videos/combination_01.mp4",
    "combination_02": "https://bjhyzj.github.io/dream-web/media/videos/combination_02.mp4",
}
TIMELINES = {
    "combination_01": ((14, "Online mapping"), (42, "Target search"), (69, "Pickup/navigation"), (96, "Placement")),
    "combination_02": ((19, "Online mapping"), (56, "Target search"), (94, "Pickup/navigation"), (130, "Placement")),
}


def download(url: str, path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def read_frames(path: Path, timestamps: list[float]):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    targets = [int(round(seconds * fps)) for seconds in timestamps]
    frames = []
    frame_index = 0
    target_index = 0
    while target_index < len(targets):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"Could not decode {path} at {timestamps[target_index]} s")
        if frame_index >= targets[target_index]:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            target_index += 1
        frame_index += 1
    capture.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data/videos"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/video_timeline"))
    args = parser.parse_args()
    local_paths = {}
    for name, url in VIDEOS.items():
        path = args.data_root / f"{name}.mp4"
        download(url, path)
        local_paths[name] = path

    figure, axes = plt.subplots(2, 4, figsize=(12, 5.0), constrained_layout=True)
    for row, name in enumerate(VIDEOS):
        timestamps = [timestamp for timestamp, _ in TIMELINES[name]]
        frames = read_frames(local_paths[name], timestamps)
        for axis, (timestamp, stage), frame in zip(axes[row], TIMELINES[name], frames):
            axis.imshow(frame)
            axis.set_title(f"{stage} ({timestamp} s)", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(
            "Screwdriver $\\rightarrow$ green box" if row == 0 else "Yellow knife $\\rightarrow$ red basket",
            fontsize=9,
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_root / "real_robot_timeline.png", dpi=240)
    figure.savefig(args.output_root / "real_robot_timeline.pdf")
    plt.close(figure)
    metadata = {
        "source_urls": VIDEOS,
        "timestamps_seconds": TIMELINES,
        "note": "Frames are sampled from previously released, sped-up public demonstrations; they are qualitative evidence, not independent trials.",
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
