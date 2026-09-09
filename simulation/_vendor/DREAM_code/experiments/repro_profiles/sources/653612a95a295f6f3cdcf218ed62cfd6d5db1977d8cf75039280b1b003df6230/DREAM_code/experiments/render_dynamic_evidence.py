#!/usr/bin/env python3
"""Render reviewer-facing contact sheets from successful dynamic manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


PIPELINE_EVENTS = (
    ("initial_360_scan_complete", "1  INITIAL 360 SCAN"),
    ("target_observed_from_current_rgbd", "2  FIRST RGB-D DETECTION"),
    ("dynamic_motion_complete", "3  FORCE-DRIVEN RELOCATION"),
    ("stale_target_region_rejected", "4  STALE VOXELS REMOVED"),
    ("target_observed_while_base_moving", "5  ONLINE REACQUISITION"),
    ("bilateral_contact_check", "6  BILATERAL GRASP"),
    ("measured_lift_contact_check", "7  MEASURED LIFT"),
    ("post_grasp_clearance_retreat", "8  PAYLOAD RETREAT/CARRY"),
    ("place_verification", "9  VERIFIED PLACEMENT"),
)


def success_episodes(root: Path) -> list[tuple[Path, dict]]:
    episodes = []
    for path in sorted(root.glob("episode_*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") == "success":
            episodes.append((path.parent, manifest))
    if not episodes:
        for path in sorted(root.glob("**/manifest.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") == "success":
                episodes.append((path.parent, manifest))
    # Reviewer-facing ordering alternates official interiors instead of
    # presenting every layout from one home consecutively.
    groups: dict[str, list[tuple[Path, dict]]] = {}
    for episode in episodes:
        groups.setdefault(episode[1]["official_scene_id"], []).append(episode)
    interleaved: list[tuple[Path, dict]] = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                interleaved.append(group.pop(0))
    return interleaved


def events(episode: Path) -> list[dict[str, str]]:
    with (episode / "events.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def frame_at(video: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(max(index, 0), max(count - 1, 0)))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} from {video}")
    return frame


def tile(frame: np.ndarray, label: str, width: int = 480, height: int = 270) -> np.ndarray:
    view = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    overlay = view.copy()
    cv2.rectangle(overlay, (0, height - 33), (width, height), (12, 18, 24), -1)
    cv2.addWeighted(overlay, 0.88, view, 0.12, 0, view)
    cv2.putText(
        view, label, (12, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.53, (250, 250, 250), 1, cv2.LINE_AA,
    )
    return view


def event_frame(episode: Path, names: tuple[str, ...]) -> tuple[np.ndarray, str]:
    rows = events(episode)
    for name in names:
        matches = [row for row in rows if row.get("event") == name]
        if matches:
            row = matches[-1]
            index = int(float(row.get("video_frame_index") or 0))
            return frame_at(episode / "dream_dynamic_reacquisition.mp4", index), name
    video = episode / "dream_dynamic_reacquisition.mp4"
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return frame_at(video, count // 2), "midpoint"


def layout_label(manifest: dict) -> str:
    """Return a human-readable physical layout ID without relying on the seed."""

    arguments = manifest.get("execution_arguments", {})
    mode = arguments.get("layout_mode")
    index = int(arguments.get("layout_index", 0))
    relocation = int(arguments.get("relocation_layout_index", 0))
    if mode == "custom":
        return f"custom-{index} / move-{relocation}"
    if mode == "legacy" or manifest["spec"]["scene_family"] == "replicacad":
        return f"legacy-0 / move-{relocation}"
    return f"auto-{index} / move-{relocation}"


def render_contact_sheet(episodes: list[tuple[Path, dict]], output: Path) -> None:
    views = []
    for index, (episode, manifest) in enumerate(episodes[:10], 1):
        frame, _ = event_frame(
            episode,
            (
                "target_observed_while_base_moving",
                "target_reacquired_in_focused_frame",
                "target_observed_from_current_rgbd",
            ),
        )
        spec = manifest["spec"]
        label = f"{index:02d}  {spec['scene_id']}  |  {layout_label(manifest)}"
        views.append(tile(frame, label))
    if len(views) < 10:
        raise RuntimeError(f"need 10 successful manifests, found {len(views)}")
    sheet = np.vstack([np.hstack(views[row : row + 2]) for row in range(0, 10, 2)])
    cv2.imwrite(str(output), sheet)


def render_pipeline(episode: Path, output: Path) -> None:
    rows = events(episode)
    first_dynamic_step = min(
        int(float(row["step"])) for row in rows if row.get("event") == "dynamic_motion_start"
    )
    stale_clear_step = max(
        int(float(row["step"])) for row in rows if row.get("event") == "stale_target_region_rejected"
    )

    def select(event_name: str) -> dict[str, str] | None:
        if event_name == "target_observed_from_current_rgbd":
            names = {
                "target_observed_from_current_rgbd",
                "target_observed_while_base_moving",
                "target_observed_during_initial_360_scan",
            }
            candidates = [
                row for row in rows
                if row.get("event") in names and int(float(row["step"])) <= first_dynamic_step
            ]
            return candidates[0] if candidates else None
        if event_name == "target_observed_while_base_moving":
            names = {
                "target_observed_while_base_moving",
                "target_reacquired_in_focused_frame",
                "target_observed_from_current_rgbd",
                "target_observed_during_stall_focus",
            }
            candidates = [
                row for row in rows
                if row.get("event") in names and int(float(row["step"])) >= stale_clear_step
            ]
            return candidates[0] if candidates else None
        candidates = [row for row in rows if row.get("event") == event_name]
        return candidates[-1] if candidates else None

    views = []
    video = episode / "dream_dynamic_reacquisition.mp4"
    for event_name, label in PIPELINE_EVENTS:
        row = select(event_name)
        if row is None:
            raise RuntimeError(f"{episode.name} is missing pipeline event {event_name}")
        views.append(tile(frame_at(video, int(float(row["video_frame_index"]))), label))
    montage = np.vstack([np.hstack(views[row : row + 3]) for row in range(0, 9, 3)])
    cv2.imwrite(str(output), montage)


def render_table(episodes: list[tuple[Path, dict]], output: Path) -> None:
    width, row_height, header_height = 2080, 58, 72
    canvas = np.full((header_height + row_height * (len(episodes) + 1), width, 3), 255, np.uint8)
    cv2.putText(
        canvas, "DREAM DYNAMIC PHYSICS SUITE - ONLY status=success MANIFESTS",
        (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (20, 28, 36), 2, cv2.LINE_AA,
    )
    columns = (
        (25, "ID"), (105, "Official scene"), (520, "layout"),
        (790, "steps"), (940, "base (m)"), (1120, "move (m)"),
        (1300, "stale voxels"), (1515, "peak/held lift"), (1775, "grasp/place"),
    )
    canvas[header_height : header_height + row_height] = (41, 60, 72)
    y = header_height + 38
    for x, label in columns:
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    for index, (_, manifest) in enumerate(episodes, 1):
        y0 = header_height + row_height * index
        if index % 2 == 0:
            canvas[y0 : y0 + row_height] = (241, 245, 247)
        spec = manifest["spec"]
        values = (
            f"{index:02d}", spec["scene_id"], layout_label(manifest),
            str(manifest["control_steps"]), f"{manifest['base_travel_m']:.2f}",
            f"{manifest['measured_relocation_m']:.3f}",
            f"{manifest['stale_region_voxels_before']} -> {manifest['stale_region_voxels_after']}",
            f"{manifest['peak_measured_object_lift_m']:.3f} / {manifest['measured_object_lift_m']:.3f}",
            f"{'YES' if manifest['grasp_bilateral_contact'] else 'NO'} / {'YES' if manifest['place_success'] else 'NO'}",
        )
        baseline = y0 + 37
        for (x, _), value in zip(columns, values):
            cv2.putText(canvas, value, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 43, 49), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def render_paper_summary(episodes: list[tuple[Path, dict]], output: Path) -> None:
    """Make a compact four-scene/pipeline panel suitable for a two-column paper."""

    by_scene: dict[str, tuple[Path, dict]] = {}
    for episode, manifest in episodes:
        by_scene.setdefault(manifest["official_scene_id"], (episode, manifest))
    if len(by_scene) != 4:
        raise RuntimeError(f"expected four official furnished scenes, found {len(by_scene)}")

    scene_views = []
    for scene, (episode, _) in by_scene.items():
        frame, _ = event_frame(
            episode,
            (
                "target_observed_while_base_moving",
                "target_reacquired_in_focused_frame",
            ),
        )
        scene_views.append(tile(frame, f"SCENE  {scene}", width=450, height=253))

    pipeline_names = (
        ("initial_360_scan_complete", "ROTATE + BUILD MEMORY"),
        ("dynamic_motion_complete", "PHYSICAL TARGET MOVE"),
        ("stale_target_region_rejected", "CLEAR STALE MEMORY"),
        ("measured_lift_contact_check", "GRASP + MEASURED LIFT"),
        ("place_verification", "CARRY + PLACE"),
    )
    first_episode = episodes[0][0]
    rows = events(first_episode)
    video = first_episode / "dream_dynamic_reacquisition.mp4"
    pipeline_views = []
    for event_name, label in pipeline_names:
        matching = [row for row in rows if row.get("event") == event_name]
        if not matching:
            raise RuntimeError(f"{first_episode.name} is missing {event_name}")
        row = matching[-1]
        pipeline_views.append(
            tile(
                frame_at(video, int(float(row["video_frame_index"]))),
                label,
                width=360,
                height=203,
            )
        )
    cv2.imwrite(str(output), np.vstack((np.hstack(scene_views), np.hstack(pipeline_views))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("../clips/dynamic_final"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = success_episodes(args.suite_root)
    if len(episodes) < 10:
        raise RuntimeError(f"need at least 10 successful episodes, found {len(episodes)}")
    episodes = episodes[:10]
    render_contact_sheet(episodes, args.output / "01_ten_video_contact_sheet.png")
    render_pipeline(episodes[0][0], args.output / "02_complete_pipeline_montage.png")
    render_table(episodes, args.output / "03_verified_result_table.png")
    render_paper_summary(episodes, args.output / "04_four_scene_pipeline_summary.png")


if __name__ == "__main__":
    main()
