#!/usr/bin/env python3
"""Package the audited ten-video TMECH revision without simulator assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import zipfile


WORKSPACE = Path(__file__).resolve().parents[2]
REPOSITORY = WORKSPACE / "DREAM_code"
MANUSCRIPT = WORKSPACE / "manuscript"
CLIPS = WORKSPACE / "clips"


def write_zip(output: Path, files: list[tuple[Path, str]]) -> None:
    temporary = output.with_suffix(output.suffix + ".part")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for source, name in sorted(files, key=lambda pair: pair[1]):
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, name)
    os.replace(temporary, output)


def overleaf_files() -> list[tuple[Path, str]]:
    names = (
        "DREAM.tex",
        "DREAM.pdf",
        "DREAM_R2_Tracked_Changes.tex",
        "DREAM_R2_Tracked_Changes.pdf",
        "Response_R2.tex",
        "Response_R2.pdf",
        "reference.bib",
        "IEEEtran.cls",
        "fig1.png",
    )
    files = [(MANUSCRIPT / name, name) for name in names]
    files.extend(
        (path, path.relative_to(MANUSCRIPT).as_posix())
        for path in (MANUSCRIPT / "figs").iterdir()
        if path.is_file()
    )
    return files


def source_files(suite_root: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for name in ("README.md", "LICENSE", ".gitignore"):
        files.append((REPOSITORY / name, f"DREAM_code/{name}"))
    files.append((REPOSITORY / "src/setup.py", "DREAM_code/src/setup.py"))
    for directory in (
        REPOSITORY / "src/dream",
        REPOSITORY / "tests",
        REPOSITORY / "experiments",
    ):
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(REPOSITORY)
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if "results" in relative.parts:
                continue
            if relative.parts[:2] == ("experiments", "data"):
                continue
            files.append((path, f"DREAM_code/{relative.as_posix()}"))
    report = REPOSITORY / "experiments/results/REPORT.md"
    files.append((report, "DREAM_code/experiments/results/REPORT.md"))
    files.append((suite_root / "aggregate.json", "DREAM_code/experiments/results/maniskill_dynamic_video_suite_r2/aggregate.json"))
    for manifest in sorted(suite_root.glob("episode_*/manifest.json")):
        episode = manifest.parent.name
        for name in ("manifest.json", "events.csv", "memory.csv", "control.csv", "trajectory.csv", "run.log"):
            path = manifest.parent / name
            files.append((path, f"DREAM_code/experiments/results/maniskill_dynamic_video_suite_r2/{episode}/{name}"))
    return files


def supplementary_files(suite_root: Path, evidence_root: Path) -> list[tuple[Path, str]]:
    files = [(suite_root / "aggregate.json", "aggregate.json")]
    for manifest in sorted(suite_root.glob("episode_*/manifest.json")):
        episode = manifest.parent.name
        files.extend(
            (
                manifest.parent / name,
                f"episodes/{episode}/{name}",
            )
            for name in (
                "dream_dynamic_reacquisition.mp4",
                "manifest.json",
                "events.csv",
                "memory.csv",
                "control.csv",
                "trajectory.csv",
            )
        )
    files.extend(
        (path, f"evidence/{path.name}")
        for path in sorted(evidence_root.glob("*.png"))
    )
    files.append((CLIPS / "README.md", "README.md"))
    return files


def link_videos(suite_root: Path) -> None:
    video_root = CLIPS / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = {}
    for manifest in sorted(suite_root.glob("episode_*/manifest.json")):
        scene = json.loads(manifest.read_text(encoding="utf-8"))["official_scene_id"]
        groups.setdefault(scene, []).append(manifest)
    manifests: list[Path] = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                manifests.append(group.pop(0))
    for index, manifest in enumerate(manifests, 1):
        source = manifest.parent / "dream_dynamic_reacquisition.mp4"
        target = video_root / f"{index:02d}_{manifest.parent.name.removeprefix('episode_')}.mp4"
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            raise FileExistsError(f"refusing to replace non-symlink video: {target}")
        target.symlink_to(os.path.relpath(source, video_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=REPOSITORY / "experiments/results/maniskill_dynamic_video_suite_r2",
    )
    parser.add_argument(
        "--evidence-root", type=Path, default=CLIPS / "dynamic_r2_final"
    )
    args = parser.parse_args()
    aggregate = json.loads((args.suite_root / "aggregate.json").read_text(encoding="utf-8"))
    if aggregate["successful_episode_count"] != 10 or aggregate["failed_episode_count"]:
        raise RuntimeError("the formal suite is not a clean audited 10/10 completion")
    if aggregate["distinct_successful_scene_count"] != 4:
        raise RuntimeError("the formal suite does not contain four official interiors")
    if aggregate["distinct_successful_target_placement_layout_count"] != 6:
        raise RuntimeError("the formal suite does not contain six target/place layouts")
    if aggregate["distinct_successful_dynamic_endpoint_count"] != 10:
        raise RuntimeError("the formal suite does not contain ten dynamic endpoints")

    CLIPS.mkdir(exist_ok=True)
    link_videos(args.suite_root)
    write_zip(CLIPS / "UPLOAD_Overleaf_TMECH_R2.zip", overleaf_files())
    write_zip(
        CLIPS / "REPRODUCIBILITY_DREAM_Code_and_Logs.zip",
        source_files(args.suite_root),
    )
    write_zip(
        CLIPS / "SUPPLEMENTARY_ManiSkill_10_Videos.zip",
        supplementary_files(args.suite_root, args.evidence_root),
    )


if __name__ == "__main__":
    main()
