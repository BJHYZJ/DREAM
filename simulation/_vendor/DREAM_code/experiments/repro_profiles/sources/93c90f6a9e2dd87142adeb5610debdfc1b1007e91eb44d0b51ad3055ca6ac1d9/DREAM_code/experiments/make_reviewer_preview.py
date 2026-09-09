#!/usr/bin/env python3
"""Make a low-text layout preview from recorded, burned-in simulator video.

This is an explicitly edited excerpt, never a new experiment. Pixel crops are
copied from the source; no inpainting or synthesized observations are used.
The original video and experiment manifests are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EPISODE = ROOT / "DREAM_code/experiments/results/maniskill_dynamic_video_suite_r2/episode_01_test-03_far-southeast-reloc-0"
DEFAULT_OUTPUT = ROOT / "clips/reviewer_preview"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SIZE = (1200, 640)
BACKGROUND = (246, 247, 249)
INK = (30, 40, 52)
MUTED = (95, 104, 116)
BLUE = (33, 105, 157)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_text(image: Image.Image, xy: tuple[int, int], value: str, size: int = 22,
              *, bold: bool = False, color=INK) -> None:
    ImageDraw.Draw(image).text(xy, value, font=ImageFont.truetype(BOLD if bold else FONT, size), fill=color)


def contain(image: Image.Image, width: int, height: int) -> Image.Image:
    # Enlarge only for readability. This does not create additional detail.
    scale = min(width / image.width, height / image.height)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def crop_boxes() -> dict[str, tuple[int, int, int, int]]:
    # Validated against the recorded 768x432 Test-03 source. Old opaque inset
    # labels/header are outside these crops; green target labels already burned
    # into the camera pixels are intentionally preserved, never inpainted.
    return {
        "observation": (496, 204, 752, 394),
        "physics": (216, 118, 490, 394),
        "memory": (16, 118, 212, 394),
    }


def compose(frame: np.ndarray, *, stage: str, note: str, source_seconds: float) -> np.ndarray:
    source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw_text(canvas, (24, 20), "Find and retrieve the orange object", 29, bold=True)
    draw_text(canvas, (24, 64), stage, 23, color=BLUE)
    draw_text(canvas, (890, 27), f"Recording {int(source_seconds)//60:02d}:{int(source_seconds)%60:02d}", 20, color=MUTED)
    boxes = crop_boxes()
    panels = (
        ("observation", "Robot observation", 24, 150, 446, 350),
        ("physics", "Physical execution", 490, 132, 342, 385),
        ("memory", "Map", 870, 147, 238, 344),
    )
    for key, label, x, y, width, height in panels:
        draw_text(canvas, (x, 108), label, 20, bold=True)
        content = contain(source.crop(boxes[key]), width, height)
        canvas.paste(content, (x + (width-content.width)//2, y + (height-content.height)//2))
    draw_text(canvas, (870, 493), "RGB-D + instance IDs", 14, color=MUTED)
    draw_text(canvas, (870, 514), "Orange: path   X: cached target", 13, color=MUTED)
    draw_text(canvas, (24, 543), note, 22)
    draw_text(canvas, (24, 599), "Edited excerpts  |  1x source playback  |  Controlled semantic input", 17, color=MUTED)
    return np.asarray(canvas)


def title_card() -> np.ndarray:
    canvas = Image.new("RGB", SIZE, BACKGROUND)
    draw_text(canvas, (72, 155), "Dynamic target reacquisition", 42, bold=True)
    draw_text(canvas, (72, 225), "Observe  /  object moves  /  observe again  /  grasp and place", 25, color=BLUE)
    draw_text(canvas, (72, 335), "A layout preview using excerpts from an existing simulation run.", 24)
    draw_text(canvas, (72, 380), "Instance IDs supply target semantics. This is not a full DREAM evaluation.", 22)
    draw_text(canvas, (72, 475), "No new experiment or baseline result is presented here.", 21, color=MUTED)
    return np.asarray(canvas)


def selected_segments(rows: list[dict[str, str]], fps: float, count: int) -> list[dict]:
    def frame(event: str) -> int:
        return int(next(row["video_frame_index"] for row in rows if row["event"] == event))

    cached = frame("target_cached")
    moving = frame("dynamic_motion_start")
    clear = frame("stale_target_region_rejected")
    lift = frame("measured_lift_contact_check")
    carry = frame("grid_planned_physical_carry")
    placed = frame("place_verification")
    segments = [
        (max(0, cached - 10*fps), moving, "1 / Observe the target", "The first target observation is cached."),
        (moving, min(clear + 2*fps, lift - 10*fps), "2 / The scene changes", "The support moves; focus checking follows. Reacquisition is quick in this run."),
        (max(clear + 2*fps, lift - 8*fps), lift + 2*fps, "3 / Grasp and lift", "The fingers close and the object is lifted."),
        (carry, carry + 5*fps, "4 / Carry", "The robot starts carrying the object toward the receptacle."),
        (max(carry + 5*fps, placed - 8*fps), min(count, placed + 3*fps), "5 / Place", "The robot releases the object onto the receptacle."),
    ]
    return [dict(first_frame=int(a), last_frame_exclusive=int(b), stage=stage, note=note)
            for a, b, stage, note in segments if b > a]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.episode / "manifest.json").read_text())
    if manifest["official_scene_id"] != "ArchitecTHOR-Test-03":
        raise ValueError("preview crop geometry has only been validated on Test-03")
    source = args.episode / "dream_dynamic_reacquisition.mp4"
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (768, 432):
        raise ValueError("preview requires the recorded 768x432 input")
    with (args.episode / "events.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    segments = selected_segments(rows, fps, count)
    args.output.mkdir(parents=True, exist_ok=True)
    movie = args.output / "01_low_text_preview.mp4"
    written = 0
    with imageio.get_writer(movie, fps=fps, codec="libx264", quality=7, macro_block_size=None) as writer:
        for _ in range(round(4 * fps)):
            writer.append_data(title_card())
            written += 1
        for i, segment in enumerate(segments):
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment["first_frame"])
            segment["output_first_frame"] = written
            for source_index in range(segment["first_frame"], segment["last_frame_exclusive"]):
                ok, raw = cap.read()
                if not ok:
                    raise RuntimeError(f"source decoding failed at {source_index}")
                result = compose(raw, stage=segment["stage"], note=segment["note"], source_seconds=source_index/fps)
                if source_index == segment["first_frame"] + min(15, (segment["last_frame_exclusive"]-segment["first_frame"])//2):
                    Image.fromarray(result).save(args.output / f"stage_{i+1:02d}.png")
                writer.append_data(result)
                written += 1
            segment["output_last_frame_exclusive"] = written
    cap.release()
    payload = {
        "artifact_type": "presentation_preview_from_recorded_video_not_new_experiment",
        "source_video": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "source_fps": fps,
        "output_sha256": sha256(movie),
        "output_duration_s": written / fps,
        "source_crops_xyxy": crop_boxes(),
        "edit_policy": "Chronological excerpts at original frame cadence. Cropping, scaling, titles only; no synthesized scene pixels, no baseline, no change to raw artifacts. Source timestamps are video time, not simulation time.",
        "limitations": "Existing burned-in camera annotations and low-resolution source remain. Camera perspective is cropped, not re-rendered. This run reacquires in the first focused observation; do not imply a long post-relocation search.",
        "segments": segments,
    }
    (args.output / "preview_provenance.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"output": str(movie), "seconds": written/fps, "frames": written}, indent=2))


if __name__ == "__main__":
    main()
