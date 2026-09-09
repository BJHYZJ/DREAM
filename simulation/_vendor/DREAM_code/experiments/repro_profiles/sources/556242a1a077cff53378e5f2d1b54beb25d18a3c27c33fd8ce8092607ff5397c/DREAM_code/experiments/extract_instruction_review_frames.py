#!/usr/bin/env python3
"""Extract unmodified decoded frames for human inspection, not task scoring."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_frames(frames, trajectory, evidence):
    """Use audited events and evaluator timestamps only to choose review frames."""
    if not frames:
        raise ValueError("No recorded frames")
    selected = {}
    for label, key in (("observed_target_missing", "observed_loss_step"),
                       ("reacquired", "reacquisition_step")):
        step = evidence.get(key)
        if step is None:
            raise ValueError(f"Missing audited event: {key}")
        selected[label] = next((f for f in frames if f["sim_step"] >= step), None)
        if selected[label] is None:
            raise ValueError(f"Event outside video: {key}")
    held = [r for r in trajectory if r["task_stage"] == "placement_search"
            and r["bilateral_contact"]]
    if not held:
        raise ValueError("No evaluator-recorded held transport")
    midpoint = held[len(held) // 2]["step"]
    selected["held_transport"] = min(frames, key=lambda f: abs(f["sim_step"] - midpoint))
    selected["released_placement"] = frames[-1]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alternate-replay", type=Path,
                        help="Optional independently checked same-controls spectator re-render")
    args = parser.parse_args()
    review = json.loads((args.review / "record_review.json").read_text())
    if not review["record_review_passed"]:
        raise ValueError("This extraction entrypoint requires a passing recorded-data audit")
    run = Path(review["source_run"])
    video = args.review / "reviewer_view.mp4"
    expected = review.get("annotated_video_sha256") or review["source_video_sha256"]["reviewer_view"]
    if sha(video) != expected:
        raise ValueError("Video differs from the audited file")
    frames = json.loads((run / "video_frames.json").read_text())
    trajectory = json.loads((run / "evaluator_trajectory.json").read_text())
    alternate_provenance = None
    if args.alternate_replay:
        alternate = args.alternate_replay
        render = json.loads((alternate / "spectator_video_review.json").read_text())
        physics = json.loads((alternate / "audit.json").read_text())
        checks = (
            render["render_checks_passed"], all(render["checks"].values()),
            render["original_video_sha256"] == expected,
            render["original_review_sha256"] == sha(args.review / "record_review.json"),
            Path(render["source_run"]).resolve() == run.resolve(),
            Path(physics["source_run"]).resolve() == run.resolve(),
            physics["physical_reexecution_passed"], physics["replay_source_unchanged"],
            physics["replay_environment_source_matches_recording"],
            physics["steps"] == len(trajectory),
            physics["contact_audit"]["native_environment_contact_control_steps"] == 0,
            render["frames"] == len(frames),
        )
        if not all(checks):
            raise ValueError("Alternate spectator replay lacks matching complete evidence")
        video = alternate / "reviewer_view.mp4"
        expected = render["reframed_video_sha256"]
        if sha(video) != expected:
            raise ValueError("Alternate video differs from its audited file")
        alternate_provenance = dict(spectator_review_sha256=sha(alternate / "spectator_video_review.json"),
                                    physical_audit_sha256=sha(alternate / "audit.json"))
    selected = select_frames(frames, trajectory, review["temporal_evidence"])
    args.output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video))
    provenance = []
    try:
        for label, row in selected.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {label} frame {row['frame']}")
            output = args.output / f"{label}.png"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"Cannot save {output}")
            provenance.append(dict(label=label, source_frame=row["frame"],
                sim_step=row["sim_step"], image_sha256=sha(output)))
    finally:
        capture.release()
    (args.output / "frame_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = dict(video_sha256=expected, record_review_sha256=sha(args.review / "record_review.json"),
        extracted_frames=len(provenance), human_visual_inspection_pending=True,
        alternate_same_controls_replay=alternate_provenance,
        boundary="Unmodified decoded frames. Extraction is not human inspection, a new episode, or a new success verdict.")
    (args.output / "extraction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
