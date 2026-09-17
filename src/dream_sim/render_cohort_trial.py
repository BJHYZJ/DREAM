"""Render a recorded cohort trial, including unsuccessful trials.

Every control and disturbance is physically replayed. The full timeline is
sampled once per simulated second and shown at 12x. Optional manipulation clips
retain 1x timing. The side panel uses timestamped observations saved by the policy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from dream_sim.io import atomic_json, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manipulation-clips", action="store_true")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    source, record = run / "frozen_workspace/DREAM_code", run / args.name
    review_path = run / "physics" / args.name / "screen_physics_review.json"
    original_result = json.loads((record / "result.json").read_text())
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    if review is not None and review.get("name") != args.name:
        raise ValueError("The independent review belongs to another trial")
    if original_result["evaluator_task_success"] and review is None:
        raise ValueError("A successful trial needs its independent review")
    qualified_success = bool(review and review["task_physics_and_observation_checks_passed"])
    protocol = json.loads((run / "protocol.json").read_text())
    if not all(digest(source / name) == value for name, value in protocol["source_sha256"].items()):
        raise ValueError("Frozen source does not match the recorded protocol")
    inputs = {
        name: digest(record / name)
        for name in (
            "configuration.json",
            "environment_task.json",
            "actions.json",
            "events.jsonl",
            "result.json",
            "evaluator_trajectory.json",
            "disturbance_forces.json",
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
        MS_ASSET_DIR=protocol["asset_root"],
        HF_HUB_CACHE=protocol["model_cache"],
    )
    sys.path[:0] = [str(source / "experiments"), str(source / "src")]
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import replay_instruction_actions as replay
    from instruction_replay_video import spectator_geometry
    from mani_skill.utils import sapien_utils

    cv2.setNumThreads(2)
    trajectory = json.loads((record / "evaluator_trajectory.json").read_text())
    actions = json.loads((record / "actions.json").read_text())
    events = [json.loads(line) for line in (record / "events.jsonl").read_text().splitlines()]
    encoding_directory = tempfile.TemporaryDirectory(prefix="dream-render-")
    video_root = Path(encoding_directory.name)
    keys = ["overview"] + (["grasp", "place"] if args.manipulation_clips else [])
    writers, frames = {}, {key: [] for key in keys}
    observations = sorted(
        (event for event in events if event["event"] == "observation"),
        key=lambda event: event["step"],
    )
    observation_index = 0
    observation_rgb = None
    observation = None
    observation_hashes = {}
    for key in frames:
        writers[key] = imageio.get_writer(
            video_root / f"{key}.mp4",
            fps=12 if key == "overview" else 5,
            codec="libx264",
            quality=None,
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=[
                "-crf",
                "25",
                "-preset",
                "veryfast",
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )
    started = time.monotonic()
    original_io = replay.SimulatorIO

    class VideoIO(original_io):
        def __init__(self, env):
            super().__init__(env)
            self.video_step = 0
            original_step = env.step

            def step_and_render(action):
                result = original_step(action)
                self.video_step += 1
                step = self.video_step
                truth = trajectory[step - 1]
                targets = []
                if step % 20 == 0 or step == len(actions):
                    targets.append("overview")
                if (
                    args.manipulation_clips
                    and truth["task_stage"] in ("grasp", "place")
                    and step % 4 == 0
                ):
                    targets.append(truth["task_stage"])
                if targets:
                    render_frame(self, step, targets)
                return result

            env.step = step_and_render

    def render_frame(io, step, targets):
        nonlocal observation_index, observation_rgb, observation
        while (
            observation_index < len(observations)
            and observations[observation_index]["step"] <= step
        ):
            observation = observations[observation_index]
            observation_index += 1
        if observation is not None and (
            observation_rgb is None or observation_rgb[0] != observation["frame_id"]
        ):
            path = record / f"observation_{observation['frame_id']:05d}.npz"
            with np.load(path) as saved:
                if int(saved["sim_step"]) != observation["step"]:
                    raise ValueError("Observation timestamp differs from the event log")
                observation_rgb = (observation["frame_id"], cv2.resize(saved["rgb"], (320, 240)))
            observation_hashes[path.name] = digest(path)
        truth = trajectory[step - 1]
        pose, tcp = io.pose(), replay.array(io.robot.tcp_pose.p)[0]
        eye, look, fovy = spectator_geometry(
            pose, tcp, io.overview_heading, truth["phase"], "tool-front-placement"
        )
        camera = io.base._human_render_cameras["render_camera"].camera
        camera.set_fovy(fovy)
        camera.set_local_pose(sapien_utils.look_at(eye, look).sp)
        rgb = replay.array(io.base.render_rgb_array("render_camera"))[0].astype(np.uint8)
        for key in targets:
            canvas = np.full((600, 1280, 3), 248, dtype=np.uint8)
            canvas[48:588, :960] = rgb

            def label(text, y, scale=0.43):
                cv2.putText(
                    canvas,
                    text,
                    (972, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (35, 35, 35),
                    1,
                    cv2.LINE_AA,
                )

            label("Saved robot observation", 65)
            if observation_rgb is not None:
                canvas[78:318, 960:1280] = observation_rgb[1]
                label(f"Captured at {observation['step'] / 20:.2f} s", 342)
                label(f"Frame {observation['frame_id']} | fetch_head", 363)
                detections = observation.get("detections", [])
                label("Detections in this observation:", 397)
                for index, detection in enumerate(detections[:4]):
                    label(
                        f"{detection.get('query', 'object')}: {detection.get('score', 0):.2f}",
                        419 + 20 * index,
                    )
                if not detections:
                    label("None", 419)
            else:
                label("No policy observation yet", 130)
            label("Recorded trial outcome:", 530)
            label("Success" if qualified_success else "Failure", 553)
            label("Exact control replay; no new policy run", 577, 0.36)
            speed = 12 if key == "overview" else 1
            title = f"Case {args.name[:2]} | {step / 20:.1f} s robot time | {speed}x playback"
            cv2.putText(
                canvas,
                title,
                (14, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                truth["phase"],
                (14, 41),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            writers[key].append_data(canvas)
            frames[key].append(dict(sim_step=step, sim_time_s=step / 20, phase=truth["phase"]))
            if len(frames[key]) == 1 or step % 4000 == 0:
                cv2.imwrite(str(output / f"{key}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        if step % 1000 == 0:
            print(
                json.dumps(
                    dict(
                        render_step=step,
                        total_steps=len(actions),
                        elapsed_seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )

    replay.SimulatorIO = VideoIO
    sys.argv = [
        str(source / "experiments/replay_instruction_actions.py"),
        "--source-run",
        str(record),
        "--output",
        str(output / "physical"),
    ]
    try:
        replay.main()
    finally:
        for writer in writers.values():
            writer.close()
    physical = json.loads((output / "physical/audit.json").read_text())
    checks = dict(
        physical_reexecution_matches=physical["physical_reexecution_passed"],
        task_outcome_matches=(
            physical["evaluation"]["evaluator_task_success"]
            == original_result["evaluator_task_success"]
        ),
        task_criteria_match=physical["evaluation"]["task_criteria"]
        == original_result["task_criteria"],
        all_controls_replayed=physical["steps"] == len(actions),
        inputs_unchanged=all(digest(record / name) == value for name, value in inputs.items()),
        observations_unchanged=all(
            digest(record / name) == value for name, value in observation_hashes.items()
        ),
        frozen_sources_unchanged=all(
            digest(source / name) == value for name, value in protocol["source_sha256"].items()
        ),
    )
    if not all(checks.values()):
        raise RuntimeError(f"Render replay validation failed: {checks}")
    videos = {}
    for key, timeline in frames.items():
        capture = cv2.VideoCapture(str(video_root / f"{key}.mp4"))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        if not timeline or decoded != len(timeline):
            raise RuntimeError(f"Incomplete video: {key}")
        shutil.copyfile(video_root / f"{key}.mp4", output / f"{key}.mp4")
        if digest(video_root / f"{key}.mp4") != digest(output / f"{key}.mp4"):
            raise OSError("Video changed while copying to the output directory")
        fps = 12 if key == "overview" else 5
        atomic_json(output / f"{key}_frames.json", timeline)
        videos[key] = dict(
            file=f"{key}.mp4",
            frames=decoded,
            fps=fps,
            playback_speed=12 if key == "overview" else 1,
            duration_s=decoded / fps,
            bytes=(output / f"{key}.mp4").stat().st_size,
            sha256=digest(output / f"{key}.mp4"),
        )
    if len(frames["overview"]) != (len(actions) + 19) // 20:
        raise RuntimeError("Overview does not cover the complete timeline")
    receipt = dict(
        case=args.name,
        protocol_sha256=digest(run / "protocol.json"),
        qualified_success=qualified_success,
        observation_sha256=observation_hashes,
        simulation_duration_s=len(actions) / 20,
        all_controls_replayed=True,
        new_policy_execution=False,
        checks=checks,
        independent_review_sha256=digest(review_path) if review is not None else None,
        input_sha256=inputs,
        renderer_sha256=digest(Path(__file__)),
        videos=videos,
        maximum_replay_error=physical["maximum_error"],
        elapsed_seconds=time.monotonic() - started,
    )
    atomic_json(output / "render_receipt.json", receipt)
    encoding_directory.cleanup()
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
