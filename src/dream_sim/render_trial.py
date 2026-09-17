"""Export a qualified trial by replaying every recorded control and force.

The overview samples the complete timeline at 24x. Separate grasp and placement
videos retain 1x timing. Rendering never supplies observations to the controller.
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

from dream_sim.io import atomic_json, configure_vulkan, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audits", type=Path, help="Independent review directory; default: RUN/physics"
    )
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    source, record = run / "frozen_workspace/DREAM_code", run / args.name
    review_path = (args.audits or run / "physics") / args.name / "screen_physics_review.json"
    review = json.loads(review_path.read_text())
    if review.get("name") != args.name:
        raise ValueError("The independent review belongs to another trial")
    if review.get("task_physics_and_observation_checks_passed") is not True:
        raise ValueError("A passing task, observation, and arm-return review is required")
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
    configure_vulkan()
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
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

    from dream_sim.fold_review import review_fold

    cv2.setNumThreads(2)
    trajectory = json.loads((record / "evaluator_trajectory.json").read_text())
    actions = json.loads((record / "actions.json").read_text())
    events = [json.loads(line) for line in (record / "events.jsonl").read_text().splitlines()]
    encoding_directory = tempfile.TemporaryDirectory(prefix="dream-render-")
    video_root = Path(encoding_directory.name)
    writers, frames = {}, {key: [] for key in ("overview", "grasp", "place")}
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
                if step % 40 == 0 or step == len(actions):
                    targets.append("overview")
                if truth["task_stage"] in ("grasp", "place") and step % 4 == 0:
                    targets.append(truth["task_stage"])
                if targets:
                    render_frame(self, step, targets)
                return result

            env.step = step_and_render

    def render_frame(io, step, targets):
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
            canvas = np.full((600, 960, 3), 248, dtype=np.uint8)
            canvas[48:588] = rgb
            speed = 24 if key == "overview" else 1
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
    fold = review_fold(
        events, actions, physical["contact_audit"], physical.get("arm_joint_motion_reexecution")
    )
    checks = dict(
        physical_reexecution_matches=physical["physical_reexecution_passed"],
        reexecuted_task_success=physical["evaluation"]["evaluator_task_success"],
        both_arm_returns_passed=fold["passed"],
        no_robot_self_contact=physical["contact_audit"]["robot_self_contact_control_steps"] == 0,
        inputs_unchanged=all(digest(record / name) == value for name, value in inputs.items()),
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
            playback_speed=24 if key == "overview" else 1,
            duration_s=decoded / fps,
            bytes=(output / f"{key}.mp4").stat().st_size,
            sha256=digest(output / f"{key}.mp4"),
        )
    if len(frames["overview"]) != (len(actions) + 39) // 40:
        raise RuntimeError("Overview does not cover the complete timeline")
    receipt = dict(
        case=args.name,
        simulation_duration_s=len(actions) / 20,
        all_controls_replayed=True,
        new_policy_execution=False,
        checks=checks,
        independent_review_sha256=digest(review_path),
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
