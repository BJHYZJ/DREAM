"""Render current head-camera images at the exact frames of a verified replay.

The camera is a display observer. Every original control and disturbance is
re-executed; no newly rendered image is passed to the policy or semantic memory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from dream_sim.io import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, render, output = args.run.resolve(), args.render.resolve(), args.output.resolve()
    source, record = run / "frozen_workspace/DREAM_code", run / args.name
    protocol = json.loads((run / "protocol.json").read_text())
    base_receipt = json.loads((render / "render_receipt.json").read_text())
    if base_receipt["case"] != args.name or not all(base_receipt["checks"].values()):
        raise ValueError("The external-view replay must match this trial and pass its checks")
    inputs = base_receipt["input_sha256"]
    if any(digest(record / name) != value for name, value in inputs.items()):
        raise ValueError("Record identity differs from the external-view replay")
    if any(digest(source / name) != value for name, value in protocol["source_sha256"].items()):
        raise ValueError("Frozen source identity differs from the recorded protocol")
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
    import replay_instruction_actions as replay

    cv2.setNumThreads(2)
    original = json.loads((record / "result.json").read_text())
    timeline = {}
    for key in base_receipt["videos"]:
        for frame in json.loads((render / f"{key}_frames.json").read_text()):
            timeline.setdefault(frame["sim_step"], []).append(key)
    output.mkdir(parents=True, exist_ok=False)
    counts = {key: 0 for key in base_receipt["videos"]}
    with tempfile.TemporaryDirectory(prefix="dream-head-") as temporary:
        local = Path(temporary)
        writers = {
            key: imageio.get_writer(
                local / f"{key}.mp4",
                fps=metadata["fps"],
                codec="libx264",
                quality=None,
                macro_block_size=2,
                ffmpeg_log_level="error",
                output_params=[
                    "-crf",
                    "24",
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
            for key, metadata in base_receipt["videos"].items()
        }
        original_io = replay.SimulatorIO

        class HeadViewIO(original_io):
            def __init__(self, env):
                super().__init__(env)
                self.render_step = 0
                original_step = env.step

                def step_and_capture(action):
                    result = original_step(action)
                    self.render_step += 1
                    keys = timeline.get(self.render_step, [])
                    if keys:
                        rgb = self.capture("fetch_head", keyframe=False).rgb
                        for key in keys:
                            writers[key].append_data(rgb)
                            counts[key] += 1
                    return result

                env.step = step_and_capture

        replay.SimulatorIO = HeadViewIO
        sys.argv = [
            "replay_instruction_actions",
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
            task_outcome_matches=physical["evaluation"]["evaluator_task_success"]
            == original["evaluator_task_success"],
            task_criteria_match=physical["evaluation"]["task_criteria"]
            == original["task_criteria"],
            all_controls_replayed=physical["steps"] == original["robot_action_steps"],
            inputs_unchanged=all(digest(record / name) == value for name, value in inputs.items()),
            frozen_sources_unchanged=all(
                digest(source / name) == value for name, value in protocol["source_sha256"].items()
            ),
        )
        if not all(checks.values()):
            raise ValueError(f"Head-view replay failed: {checks}")
        videos = {}
        for key, metadata in base_receipt["videos"].items():
            capture = cv2.VideoCapture(str(local / f"{key}.mp4"))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            if count != metadata["frames"] or count != counts[key]:
                raise ValueError("Camera frames do not match the external-view timeline")
            shutil.copyfile(local / f"{key}.mp4", output / f"{key}.mp4")
            if digest(local / f"{key}.mp4") != digest(output / f"{key}.mp4"):
                raise OSError("Head-camera video changed during copying")
            videos[key] = dict(
                metadata,
                file=f"{key}.mp4",
                sha256=digest(output / f"{key}.mp4"),
                bytes=(output / f"{key}.mp4").stat().st_size,
            )
        atomic_json(
            output / "head_receipt.json",
            dict(
                case=args.name,
                checks=checks,
                input_sha256=inputs,
                protocol_sha256=digest(run / "protocol.json"),
                base_render_receipt_sha256=digest(render / "render_receipt.json"),
                renderer_sha256=digest(Path(__file__)),
                videos=videos,
                maximum_replay_error=physical["maximum_error"],
                new_policy_execution=False,
                head_view_used_by_original_policy=False,
            ),
        )
    print(json.dumps(dict(case=args.name, completed=True, frames=counts)), flush=True)


if __name__ == "__main__":
    main()
