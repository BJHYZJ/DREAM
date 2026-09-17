"""Compose external replay, current head view, saved observation, and semantic heat.

The display uses saved observations and a verified semantic reconstruction.
It never reads scene meshes, evaluator object poses, or future observations.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from dream_sim.io import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import imageio.v2 as imageio
    import numpy as np

    cv2.setNumThreads(2)
    record, render, output = args.record.resolve(), args.render.resolve(), args.output.resolve()
    receipt = json.loads((render / "render_receipt.json").read_text())
    if receipt["case"] != record.name or not all(receipt["checks"].values()):
        raise ValueError("The base render must match this trial and pass replay checks")
    if any(digest(record / name) != value for name, value in receipt["input_sha256"].items()):
        raise ValueError("Original records changed since the physical replay")
    semantic_root = args.semantic.resolve()
    semantic_receipt_path = semantic_root / "semantic_receipt.json"
    semantic_receipt = json.loads(semantic_receipt_path.read_text())
    if (
        semantic_receipt["case"] != record.name
        or not semantic_receipt["all_recorded_memory_counts_match"]
    ):
        raise ValueError("Semantic reconstruction does not match this recording")
    semantic_frames = {row["frame_id"]: row for row in semantic_receipt["frames"]}
    semantic_xy = np.empty((0, 2), np.float32)
    semantic_fields = {}
    semantic_stage = "pickup"
    if (
        receipt.get("protocol_sha256", semantic_receipt["protocol_sha256"])
        != semantic_receipt["protocol_sha256"]
    ):
        raise ValueError("Semantic reconstruction belongs to another protocol")
    if any(
        digest(record / name) != value
        for name, value in semantic_receipt["observation_sha256"].items()
    ):
        raise ValueError("Semantic reconstruction used different observations")
    if any(
        digest(render / f"{key}.mp4") != video["sha256"] for key, video in receipt["videos"].items()
    ):
        raise ValueError("A base video differs from its physical replay receipt")
    head_root = args.head.resolve()
    head_receipt_path = head_root / "head_receipt.json"
    head_receipt = json.loads(head_receipt_path.read_text())
    if head_receipt["case"] != record.name or not all(head_receipt["checks"].values()):
        raise ValueError("Current camera replay did not pass its physical checks")
    if head_receipt["base_render_receipt_sha256"] != digest(render / "render_receipt.json"):
        raise ValueError("Current camera uses a different external-view timeline")
    for key, video in receipt["videos"].items():
        camera = head_receipt["videos"][key]
        if camera["frames"] != video["frames"] or camera["fps"] != video["fps"]:
            raise ValueError("Camera and external-view frame timings differ")
        if digest(head_root / f"{key}.mp4") != camera["sha256"]:
            raise ValueError("Current camera file differs from its replay receipt")
    events = [json.loads(line) for line in (record / "events.jsonl").read_text().splitlines()]
    if any(first["step"] > second["step"] for first, second in zip(events, events[1:])):
        raise ValueError("Event timestamps must be chronological")
    controls = json.loads((record / "actions.json").read_text())
    timeline = {}
    for key in receipt["videos"]:
        for frame in json.loads((render / f"{key}_frames.json").read_text()):
            timeline.setdefault(frame["sim_step"], []).append((key, frame))
    output.mkdir(parents=True, exist_ok=False)
    points = np.empty((0, 3), np.float32)
    colors = np.empty((0, 3), np.uint8)
    rgb = None
    saved_rgb = None
    saved_step = 0
    saved_frame_id = None
    capture_step = 0
    frame_id = None
    target = None
    route = []
    memory_size = 0
    event_index = 0
    observation_hashes = {}
    frame_records = {key: [] for key in receipt["videos"]}

    def integrate(path):
        nonlocal points, colors, rgb, capture_step, frame_id
        with np.load(path) as saved:
            depth = saved["depth_m"]
            color = saved["rgb"]
            intrinsics = saved["intrinsics"]
            transform = saved["camera_to_world_cv"]
            base = saved["base_xyyaw"]
            capture_step = int(saved["sim_step"])
            frame_id = int(saved["frame_id"])
        observation_hashes[path.name] = digest(path)
        rgb = cv2.resize(color, (360, 270))
        if len(points):
            local = (points - transform[:3, 3]) @ transform[:3, :3]
            projected = local @ intrinsics.T
            pixels = np.rint(projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)).astype(int)
            valid = (local[:, 2] > 0.1) & (local[:, 2] <= 4)
            valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < depth.shape[1])
            valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < depth.shape[0])
            ids = np.flatnonzero(valid)
            measured = depth[pixels[ids, 1], pixels[ids, 0]]
            clear = ids[np.isfinite(measured) & (measured > local[ids, 2] + 0.12)]
            keep = np.ones(len(points), bool)
            keep[clear] = False
            points, colors = points[keep], colors[keep]
        yy, xx = np.mgrid[0 : depth.shape[0] : 12, 0 : depth.shape[1] : 12]
        z = depth[::12, ::12]
        local = np.stack(
            (
                (xx - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (yy - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ),
            axis=-1,
        )
        world = local.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
        valid = np.isfinite(world).all(axis=1) & (z.ravel() >= 0.1) & (z.ravel() <= 4)
        valid &= (world[:, 2] >= 0.08) & (world[:, 2] <= 1.65)
        valid &= np.linalg.norm(world[:, :2] - base[:2], axis=1) > 0.34
        points = np.vstack((points, world[valid])).astype(np.float32)
        colors = np.vstack((colors, color[::12, ::12].reshape(-1, 3)[valid]))
        if len(points):
            # Retain the most recent observation in each display voxel.
            _, reverse_ids = np.unique(
                np.floor(points[::-1] / 0.08).astype(np.int32), axis=0, return_index=True
            )
            ids = len(points) - 1 - reverse_ids
            points, colors = points[ids], colors[ids]

    def spatial_view(step):
        base = np.asarray(controls[step - 1]["base_xyyaw"][:2])
        trail = np.asarray([control["base_xyyaw"][:2] for control in controls[:step:20]])
        extent = np.vstack((points[:, :2], semantic_xy, trail, base[None]))
        low, high = extent.min(axis=0) - 0.4, extent.max(axis=0) + 0.4
        center = (low + high) / 2
        scale = min(620 / max(high[0] - low[0], 1), 236 / max(high[1] - low[1], 1))
        canvas = np.full((250, 640, 3), 241, np.uint8)

        def pixels(xy):
            coords = (np.asarray(xy) - center) * [scale, -scale] + [320, 125]
            return np.rint(coords).astype(np.int32)

        if len(points):
            # Higher observed surfaces are drawn last in this top-down view.
            order = np.argsort(points[:, 2])
            uv = pixels(points[order, :2])
            canvas[uv[:, 1], uv[:, 0]] = colors[order]
        if len(semantic_xy):
            scores = semantic_fields[semantic_stage]
            selected = np.isfinite(scores) & (scores >= 0.17)
            uv = pixels(semantic_xy[selected])
            weights = np.clip((scores[selected] - 0.17) / 0.09, 0, 1)
            order = np.argsort(weights)
            uv, weights = uv[order], weights[order, None]
            canvas[uv[:, 1], uv[:, 0]] = (
                canvas[uv[:, 1], uv[:, 0]] * (1 - weights * 0.85)
                + np.array([242, 130, 40]) * weights * 0.85
            ).astype(np.uint8)
        if len(trail) > 1:
            cv2.polylines(canvas, [pixels(trail)], False, (45, 100, 185), 2)
        if len(route) > 1:
            cv2.polylines(canvas, [pixels(route)], False, (30, 160, 100), 1)
        if target is not None:
            cv2.circle(canvas, tuple(pixels(target[:2])), 5, (230, 130, 40), 2)
        cv2.circle(canvas, tuple(pixels(base)), 4, (30, 70, 175), -1)
        return canvas

    with tempfile.TemporaryDirectory(prefix="dream-panels-") as directory:
        local = Path(directory)
        readers = {key: cv2.VideoCapture(str(render / f"{key}.mp4")) for key in receipt["videos"]}
        head_readers = {
            key: cv2.VideoCapture(str(head_root / f"{key}.mp4")) for key in receipt["videos"]
        }
        writers = {
            key: imageio.get_writer(
                local / f"{key}.mp4",
                fps=video["fps"],
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
            for key, video in receipt["videos"].items()
        }
        try:
            for step, streams in sorted(timeline.items()):
                while event_index < len(events) and events[event_index]["step"] <= step:
                    event = events[event_index]
                    name = event["event"]
                    if event.get("task_stage") in ("placement_search", "place", "finished"):
                        semantic_stage = "placement"
                    if name in ("observation", "navigation_depth_update"):
                        prefix = "observation" if name == "observation" else "navigation_depth"
                        path = record / f"{prefix}_{event['frame_id']:05d}.npz"
                        integrate(path)
                        if capture_step != event["step"]:
                            raise ValueError("Saved observation and event timestamps disagree")
                        if name == "observation":
                            saved_rgb = cv2.resize(rgb, (320, 240))
                            saved_step, saved_frame_id = capture_step, frame_id
                            memory_size = event.get("memory", {}).get("after", memory_size)
                            semantic_frame = semantic_frames[event["frame_id"]]
                            semantic_file = semantic_root / semantic_frame["file"]
                            if digest(semantic_file) != semantic_frame["sha256"]:
                                raise ValueError("Semantic frame checksum differs from its receipt")
                            with np.load(semantic_file) as saved:
                                semantic_xy = saved["xy"]
                                semantic_fields = {
                                    key: saved[key] for key in ("pickup", "placement")
                                }
                    elif name in ("memory_retrieval", "verified_visual_track"):
                        detection = event.get("detection")
                        target = detection.get("point_world") if detection else None
                    elif name == "astar_plan":
                        route = event.get("path_xy", [])
                    elif name == "executed_observed_route_chunk":
                        route = []
                    elif name == "observed_memory_update":
                        target = event.get("new_point")
                    elif name == "focused_verification" and (
                        event.get("missing") or event.get("displaced")
                    ):
                        target = None
                    event_index += 1
                map_rgb = spatial_view(step)
                for key, frame in streams:
                    ok, base_frame = readers[key].read()
                    if not ok:
                        raise ValueError("Base video ended before its frame record")
                    head_ok, head_frame = head_readers[key].read()
                    if not head_ok:
                        raise ValueError("Current camera ended before the matching external frame")
                    canvas = np.full((600, 1600, 3), 248, np.uint8)
                    canvas[:, :960] = cv2.cvtColor(base_frame[:, :960], cv2.COLOR_BGR2RGB)
                    canvas[66:306, 960:1280] = cv2.resize(
                        cv2.cvtColor(head_frame, cv2.COLOR_BGR2RGB), (320, 240)
                    )
                    if saved_rgb is not None:
                        canvas[66:306, 1280:1600] = saved_rgb
                    canvas[344:594, 960:] = map_rgb
                    displayed_query = semantic_receipt["queries"][semantic_stage == "placement"]
                    labels = [
                        ("First-person view", 972, 25, 0.5),
                        (f"Head-camera replay | {step / 20:.2f} s", 972, 49, 0.4),
                        ("Saved observation", 1292, 25, 0.5),
                        (
                            f"Captured at {saved_step / 20:.2f} s"
                            if saved_rgb is not None
                            else "No observation yet",
                            1292,
                            49,
                            0.4,
                        ),
                        (
                            f"Semantic memory: {displayed_query} | {memory_size} voxels | alignment 0.17-0.26",
                            972,
                            324,
                            0.4,
                        ),
                        (
                            "Blue: traveled   Green: planned   Orange: queried target",
                            972,
                            341,
                            0.35,
                        ),
                    ]
                    for label, x, y, size in labels:
                        cv2.putText(
                            canvas,
                            label,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            size,
                            (35, 35, 35),
                            1,
                            cv2.LINE_AA,
                        )
                    writers[key].append_data(canvas)
                    frame_records[key].append(
                        dict(
                            frame,
                            observation_step=saved_step,
                            observation_frame_id=saved_frame_id,
                            current_camera_step=step,
                            spatial_depth_step=capture_step,
                            last_event_step=events[event_index - 1]["step"] if event_index else 0,
                        )
                    )
                    if len(frame_records[key]) == 1 or step % 4000 == 0:
                        cv2.imwrite(
                            str(output / f"{key}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                        )
                if step % 4000 == 0:
                    print(
                        json.dumps(
                            dict(case=record.name, panel_step=step, total_steps=len(controls))
                        ),
                        flush=True,
                    )
        finally:
            for writer in writers.values():
                writer.close()
            for reader in [*readers.values(), *head_readers.values()]:
                reader.release()
        videos = {}
        for key, original in receipt["videos"].items():
            capture = cv2.VideoCapture(str(local / f"{key}.mp4"))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            if count != original["frames"] or count != len(frame_records[key]):
                raise ValueError("Panel export frame count differs from the full replay")
            shutil.copyfile(local / f"{key}.mp4", output / f"{key}.mp4")
            if digest(local / f"{key}.mp4") != digest(output / f"{key}.mp4"):
                raise OSError("Panel export changed during copying")
            videos[key] = dict(
                original,
                bytes=(output / f"{key}.mp4").stat().st_size,
                sha256=digest(output / f"{key}.mp4"),
            )
            atomic_json(output / f"{key}_frames.json", frame_records[key])
    if any(digest(record / name) != value for name, value in observation_hashes.items()):
        raise ValueError("An observation changed during rendering")
    receipt.update(
        videos=videos,
        base_render_receipt_sha256=digest(render / "render_receipt.json"),
        panel_renderer_sha256=digest(Path(__file__)),
        observation_sha256=observation_hashes,
        semantic_receipt_sha256=digest(semantic_receipt_path),
        head_receipt_sha256=digest(head_receipt_path),
        panel_description="External replay, synchronous first-person camera replay, saved semantic observation, frozen-encoder semantic heatmap before candidate rejection, and logged target/routes. Newly rendered camera images are for display only and never update task memory.",
    )
    receipt["checks"].update(
        panel_timestamps_not_future=all(
            frame["observation_step"] <= frame["sim_step"]
            and frame["last_event_step"] <= frame["sim_step"]
            for frames in frame_records.values()
            for frame in frames
        ),
        panel_frames_complete=True,
    )
    if not all(receipt["checks"].values()):
        raise ValueError("Panel timeline validation failed")
    atomic_json(output / "render_receipt.json", receipt)
    print(json.dumps(dict(case=record.name, completed=True, videos=len(videos))), flush=True)


if __name__ == "__main__":
    main()
