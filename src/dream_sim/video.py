"""Encode every recorded frame at 4x playback, without changing task imagery.

Only the time/speed header is redrawn; simulator timestamps remain unchanged.
The input video and the frame-to-simulation-time record must have equal lengths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np

from dream_sim.run import digest


def speed_header(frame: np.ndarray, sim_time: float) -> np.ndarray:
    result = frame.copy()
    # Original composite is RGB (245,247,248); OpenCV decoded video is BGR.
    result[7:33, 746:978] = (248, 247, 245)
    cv2.putText(result, f"{sim_time:.1f} s | 4x playback", (750, 26),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (53, 46, 32), 1, cv2.LINE_AA)
    return result


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
    right = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu_left = cv2.GaussianBlur(left, (11, 11), 1.5)
    mu_right = cv2.GaussianBlur(right, (11, 11), 1.5)
    var_left = cv2.GaussianBlur(left * left, (11, 11), 1.5) - mu_left * mu_left
    var_right = cv2.GaussianBlur(right * right, (11, 11), 1.5) - mu_right * mu_right
    covariance = cv2.GaussianBlur(left * right, (11, 11), 1.5) - mu_left * mu_right
    score = ((2 * mu_left * mu_right + 6.5025) * (2 * covariance + 58.5225)) / (
        (mu_left ** 2 + mu_right ** 2 + 6.5025) * (var_left + var_right + 58.5225))
    return float(score[5:-5, 5:-5].mean())


def encode(source: Path, frames_path: Path, output: Path, *, crf: int = 20) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Choose a new video output; never overwrite an earlier artifact")
    frames = json.loads(frames_path.read_text())
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Cannot open input video: {source}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(capture.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if (width, height) != (1440, 600) or abs(fps - 5) > 1e-6 or count != len(frames):
        raise ValueError("Expected the complete 1440x600, 5fps DREAM composite and matching frame record")
    if any(row["frame"] != index for index, row in enumerate(frames)):
        raise ValueError("Frame record is not complete and ordered")
    if any(abs(frames[index]["sim_time_s"] - frames[index - 1]["sim_time_s"] - 1 / fps) > 1e-6
           for index in range(1, count)):
        raise ValueError("Non-uniform source simulation timeline")
    source_hash, frames_hash = digest(source), digest(frames_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Faststart relocates MP4 byte ranges. Encode and check on local storage,
    # then copy the completed file with a hash check to shared/NFS storage.
    temporary = tempfile.TemporaryDirectory(prefix="dream-video-")
    encoded_path = Path(temporary.name) / "encoded.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "warning", "-n",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
               "-framerate", "20", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "slow",
               "-crf", str(crf), "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-metadata", "comment=4x playback; all source frames preserved; overlay clock is simulation time",
               str(encoded_path)]
    sample_ids = set(np.linspace(0, count - 1, 21, dtype=int).tolist())
    references = {}
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index, row in enumerate(frames):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Source decode stopped at frame {index}")
            frame = speed_header(frame, row["sim_time_s"])
            if index in sample_ids:
                references[index] = frame
            process.stdin.write(frame.tobytes())
        if capture.read()[0]:
            raise ValueError("Input contains unindexed frames")
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Video encoder failed")
    finally:
        capture.release()
        if process.poll() is None:
            process.terminate()
            process.wait()
    encoded = cv2.VideoCapture(str(encoded_path))
    out_count = int(encoded.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = encoded.get(cv2.CAP_PROP_FPS)
    quality = []
    decoded = 0
    while True:
        ok, frame = encoded.read()
        if not ok:
            break
        if decoded in references:
            reference = references[decoded]
            quality.append({"frame": decoded, "full_ssim": ssim(reference, frame),
                            "head_ssim": ssim(reference[60:330, 960:], frame[60:330, 960:]),
                            "map_ssim": ssim(reference[360:600, 960:], frame[360:600, 960:]),
                            "psnr_db": float(cv2.PSNR(reference, frame))})
        decoded += 1
    encoded.release()
    if decoded != count or out_count != count or abs(out_fps - 20) > 1e-6:
        raise ValueError("Encoded frame count or playback rate differs from declared 4x")
    if digest(source) != source_hash or digest(frames_path) != frames_hash:
        raise ValueError("Input changed during export")
    shutil.copyfile(encoded_path, output)
    if digest(output) != digest(encoded_path):
        raise ValueError("Encoded file changed while copying to the destination filesystem")
    temporary.cleanup()
    passed = output.stat().st_size < 90 * 1024 ** 2 and all(
        row["full_ssim"] >= .94 and row["head_ssim"] >= .94 and row["map_ssim"] >= .94 for row in quality)
    report = {
        "source_video_sha256": source_hash, "frame_record_sha256": frames_hash,
        "output_sha256": digest(output), "output_bytes": output.stat().st_size,
        "source_frames": count, "decoded_output_frames": decoded, "resolution": [width, height],
        "source_fps": fps, "output_fps": out_fps, "playback_speed": 4,
        "source_duration_s": count / fps, "output_duration_s": count / out_fps,
        "codec": "h264", "pixel_format": "yuv420p", "preset": "slow", "crf": crf,
        "all_frames_preserved": True, "interpolation": False, "audio_in_source": False,
        "source_inputs_unchanged": True, "edited_region_xyxy": [746, 7, 978, 33],
        "edited_content": "time/speed header only; original simulator timestamps preserved",
        "quality_samples": quality, "sampled_quality_and_size_gate_passed": passed,
        "visual_review_pending": True,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise ValueError(f"Size or sampled-quality gate failed; retain output for review: {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crf", type=int, choices=list(range(16, 23)), default=20)
    args = parser.parse_args()
    print(json.dumps(encode(args.input, args.frames, args.output, crf=args.crf), indent=2))


if __name__ == "__main__":
    main()
