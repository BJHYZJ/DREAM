#!/usr/bin/env python3
"""Controlled RMP replay on public TUM-format RGB-D sequences.

The sensor data and reference camera trajectory are real.  Smooth pose drift
and the subsequent pose-graph correction are deliberately injected so that
map error is measurable against a known reference.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from datasets import DATASETS, associate_sequence, download_dataset


# Small tensor kernels dominate this replay; one thread avoids OpenMP
# oversubscription and makes timing reproducible when datasets run in parallel.
torch.set_num_threads(int(os.environ.get("DREAM_BENCHMARK_THREADS", "1")))


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "6d558e25f045a0a414e2f7ebdfcab5b78940c83d"
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VOXEL_MODULE = _load_module(
    "dream_benchmark_voxel_util",
    REPO_ROOT / "src/dream/mapping/voxel/voxel_util.py",
)
RMP_MODULE = _load_module(
    "dream_benchmark_rmp_policy", REPO_ROOT / "src/dream/mapping/rmp_policy.py"
)
VoxelizedPointcloud = VOXEL_MODULE.VoxelizedPointcloud
select_reintegration_scope = RMP_MODULE.select_reintegration_scope
select_observations_to_prune = RMP_MODULE.select_observations_to_prune


INTRINSICS = {
    "tum_fr1_room": (517.3, 516.5, 318.6, 255.3),
    "tum_fr2_pioneer_slam2": (520.9, 521.0, 325.1, 249.7),
    "bonn_moving_nonobstructing_box": (542.822841, 542.576870, 315.593520, 237.756098),
    "bonn_moving_obstructing_box": (542.822841, 542.576870, 315.593520, 237.756098),
    "bonn_removing_nonobstructing_box": (542.822841, 542.576870, 315.593520, 237.756098),
}


@dataclass
class FrameData:
    depth: np.ndarray
    rgb: np.ndarray
    points_camera: np.ndarray
    features: np.ndarray
    pose_true: np.ndarray
    pose_drifted: np.ndarray


@dataclass
class Result:
    dataset: str
    seed: int
    variant: str
    frames: int
    keyframes: int
    update_scope: str
    reintegrated_frames: int
    chamfer_m: float
    precision_10cm: float
    recall_10cm: float
    fscore_10cm: float
    ghost_voxel_rate: float
    free_space_violation_rate: float
    rgb_feature_residual: float
    active_voxels: int
    retained_observations: int
    retained_keyframes: int
    cache_megabytes: float
    integration_seconds: float
    correction_seconds: float
    source_commit: str = SOURCE_COMMIT


def pose_matrix(values: tuple[float, ...]) -> np.ndarray:
    tx, ty, tz, qx, qy, qz, qw = values
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    qx, qy, qz, qw = quaternion
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = (tx, ty, tz)
    return transform


def add_drift(
    poses: list[np.ndarray], seed: int, max_translation: float, max_yaw_deg: float
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=3)
    direction[2] *= 0.2
    direction /= np.linalg.norm(direction)
    yaw_sign = rng.choice([-1.0, 1.0])
    drifted: list[np.ndarray] = []
    for index, pose in enumerate(poses):
        progress = index / max(1, len(poses) - 1)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        yaw = math.radians(max_yaw_deg * yaw_sign * smooth)
        correction = np.eye(4, dtype=np.float32)
        correction[:3, :3] = np.array(
            [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]],
            dtype=np.float32,
        )
        correction[:3, 3] = direction * max_translation * smooth
        drifted.append(correction @ pose)
    return drifted


def load_frames(
    dataset_name: str,
    sequence_dir: Path,
    seed: int,
    *,
    stride: int,
    max_frames: int | None,
    max_translation: float,
    max_yaw_deg: float,
) -> tuple[list[FrameData], np.ndarray]:
    records = associate_sequence(sequence_dir)
    if max_frames is not None:
        records = records[:max_frames]
    true_poses = [pose_matrix(record.pose) for record in records]
    drifted_poses = add_drift(true_poses, seed, max_translation, max_yaw_deg)
    fx, fy, cx, cy = INTRINSICS[dataset_name]
    intrinsics = np.array(
        [[fx / stride, 0, cx / stride], [0, fy / stride, cy / stride], [0, 0, 1]],
        dtype=np.float32,
    )
    frames: list[FrameData] = []
    for record, true_pose, drifted_pose in zip(records, true_poses, drifted_poses):
        depth_raw = cv2.imread(str(record.depth_path), cv2.IMREAD_UNCHANGED)
        rgb_bgr = cv2.imread(str(record.rgb_path), cv2.IMREAD_COLOR)
        if depth_raw is None or rgb_bgr is None:
            continue
        depth = depth_raw[::stride, ::stride].astype(np.float32) / 5000.0
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)[::stride, ::stride]
        height, width = depth.shape
        yy, xx = np.indices((height, width), dtype=np.float32)
        valid = (depth > 0.25) & (depth < 3.5)
        z = depth[valid]
        points = np.column_stack(
            (
                (xx[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (yy[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            )
        ).astype(np.float32)
        colors = rgb[valid].astype(np.float32)
        features = colors / 255.0
        frames.append(
            FrameData(depth, rgb, points, features, true_pose, drifted_pose)
        )
    return frames, intrinsics


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def integrate_frame(memory, frame: FrameData, pose: np.ndarray, obs_id: int, intrinsics: np.ndarray) -> None:
    if memory.points is not None:
        memory.clear_points(
            depth=torch.from_numpy(frame.depth),
            intrinsics=torch.from_numpy(intrinsics),
            camera_pose=torch.from_numpy(pose),
            min_samples_clear=None,
            depth_in_view_max_distance=3.5,
            depth_in_view_min_distance=0.25,
        )
    points_world = transform_points(frame.points_camera, pose)
    memory.add(
        points=torch.from_numpy(points_world),
        obs_id=obs_id,
        features=torch.from_numpy(frame.features),
        rgb=torch.from_numpy(frame.features * 255.0),
        min_weight_per_voxel=0.0,
    )


def keyframe_ids(frame_count: int, interval: int) -> list[int]:
    ids = list(range(0, frame_count, interval))
    if frame_count and ids[-1] != frame_count - 1:
        ids.append(frame_count - 1)
    return ids


def cache_state(
    frames: list[FrameData],
    keyframes: set[int],
    pruning: bool,
    max_observations: int,
    max_keyframes: int,
    recent_window: int,
) -> tuple[list[int], int]:
    retained = list(range(len(frames)))
    if pruning:
        while len(retained) > max_observations:
            remove_ids = select_observations_to_prune(
                retained,
                is_pose_graph_node=lambda obs_id: obs_id in keyframes,
                max_observations=max_observations,
                max_pose_graph_observations=max_keyframes,
            )
            if not remove_ids:
                break
            removed = set(remove_ids)
            retained = [obs_id for obs_id in retained if obs_id not in removed]

    recent = set(retained[-recent_window:])
    total_bytes = 0
    for obs_id in retained:
        frame = frames[obs_id]
        total_bytes += frame.depth.nbytes + frame.rgb.nbytes + 3 * 16 * 4
        if obs_id in keyframes or obs_id in recent:
            total_bytes += frame.features.nbytes
    return retained, total_bytes


def pose_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    delta = b[:3, :3] @ a[:3, :3].T
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return translation, math.degrees(math.acos(cosine))


def _sample(points: np.ndarray, features: np.ndarray, limit: int = 100_000):
    if len(points) <= limit:
        return points, features
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices], features[indices]


def geometry_metrics(memory, reference, final_frame: FrameData, intrinsics: np.ndarray) -> dict[str, float]:
    points = memory.points.detach().cpu().numpy()
    features = memory.features.detach().cpu().numpy()
    reference_points = reference.points.detach().cpu().numpy()
    reference_features = reference.features.detach().cpu().numpy()
    points, features = _sample(points, features)
    reference_points, reference_features = _sample(reference_points, reference_features)
    tree_reference = cKDTree(reference_points)
    tree_points = cKDTree(points)
    distance_to_reference, nearest_reference = tree_reference.query(points, workers=1)
    distance_to_points, _ = tree_points.query(reference_points, workers=1)
    precision = float(np.mean(distance_to_reference <= 0.10))
    recall = float(np.mean(distance_to_points <= 0.10))
    fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
    feature_mask = distance_to_reference <= 0.10
    feature_residual = float(
        np.mean(
            np.linalg.norm(
                features[feature_mask] - reference_features[nearest_reference[feature_mask]], axis=1
            )
        )
    ) if np.any(feature_mask) else float("nan")

    camera_from_map = np.linalg.inv(final_frame.pose_true)
    camera_points = transform_points(points, camera_from_map)
    positive = camera_points[:, 2] > 0.25
    camera_points = camera_points[positive]
    projected = camera_points @ intrinsics.T
    uv = projected[:, :2] / projected[:, 2:3]
    u = np.rint(uv[:, 0]).astype(int)
    v = np.rint(uv[:, 1]).astype(int)
    height, width = final_frame.depth.shape
    visible = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, camera_points = u[visible], v[visible], camera_points[visible]
    observed_depth = final_frame.depth[v, u]
    valid_depth = (observed_depth > 0.25) & (observed_depth < 3.5)
    violations = camera_points[valid_depth, 2] < observed_depth[valid_depth] - 0.10
    violation_rate = float(np.mean(violations)) if len(violations) else float("nan")
    return {
        "chamfer_m": float((np.mean(distance_to_reference) + np.mean(distance_to_points)) / 2.0),
        "precision_10cm": precision,
        "recall_10cm": recall,
        "fscore_10cm": fscore,
        "ghost_voxel_rate": float(np.mean(distance_to_reference > 0.10)),
        "free_space_violation_rate": violation_rate,
        "rgb_feature_residual": feature_residual,
    }


def run_variant(
    dataset_name: str,
    seed: int,
    variant: str,
    frames: list[FrameData],
    intrinsics: np.ndarray,
    reference,
    *,
    keyframe_interval: int,
    max_observations: int,
    max_keyframes: int,
    translation_threshold: float,
    rotation_threshold_deg: float,
    realtime_window: int,
    short_term_window: int,
    global_fraction: float,
    regional_fraction: float,
) -> tuple[Result, np.ndarray]:
    use_oracle_poses = variant == "oracle"
    apply_reintegration = variant in {"reintegration_only", "full_rmp"}
    apply_pruning = variant in {"pruning_only", "full_rmp"}
    memory = VoxelizedPointcloud(voxel_size=0.05)
    keyframe_list = keyframe_ids(len(frames), keyframe_interval)
    initial_ids = keyframe_list if use_oracle_poses else range(len(frames))
    started = time.perf_counter()
    for obs_id in initial_ids:
        frame = frames[obs_id]
        pose = frame.pose_true if use_oracle_poses else frame.pose_drifted
        integrate_frame(memory, frame, pose, obs_id, intrinsics)
    integration_seconds = time.perf_counter() - started

    keyframes = set(keyframe_list)
    # Match the online agent: decide/reintegrate first, then prune the cache.
    decision_observations = list(range(len(frames)))
    decision_keyframes = keyframe_list
    affected = set()
    for obs_id in decision_keyframes:
        translation, rotation = pose_delta(
            frames[obs_id].pose_drifted, frames[obs_id].pose_true
        )
        if translation > translation_threshold or rotation > rotation_threshold_deg:
            affected.add(obs_id)
    decision = select_reintegration_scope(
        decision_observations,
        decision_keyframes,
        affected,
        realtime_window=realtime_window,
        short_term_window=short_term_window,
        global_fraction=global_fraction,
        regional_fraction=regional_fraction,
        enable_global=True,
        enable_regional=True,
        enable_local=True,
    )
    correction_seconds = 0.0
    reintegrated = 0
    if apply_reintegration and decision.scope != "none":
        correction_started = time.perf_counter()
        if decision.scope == "global":
            memory.reset()
        candidate_ids = set(decision.observation_ids)
        for obs_id in decision_keyframes:
            if obs_id not in candidate_ids:
                continue
            integrate_frame(memory, frames[obs_id], frames[obs_id].pose_true, obs_id, intrinsics)
            reintegrated += 1
        correction_seconds = time.perf_counter() - correction_started

    retained, cache_bytes = cache_state(
        frames,
        keyframes,
        apply_pruning,
        max_observations,
        max_keyframes,
        realtime_window,
    )
    retained_set = set(retained)
    retained_keyframes = [obs_id for obs_id in keyframe_list if obs_id in retained_set]

    metrics = geometry_metrics(memory, reference, frames[-1], intrinsics)
    result = Result(
        dataset=dataset_name,
        seed=seed,
        variant=variant,
        frames=len(frames),
        keyframes=len(keyframe_list),
        update_scope=decision.scope if apply_reintegration else "disabled",
        reintegrated_frames=reintegrated,
        active_voxels=len(memory.points),
        retained_observations=len(retained),
        retained_keyframes=len(retained_keyframes),
        cache_megabytes=cache_bytes / (1024.0 * 1024.0),
        integration_seconds=integration_seconds,
        correction_seconds=correction_seconds,
        **metrics,
    )
    return result, memory.points.detach().cpu().numpy()


def build_reference(frames: list[FrameData], intrinsics: np.ndarray, keyframe_interval: int):
    memory = VoxelizedPointcloud(voxel_size=0.05)
    for obs_id in keyframe_ids(len(frames), keyframe_interval):
        integrate_frame(memory, frames[obs_id], frames[obs_id].pose_true, obs_id, intrinsics)
    return memory


def plot_maps(dataset_name: str, seed: int, clouds: dict[str, np.ndarray], output: Path) -> None:
    variants = [
        variant
        for variant in ("oracle", "no_rmp", "reintegration_only", "pruning_only", "full_rmp")
        if variant in clouds
    ]
    figure, axes = plt.subplots(1, len(variants), figsize=(15, 3), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, variant in zip(axes, variants):
        points = clouds[variant]
        if len(points) > 50_000:
            points = points[np.linspace(0, len(points) - 1, 50_000, dtype=int)]
        axis.scatter(points[:, 0], points[:, 1], s=0.2, c=points[:, 2], cmap="viridis")
        axis.set_title(variant.replace("_", " "))
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"{dataset_name}, injected drift seed {seed}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def append_csv(path: Path, results: list[Result]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        if not exists:
            writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="bonn_removing_nonobstructing_box")
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/rmp"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--keyframe-interval", type=int, default=5)
    parser.add_argument("--max-observations", type=int, default=500)
    parser.add_argument("--max-keyframes", type=int, default=100)
    parser.add_argument("--max-translation", type=float, default=0.35)
    parser.add_argument("--max-yaw-deg", type=float, default=12.0)
    parser.add_argument("--translation-threshold", type=float, default=0.10)
    parser.add_argument("--rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument("--realtime-window", type=int, default=10)
    parser.add_argument("--short-term-window", type=int, default=50)
    parser.add_argument("--global-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--regional-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("oracle", "no_rmp", "reintegration_only", "pruning_only", "full_rmp"),
        default=("oracle", "no_rmp", "reintegration_only", "pruning_only", "full_rmp"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sequence_dir = download_dataset(args.data_root, args.dataset)
    csv_path = args.output_root / f"{args.dataset}.csv"
    if args.overwrite and csv_path.exists():
        csv_path.unlink()
    base_frames, intrinsics = load_frames(
        args.dataset,
        sequence_dir,
        args.seeds[0],
        stride=args.stride,
        max_frames=args.max_frames,
        max_translation=args.max_translation,
        max_yaw_deg=args.max_yaw_deg,
    )
    reference = build_reference(base_frames, intrinsics, args.keyframe_interval)
    all_results: list[Result] = []
    for seed in args.seeds:
        drifted_poses = add_drift(
            [frame.pose_true for frame in base_frames],
            seed,
            args.max_translation,
            args.max_yaw_deg,
        )
        frames = [
            replace(frame, pose_drifted=drifted_pose)
            for frame, drifted_pose in zip(base_frames, drifted_poses)
        ]
        clouds: dict[str, np.ndarray] = {}
        seed_results: list[Result] = []
        for variant in args.variants:
            print(f"Running {args.dataset} seed={seed} variant={variant}", flush=True)
            result, cloud = run_variant(
                args.dataset,
                seed,
                variant,
                frames,
                intrinsics,
                reference,
                keyframe_interval=args.keyframe_interval,
                max_observations=args.max_observations,
                max_keyframes=args.max_keyframes,
                translation_threshold=args.translation_threshold,
                rotation_threshold_deg=args.rotation_threshold_deg,
                realtime_window=args.realtime_window,
                short_term_window=args.short_term_window,
                global_fraction=args.global_fraction,
                regional_fraction=args.regional_fraction,
            )
            seed_results.append(result)
            clouds[variant] = cloud
            print(json.dumps(asdict(result), indent=2), flush=True)
        plot_maps(
            args.dataset,
            seed,
            clouds,
            args.output_root / "figures" / f"{args.dataset}_seed{seed}_maps",
        )
        all_results.extend(seed_results)

    append_csv(csv_path, all_results)
    metadata = {
        "dataset": args.dataset,
        "source_url": DATASETS[args.dataset],
        "source_commit": SOURCE_COMMIT,
        "arguments": vars(args) | {"data_root": str(args.data_root), "output_root": str(args.output_root)},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"{args.dataset}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
