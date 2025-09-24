#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def pose_dict_to_matrix(pose: Dict[str, float]) -> np.ndarray:
    tx, ty, tz = pose["tx"], pose["ty"], pose["tz"]
    qx, qy, qz, qw = pose["qx"], pose["qy"], pose["qz"], pose["qw"]
    # Normalize quaternion
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def camera_info_from_json(ci: Dict) -> Optional[Tuple[float, float, float, float]]:
    if not ci:
        return None
    K = ci.get("K")
    if not K or len(K) < 9:
        return None
    fx, fy = float(K[0]), float(K[4])
    cx, cy = float(K[2]), float(K[5])
    return fx, fy, cx, cy


def write_ply_ascii(path: str, points: np.ndarray, colors: np.ndarray):
    # points: (N,3), colors: (N,3) uint8
    N = points.shape[0]
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def simple_voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        return points, colors
    # Hash by voxel index
    coords = np.floor(points / voxel_size).astype(np.int64)
    # Use dict of first occurrence per voxel
    voxel_dict = {}
    for i, key in enumerate(map(tuple, coords)):
        if key not in voxel_dict:
            voxel_dict[key] = i
    idx = np.fromiter(voxel_dict.values(), dtype=np.int64)
    return points[idx], colors[idx]


def collect_node_ids(dump_dir: str) -> List[int]:
    # Node presence decided by depth .npy and/or rgb .png files
    rgb_dir = os.path.join(dump_dir, 'rgb')
    depth_dir = os.path.join(dump_dir, 'depth')
    rgb_ids = set()
    depth_ids = set()
    if os.path.isdir(rgb_dir):
        for fn in os.listdir(rgb_dir):
            if fn.endswith('.png'):
                try:
                    rgb_ids.add(int(os.path.splitext(fn)[0]))
                except Exception:
                    pass
    if os.path.isdir(depth_dir):
        for fn in os.listdir(depth_dir):
            if fn.endswith('.npy'):
                try:
                    depth_ids.add(int(os.path.splitext(fn)[0]))
                except Exception:
                    pass
    ids = sorted(depth_ids & rgb_ids) if (depth_ids and rgb_ids) else sorted(depth_ids or rgb_ids)
    return ids


def load_intrinsics_for_node(dump_dir: str, node_id: int) -> Optional[Tuple[float, float, float, float]]:
    ci_path = os.path.join(dump_dir, 'camera_info', f'{node_id}.json')
    if not os.path.isfile(ci_path):
        return None
    ci = load_json(ci_path)
    # Prefer right (depth) camera intrinsics when present; fall back to left
    right = ci.get('right')
    left = ci.get('left')
    intr = camera_info_from_json(right) or camera_info_from_json(left)
    return intr


def backproject_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float, stride: int = 1) -> np.ndarray:
    h, w = depth.shape[:2]
    if stride > 1:
        depth = depth[::stride, ::stride]
    # Determine units: if integer depth, assume millimeters; if float, assume meters
    if np.issubdtype(depth.dtype, np.integer):
        z = depth.astype(np.float32) / 1000.0
    else:
        z = depth.astype(np.float32)
    # Mask invalid
    mask = z > 0
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float32)
    yy, xx = np.indices(z.shape, dtype=np.float32)
    if stride > 1:
        xx *= stride
        yy *= stride
    x = (xx - cx) / fx * z
    y = (yy - cy) / fy * z
    pts = np.stack([x[mask], y[mask], z[mask]], axis=1)
    return pts


def gather_points_for_node(dump_dir: str, node_id: int, map_T_node: np.ndarray, node_T_cam: Optional[Dict[str, float]], stride: int) -> Tuple[np.ndarray, np.ndarray]:
    rgb_path = os.path.join(dump_dir, 'rgb', f'{node_id}.png')
    depth_path = os.path.join(dump_dir, 'depth', f'{node_id}.npy')
    if not (os.path.isfile(depth_path) and os.path.isfile(rgb_path)):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    try:
        depth = np.load(depth_path)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    intr = load_intrinsics_for_node(dump_dir, node_id)
    if intr is None:
        # Fallback: approximate pinhole from image size
        h, w = depth.shape[:2]
        fx = fy = max(w, h)
        cx, cy = w / 2.0, h / 2.0
    else:
        fx, fy, cx, cy = intr

    # Backproject
    pts_cam = backproject_depth(depth, fx, fy, cx, cy, stride=stride)
    if pts_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # Colors (match mask used in backprojection)
    # Recompute mask to extract matching RGBs (keep in sync with backprojection logic)
    if np.issubdtype(depth.dtype, np.integer):
        z_full = depth.astype(np.float32) / 1000.0
    else:
        z_full = depth.astype(np.float32)
    mask_full = z_full > 0
    if stride > 1:
        mask = mask_full[::stride, ::stride]
        rgb_s = rgb[::stride, ::stride, :]
    else:
        mask = mask_full
        rgb_s = rgb
    cols = rgb_s[mask]
    if cols.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # Transform to map frame: map_T_cam = map_T_node * node_T_cam
    if node_T_cam is not None:
        map_T_cam = map_T_node @ pose_dict_to_matrix(node_T_cam)
    else:
        map_T_cam = map_T_node
    R = map_T_cam[:3, :3]
    t = map_T_cam[:3, 3]
    pts_map = (R @ pts_cam.T).T + t
    return pts_map.astype(np.float32), cols.astype(np.uint8)[:, ::-1]  # BGR->RGB


def main():
    parser = argparse.ArgumentParser(description="Reconstruct colored point cloud from RGBD + poses dump.")
    parser.add_argument('--dump_dir', type=str, default='./rtabmap_dump', help='Directory with rgb/depth/camera_info and poses JSONs')
    parser.add_argument('--output', type=str, default='map.ply', help='Output PLY path')
    parser.add_argument('--use_all_poses', action='store_true', help='Use all_poses.json instead of graph_poses.json')
    parser.add_argument('--stride', type=int, default=1, help='Subsample pixels by this stride')
    parser.add_argument('--voxel', type=float, default=0, help='Voxel size for downsampling (meters), 0 to disable')
    parser.add_argument('--max_points', type=int, default=5_000_000, help='Hard cap on total points to keep memory in check')
    args = parser.parse_args()

    dump_dir = os.path.abspath(args.dump_dir)
    poses_path = os.path.join(dump_dir, 'all_poses.json' if args.use_all_poses else 'graph_poses.json')
    local_tf_path = os.path.join(dump_dir, 'local_transforms.json')

    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"Missing poses file: {poses_path}")
    poses: Dict[str, Dict[str, float]] = load_json(poses_path)
    # Convert keys to int
    poses = {int(k): v for k, v in poses.items()}

    local_tf: Dict[str, Dict[str, float]] = {}
    if os.path.isfile(local_tf_path):
        local_tf = load_json(local_tf_path)
        local_tf = {int(k): v for k, v in local_tf.items()}

    node_ids = collect_node_ids(dump_dir)
    if not node_ids:
        raise RuntimeError(f"No nodes found in {dump_dir}")

    all_pts: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []

    for nid in node_ids:
        pose = poses.get(nid)
        if pose is None:
            continue
        map_T_node = pose_dict_to_matrix(pose)
        node_T_cam = local_tf.get(nid)  # may be None
        pts, cols = gather_points_for_node(dump_dir, nid, map_T_node, node_T_cam, stride=args.stride)
        if pts.shape[0] == 0:
            continue
        all_pts.append(pts)
        all_cols.append(cols)

    if not all_pts:
        raise RuntimeError("No points reconstructed.")

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)

    # Hard cap
    if points.shape[0] > args.max_points:
        idx = np.random.choice(points.shape[0], args.max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    # Voxel downsample
    points, colors = simple_voxel_downsample(points, colors, args.voxel)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    write_ply_ascii(args.output, points, colors)
    print(f"Wrote {points.shape[0]} points to {args.output}")


if __name__ == '__main__':
    main()


