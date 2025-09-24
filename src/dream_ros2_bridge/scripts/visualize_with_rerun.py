#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import time
from typing import Dict, Tuple, List, Optional

import numpy as np
import cv2
import rerun as rr

# 导入Dynamem的可视化工具
import sys
sys.path.append('/home/yanzj/dream_ws/stretch_ai/src')
from dream.visualization.rerun import RerunVisualizer


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


def collect_node_ids(dump_dir: str) -> List[int]:
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
    right = ci.get('right')
    left = ci.get('left')
    intr = camera_info_from_json(right) or camera_info_from_json(left)
    return intr


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


def visualize_with_rerun(dump_dir: str, use_all_poses: bool = False, stride: int = 4, max_nodes: int = 20):
    """使用Dynamem的Rerun可视化工具显示RTABMap点云"""
    
    # 初始化Rerun
    rr.init("RTABMap Point Cloud Viewer", spawn=True)
    
    # 设置时间轴
    rr.set_time_seconds("realtime", time.time())
    
    # 创建世界坐标系
    rr.log(
        "world/xyz",
        rr.Arrows3D(
            vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )
    
    # 加载数据
    poses_path = os.path.join(dump_dir, 'all_poses.json' if use_all_poses else 'graph_poses.json')
    local_tf_path = os.path.join(dump_dir, 'local_transforms.json')
    
    if not os.path.isfile(poses_path):
        raise FileNotFoundError(f"Missing poses file: {poses_path}")
    
    poses: Dict[str, Dict[str, float]] = load_json(poses_path)
    poses = {int(k): v for k, v in poses.items()}
    
    local_tf: Dict[str, Dict[str, float]] = {}
    if os.path.isfile(local_tf_path):
        local_tf = load_json(local_tf_path)
        local_tf = {int(k): v for k, v in local_tf.items()}
    
    node_ids = collect_node_ids(dump_dir)
    if not node_ids:
        raise RuntimeError(f"No nodes found in {dump_dir}")
    
    print(f"Found {len(node_ids)} nodes, processing up to {max_nodes}...")
    
    # 限制处理的节点数量
    node_ids = node_ids[:max_nodes]
    
    all_points = []
    all_colors = []
    camera_positions = []
    
    # 处理每个节点
    for i, nid in enumerate(node_ids):
        print(f"Processing node {i+1}/{len(node_ids)}: {nid}")
        
        pose = poses.get(nid)
        if pose is None:
            continue
            
        map_T_node = pose_dict_to_matrix(pose)
        node_T_cam = local_tf.get(nid)
        
        # 记录相机位置
        camera_positions.append(map_T_node[:3, 3])
        
        # 获取点云
        pts, cols = gather_points_for_node(dump_dir, nid, map_T_node, node_T_cam, stride=stride)
        if pts.shape[0] == 0:
            continue
            
        all_points.append(pts)
        all_colors.append(cols)
        
        # 实时显示当前节点的点云
        rr.set_time_seconds("realtime", time.time())
        rr.log(
            f"world/nodes/{nid}/points",
            rr.Points3D(
                positions=pts,
                colors=cols,
                radii=0.01
            )
        )
        
        # 显示相机位置
        rr.log(
            f"world/nodes/{nid}/camera",
            rr.Points3D(
                positions=[map_T_node[:3, 3]],
                colors=[[255, 255, 0]],  # 黄色
                radii=0.05
            )
        )
        
        # 显示相机坐标系
        rr.log(
            f"world/nodes/{nid}/camera_frame",
            rr.Transform3D(
                translation=map_T_node[:3, 3],
                mat3x3=map_T_node[:3, :3],
                axis_length=0.2
            )
        )
        
        time.sleep(0.1)  # 稍微延迟以便观察
    
    # 合并所有点云并显示
    if all_points:
        combined_points = np.concatenate(all_points, axis=0)
        combined_colors = np.concatenate(all_colors, axis=0)
        
        print(f"Total points: {len(combined_points)}")
        
        # 显示合并的点云
        rr.set_time_seconds("realtime", time.time())
        rr.log(
            "world/combined_pointcloud",
            rr.Points3D(
                positions=combined_points,
                colors=combined_colors,
                radii=0.005
            )
        )
        
        # 显示所有相机位置
        if camera_positions:
            camera_positions = np.array(camera_positions)
            rr.log(
                "world/camera_trajectory",
                rr.Points3D(
                    positions=camera_positions,
                    colors=[[255, 0, 255]] * len(camera_positions),  # 紫色
                    radii=0.02
                )
            )
            
            # 连接相机位置形成轨迹
            if len(camera_positions) > 1:
                trajectory_lines = []
                for i in range(len(camera_positions) - 1):
                    trajectory_lines.append([camera_positions[i], camera_positions[i+1]])
                
                rr.log(
                    "world/camera_trajectory_lines",
                    rr.LineStrips3D(
                        strips=trajectory_lines,
                        colors=[[255, 0, 255]] * len(trajectory_lines),
                        radii=0.005
                    )
                )
    
    print("Visualization complete! Check the Rerun viewer.")
    print("Press Ctrl+C to exit...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


def main():
    parser = argparse.ArgumentParser(description="Visualize RTABMap point cloud using Dynamem's Rerun")
    parser.add_argument('--dump_dir', type=str, default='./rtabmap_dump', 
                       help='Directory with rgb/depth/camera_info and poses JSONs')
    parser.add_argument('--use_all_poses', action='store_true', 
                       help='Use all_poses.json instead of graph_poses.json')
    parser.add_argument('--stride', type=int, default=2, 
                       help='Subsample pixels by this stride')
    parser.add_argument('--max_nodes', type=int, default=100, 
                       help='Maximum number of nodes to process')
    args = parser.parse_args()
    
    dump_dir = os.path.abspath(args.dump_dir)
    visualize_with_rerun(dump_dir, args.use_all_poses, args.stride, args.max_nodes)


if __name__ == '__main__':
    main()
