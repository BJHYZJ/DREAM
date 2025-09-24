#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rtabmap_msgs.msg import MapData
from cv_bridge import CvBridge

def pose_to_dict(p):
    return {
        "tx": float(p.position.x), "ty": float(p.position.y), "tz": float(p.position.z),
        "qx": float(p.orientation.x), "qy": float(p.orientation.y),
        "qz": float(p.orientation.z), "qw": float(p.orientation.w)
    }

def transform_to_dict(tf):
    return {
        "tx": float(tf.translation.x), "ty": float(tf.translation.y), "tz": float(tf.translation.z),
        "qx": float(tf.rotation.x), "qy": float(tf.rotation.y),
        "qz": float(tf.rotation.z), "qw": float(tf.rotation.w)
    }

def camera_info_to_dict(ci):
    if ci is None:
        return None
    return {
        "width": int(ci.width), "height": int(ci.height),
        "distortion_model": ci.distortion_model,
        "D": [float(x) for x in ci.d],
        "K": [float(x) for x in ci.k],
        "R": [float(x) for x in ci.r],
        "P": [float(x) for x in ci.p],
        "binning_x": int(ci.binning_x), "binning_y": int(ci.binning_y),
    }

def yaw_from_quat(x, y, z, w):
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

def angle_diff(a, b):
    # wrap to [-pi, pi]
    d = (b - a + math.pi) % (2*math.pi) - math.pi
    return abs(d)

class MapDataExtractor(Node):
    def __init__(self):
        super().__init__('mapdata_extractor_nodes_only')

        self.declare_parameter('mapdata_topic', '/rtabmap_ranger_xarm/mapData')
        self.declare_parameter('out_dir', './rtabmap_dump')
        self.declare_parameter('pose_trans_thresh', 0.05)   # m
        self.declare_parameter('pose_rot_thresh_deg', 1.0)  # deg
        self.declare_parameter('qos_depth', 10)

        self.data_topic = self.get_parameter('mapdata_topic').get_parameter_value().string_value
        self.out_dir = os.path.abspath(self.get_parameter('out_dir').get_parameter_value().string_value)
        self.trans_thresh = float(self.get_parameter('pose_trans_thresh').value)
        self.rot_thresh   = math.radians(float(self.get_parameter('pose_rot_thresh_deg').value))
        qos_depth = int(self.get_parameter('qos_depth').value)

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "camera_info"), exist_ok=True)

        # # link type
        # enum Type {
        #     kNeighbor,           // 0: 相邻节点（里程计边）
        #     kGlobalClosure,      // 1: 全局回环
        #     kLocalSpaceClosure,  // 2: 局部空间回环
        #     kLocalTimeClosure,   // 3: 局部时间回环
        #     kUserClosure,        // 4: 用户定义回环
        #     kVirtualClosure,     // 5: 虚拟回环
        #     kNeighborMerged,     // 6: 合并的相邻节点
        #     kPosePrior,          // 7: 绝对位姿先验
        #     kLandmark,           // 8: 地标
        #     kGravity,            // 9: 重力约束
        #     kEnd
        # };

        self.loop_closure_types = {1, 2, 3, 4} 

        self.bridge = CvBridge()
        self.msg_count = 0

        # 缓存
        self.frames = dict()        # id -> {"stamp":(sec,nsec), "rgb":path, "depth":path}
        self.local_tf = dict()       # id -> transform dict (node->camera), 可选
        self.graph_poses = {}       # id -> dict (当前优化图中的位姿)
        self.all_poses = {}         # id -> dict (所有节点的位姿，永久保存)
        self._prev_graph_poses_msg = {}  # id -> geometry_msgs/Pose (用于阈值比较)
        self.prev_links_set = set() # 上一轮 links 签名

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            durability=DurabilityPolicy.VOLATILE
        )

        self.sub_data = self.create_subscription(MapData, self.data_topic, self.cb_mapdata, qos)

        self.get_logger().info(
            f"Listening: mapData={self.data_topic}, out={self.out_dir}"
        )


    def cb_mapdata(self, msg: MapData):
        self.msg_count += 1
        
        # 1) 保存新增帧载荷（RGB/Depth/CameraInfo/LocalTF）—只保存一次
        for n in msg.nodes:
            if n.id not in self.frames:
                self._save_node_payload_once(n)
        
        # 2) 处理graph信息（位姿和链接）
        # 全量位姿（map->node）覆盖更新
        new_graph_poses = {nid: pose_to_dict(p) for nid, p in zip(msg.graph.poses_id, msg.graph.poses)}

        changed_ids = []
        for nid, p in zip(msg.graph.poses_id, msg.graph.poses):
            if nid in self._prev_graph_poses_msg:
                a = self._prev_graph_poses_msg[nid]
                dt = math.sqrt(
                    (a.position.x - p.position.x) ** 2 +
                    (a.position.y - p.position.y) ** 2 +
                    (a.position.z - p.position.z) ** 2
                )
                ya = yaw_from_quat(a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w)
                yb = yaw_from_quat(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
                dy = angle_diff(ya, yb)
                if dt > self.trans_thresh or dy > self.rot_thresh:
                    changed_ids.append(nid)

        # 更新"上一帧图优化位姿"
        self._prev_graph_poses_msg = {nid: p for nid, p in zip(msg.graph.poses_id, msg.graph.poses)}
        self.graph_poses = new_graph_poses
        
        # 更新所有位姿记录（永久保存，只增加和修改，不删除）
        for nid, pose_dict in new_graph_poses.items():
            self.all_poses[nid] = pose_dict

        # 3) 链接变化检测（新增闭环）
        current_links = {(l.from_id, l.to_id, l.type) for l in msg.graph.links}
        added_links = current_links - self.prev_links_set
        self.prev_links_set = current_links

        loop_happened = False
        loop_added_edges = []
        for a, b, t in added_links:
            if t in self.loop_closure_types:
                loop_happened = True
                loop_added_edges.append((a, b, int(t)))
                self.get_logger().info(f"Loop closure detected: {a} -> {b}, type={t}")

        # 4) 触发输出（事件或周期）
        reason = "loop_detected" if (loop_happened or changed_ids) else "update"
        affected = sorted(set(changed_ids).union({i for e in loop_added_edges for i in e[:2]}))
        extra = {
            "loop_links_added": loop_added_edges,
            "pose_changed_count": len(changed_ids),
            "links_changed": loop_happened,
        }

        self._flush(reason=reason, affected_nodes=affected if affected else None, extra=extra)


    def _save_node_payload_once(self, n):
        nid = n.id
        stamp = (n.data.header.stamp.sec, n.data.header.stamp.nanosec)

        # RGB
        rgb = None
        if n.data.left.data and n.data.left.encoding:
            try:
                rgb = self.bridge.imgmsg_to_cv2(n.data.left, desired_encoding='bgr8')
            except Exception:
                rgb = None
        if rgb is None and n.data.left_compressed:
            try:
                arr = np.frombuffer(bytes(n.data.left_compressed), dtype=np.uint8)
                rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                rgb = None

        # Depth
        depth = None
        if n.data.right.data and n.data.right.encoding:
            try:
                depth = self.bridge.imgmsg_to_cv2(n.data.right)
            except Exception:
                depth = None
        elif n.data.right_compressed:
            try:
                arr = np.frombuffer(bytes(n.data.right_compressed), dtype=np.uint8)
                depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            except Exception:
                depth = None

        # 保存 RGB
        rgb_path = None
        if rgb is not None:
            rgb_path = os.path.join(self.out_dir, "rgb", f"{nid}.png")
            try:
                cv2.imwrite(rgb_path, rgb)
            except Exception as e:
                self.get_logger().warn(f"[id={nid}] Failed to write RGB: {e}")
                rgb_path = None

        depth_path = None
        if depth is not None:
            depth_path = os.path.join(self.out_dir, "depth", f"{nid}.npy")
            try:
                np.save(depth_path, depth)
                if depth.dtype == np.uint16:
                    self.get_logger().info(f"[id={nid}] Saved depth16: {depth.shape}, range={int(depth.min())}-{int(depth.max())}")
                elif depth.dtype == np.uint8:
                    self.get_logger().warning(f"[id={nid}] Saved depth8 (precision lost): {depth.shape}, range={int(depth.min())}-{int(depth.max())}")
                else:
                    self.get_logger().info(f"[id={nid}] Saved depth {depth.dtype}: {depth.shape}")
            except Exception as e:
                self.get_logger().warn(f"[id={nid}] Failed to write depth: {e}")
                depth_path = None


        # CameraInfo（可选）
        left_ci = n.data.left_camera_info[0] if len(n.data.left_camera_info) > 0 else None
        right_ci = n.data.right_camera_info[0] if len(n.data.right_camera_info) > 0 else None
        ci_rec = {"left": camera_info_to_dict(left_ci), "right": camera_info_to_dict(right_ci)}
        try:
            with open(os.path.join(self.out_dir, "camera_info", f"{nid}.json"), "w") as f:
                json.dump(ci_rec, f, indent=2)
        except Exception as e:
                self.get_logger().warn(f"[id={nid}] Failed to write camera info: {e}")


        # Local TF（node->camera），仅保存不使用
        if len(n.data.local_transform) > 0:
            self.local_tf[nid] = transform_to_dict(n.data.local_transform[0])

        self.frames[nid] = {"stamp": stamp, "rgb": rgb_path, "depth": depth_path}
        # self.get_logger().info(f"[id={nid}] saved payload: rgb={bool(rgb_path)}, depth={bool(depth_path)}, cam_info={'Y' if ci_rec['left'] or ci_rec['right'] else 'N'}")


    def _flush(self, reason="update", affected_nodes=None, extra=None):
        # 写出当前优化图位姿快照
        try:
            with open(os.path.join(self.out_dir, "graph_poses.json"), "w") as f:
                json.dump(self.graph_poses, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Failed to write graph_poses.json: {e}")

        # 写出所有位姿记录（永久保存，包含所有曾经出现过的节点）
        try:
            with open(os.path.join(self.out_dir, "all_poses.json"), "w") as f:
                json.dump(self.all_poses, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Failed to write all_poses.json: {e}")

        # 可选写出 local TF（覆盖）
        if self.local_tf:
            try:
                with open(os.path.join(self.out_dir, "local_transforms.json"), "w") as f:
                    json.dump(self.local_tf, f, indent=2)
            except Exception as e:
                self.get_logger().warn(f"Failed to write local_transforms.json: {e}")


        # 事件写入 jsonl（追加），用于后续"受影响节点"追踪
        if affected_nodes is not None:
            rec = {
                "msg_index": self.msg_count,
                "reason": reason,
                "affected_nodes": list(affected_nodes),
                "graph_poses": len(self.graph_poses),
                "all_poses": len(self.all_poses),
                "extra": (extra or {})
            }
            try:
                with open(os.path.join(self.out_dir, "events.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                self.get_logger().warn(f"Failed to append events.jsonl: {e}")
            self.get_logger().info(
                f"Poses updated ({reason}). graph_poses={len(self.graph_poses)}, all_poses={len(self.all_poses)}, affected={len(affected_nodes)}"
            )
        else:
            self.get_logger().info(
                f"Poses updated ({reason}). graph_poses={len(self.graph_poses)}, all_poses={len(self.all_poses)}"
            )

def main():
    rclpy.init()
    node = MapDataExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
