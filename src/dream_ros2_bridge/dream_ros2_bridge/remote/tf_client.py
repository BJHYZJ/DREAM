#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import tf2_ros
import threading
import time
import sophuspy as sp
from typing import Optional, Tuple, Dict
from dream.utils.pose import to_matrix, transform_to_list


class TFClient:
    """独立的TF客户端，专门用于获取TF数据"""
    
    def __init__(self, context=None):
        # 创建独立的TF node
        self._tf_node = rclpy.create_node("tf_client_node", context=context)
        
        # TF QoS设置
        tf_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        static_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        
        # 创建TF buffer和listener
        self.tf2_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=20.0))
        self.tf2_listener = tf2_ros.TransformListener(
            self.tf2_buffer, self._tf_node, spin_thread=False, qos=tf_qos, static_qos=static_qos
        )
        
        # 创建独立的executor
        self._tf_executor = SingleThreadedExecutor()
        self._tf_executor.add_node(self._tf_node)
        
        # 启动TF线程
        self._tf_spin_thread = threading.Thread(target=self._tf_executor.spin, daemon=True)
        self._tf_spin_thread.start()
        
        # TF数据缓存
        self.tf_data = {}
        self._tf_stop_evt = threading.Event()
        self._tf_update_thread = threading.Thread(target=self._tf_update_loop, daemon=True)
        self._tf_update_thread.start()
        
        # 等待TF数据准备就绪
        self._wait_for_tf_ready()
    
    def _wait_for_tf_ready(self):
        """等待TF数据准备就绪"""
        timeout = 10.0  # 10秒超时
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # 检查map->base_link变换是否可用
                if self.tf2_buffer.can_transform("map", "base_link", Time(), Duration(seconds=0.1)):
                    self._tf_node.get_logger().info("TF ready: map -> base_link")
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        
        self._tf_node.get_logger().warn("TF not ready after timeout")
        return False
    
    def _tf_update_loop(self):
        """TF更新循环"""
        tf_mapping = {
            "base_pose": ["base_link", "map"],
            "camera_pose_in_base": ["camera_color_optical_frame", "base_link"],
            "camera_pose_in_map": ["camera_color_optical_frame", "map"],
            "ee_pose_in_map": ["link_eef", "map"],
        }
        
        while not self._tf_stop_evt.is_set():
            for key, (frame, base) in tf_mapping.items():
                try:
                    tr = self.tf2_buffer.lookup_transform(
                        base, frame, Time(), Duration(seconds=0.0)
                    )
                    trans, rot = transform_to_list(tr)
                    timestamp = tr.header.stamp.sec + tr.header.stamp.nanosec / 1e9
                    
                    # 缓存TF数据
                    self.tf_data[key] = {
                        'pose': sp.SE3(to_matrix(trans, rot)),
                        'timestamp': timestamp,
                        'ros_stamp': tr.header.stamp
                    }
                    
                except Exception as e:
                    self._tf_node.get_logger().debug(f"TF lookup failed {base}<-{frame}: {e!r}")
            
            self._tf_stop_evt.wait(0.02)  # 50Hz
    
    def get_base_pose_in_map(self) -> Optional[Tuple[sp.SE3, float]]:
        """获取base在map中的位姿"""
        if "base_pose" in self.tf_data:
            data = self.tf_data["base_pose"]
            return data['pose'], data['timestamp']
        return None
    
    def get_camera_pose_in_base(self) -> Optional[Tuple[sp.SE3, float]]:
        """获取camera在base中的位姿"""
        if "camera_pose_in_base" in self.tf_data:
            data = self.tf_data["camera_pose_in_base"]
            return data['pose'], data['timestamp']
        return None
    
    def get_camera_pose_in_map(self) -> Optional[Tuple[sp.SE3, float]]:
        """获取camera在map中的位姿"""
        if "camera_pose_in_map" in self.tf_data:
            data = self.tf_data["camera_pose_in_map"]
            return data['pose'], data['timestamp']
        return None
    
    def get_ee_pose_in_map(self) -> Optional[Tuple[sp.SE3, float]]:
        """获取末端执行器在map中的位姿"""
        if "ee_pose_in_map" in self.tf_data:
            data = self.tf_data["ee_pose_in_map"]
            return data['pose'], data['timestamp']
        return None
    
    def get_tf_delay(self, key: str) -> Optional[float]:
        """获取TF延迟"""
        if key in self.tf_data:
            current_time = time.time()
            tf_timestamp = self.tf_data[key]['timestamp']
            return current_time - tf_timestamp
        return None
    
    def get_all_tf_data(self) -> Dict:
        """获取所有TF数据"""
        return self.tf_data.copy()
    
    def shutdown(self):
        """关闭TF客户端"""
        # 停止TF更新循环
        if hasattr(self, "_tf_stop_evt"):
            self._tf_stop_evt.set()
        if hasattr(self, "_tf_update_thread") and self._tf_update_thread.is_alive():
            self._tf_update_thread.join(timeout=3.0)
        
        # 停止TF executor
        if hasattr(self, "_tf_executor"):
            try:
                self._tf_executor.shutdown(timeout_sec=2.0)
            except Exception:
                pass
        if hasattr(self, "_tf_spin_thread") and self._tf_spin_thread.is_alive():
            self._tf_spin_thread.join(timeout=3.0)
        
        # 销毁TF node
        if hasattr(self, "_tf_node"):
            try:
                self._tf_node.destroy_node()
            except Exception:
                pass
    
    def __del__(self):
        self.shutdown()


# 全局TF客户端实例
_tf_client = None

def get_tf_client(context=None):
    """获取全局TF客户端实例"""
    global _tf_client
    if _tf_client is None:
        _tf_client = TFClient(context)
    return _tf_client

def shutdown_tf_client():
    """关闭全局TF客户端"""
    global _tf_client
    if _tf_client is not None:
        _tf_client.shutdown()
        _tf_client = None
