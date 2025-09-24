#!/usr/bin/env python3
"""
RTABMap ZMQ Server
基于dynamem架构设计的rtabmap mapData传输服务器
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rtabmap_msgs.msg import MapData
from geometry_msgs.msg import Pose, Transform
import zmq
import json
import numpy as np
import cv2
import zlib
import struct
import time
from typing import Dict, Any, Optional
import threading
from cv_bridge import CvBridge


class RTABMapZmqServer(Node):
    """RTABMap ZMQ服务器，将mapData通过ZMQ传输到指定IP"""
    
    def __init__(self, target_ip: str = "127.0.0.1", port: int = 9999):
        super().__init__('rtabmap_zmq_server')
        
        self.target_ip = target_ip
        self.port = port
        self.bridge = CvBridge()
        
        # ZMQ设置
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        
        # 连接到目标IP
        address = f"tcp://{target_ip}:{port}"
        self.socket.bind(address)
        self.get_logger().info(f"RTABMap ZMQ Server started on {address}")
        
        # 订阅RTABMap MapData话题
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        self.mapdata_subscription = self.create_subscription(
            MapData,
            '/rtabmap_ranger_xarm/mapData',
            self.mapdata_callback,
            qos
        )
        
        # 数据缓存
        self.node_data_cache = {}  # 缓存节点数据，避免重复传输
        self.last_transmission_time = 0
        self.transmission_interval = 0.1  # 100ms传输间隔
        
        self.get_logger().info("RTABMap ZMQ Server initialized")
    
    def mapdata_callback(self, msg: MapData):
        """处理MapData消息"""
        current_time = time.time()
        
        # 控制传输频率
        if current_time - self.last_transmission_time < self.transmission_interval:
            return
        
        self.last_transmission_time = current_time
        
        try:
            # 提取和压缩数据
            compressed_data = self.extract_and_compress_mapdata(msg)
            
            # 通过ZMQ发送
            self.socket.send_pyobj(compressed_data)
            
            self.get_logger().debug(f"Sent MapData with {len(msg.nodes)} nodes to {self.target_ip}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing MapData: {e}")
    
    def extract_and_compress_mapdata(self, mapdata: MapData) -> Dict[str, Any]:
        """提取并压缩MapData"""
        
        # 1. 提取位姿图信息
        graph_data = self.extract_graph_data(mapdata.graph)
        
        # 2. 提取节点数据（只传输新节点）
        node_data = self.extract_node_data(mapdata.nodes)
        
        # 3. 提取地图变换
        map_to_odom = self.extract_transform(mapdata.graph.map_to_odom)
        
        # 4. 组装传输数据
        transmission_data = {
            "timestamp": time.time(),
            "graph": graph_data,
            "nodes": node_data,
            "map_to_odom": map_to_odom,
            "node_count": len(mapdata.nodes),
            "pose_count": len(mapdata.graph.poses)
        }
        
        return transmission_data
    
    def extract_graph_data(self, graph) -> Dict[str, Any]:
        """提取位姿图数据"""
        graph_data = {
            "poses_id": list(graph.poses_id),
            "poses": [],
            "links": []
        }
        
        # 提取位姿
        for pose in graph.poses:
            pose_dict = {
                "position": {
                    "x": float(pose.position.x),
                    "y": float(pose.position.y),
                    "z": float(pose.position.z)
                },
                "orientation": {
                    "x": float(pose.orientation.x),
                    "y": float(pose.orientation.y),
                    "z": float(pose.orientation.z),
                    "w": float(pose.orientation.w)
                }
            }
            graph_data["poses"].append(pose_dict)
        
        # 提取链接
        for link in graph.links:
            link_dict = {
                "from_id": int(link.from_id),
                "to_id": int(link.to_id),
                "type": int(link.type),
                "transform": self.extract_transform(link.transform)
            }
            graph_data["links"].append(link_dict)
        
        return graph_data
    
    def extract_node_data(self, nodes) -> Dict[str, Any]:
        """提取节点数据（只传输新节点）"""
        node_data = {}
        
        for node in nodes:
            node_id = node.id
            
            # 检查是否是新节点
            if node_id not in self.node_data_cache:
                # 提取传感器数据
                sensor_data = self.extract_sensor_data(node.data)
                
                if sensor_data is not None:
                    node_data[node_id] = {
                        "id": node_id,
                        "map_id": node.map_id,
                        "stamp": node.stamp,
                        "label": node.label,
                        "pose": self.extract_pose(node.pose),
                        "sensor_data": sensor_data
                    }
                    
                    # 缓存节点ID
                    self.node_data_cache[node_id] = True
        
        return node_data
    
    def extract_sensor_data(self, sensor_data) -> Optional[Dict[str, Any]]:
        """提取传感器数据并压缩"""
        if sensor_data is None:
            return None
        
        sensor_dict = {}
        
        # 提取RGB图像
        if sensor_data.left:
            try:
                rgb_image = self.bridge.imgmsg_to_cv2(sensor_data.left, "bgr8")
                compressed_rgb = self.compress_image(rgb_image)
                sensor_dict["rgb"] = compressed_rgb
                sensor_dict["rgb_shape"] = rgb_image.shape
            except Exception as e:
                self.get_logger().warn(f"Failed to extract RGB: {e}")
        
        # 提取深度图像
        if sensor_data.right:
            try:
                depth_image = self.bridge.imgmsg_to_cv2(sensor_data.right, "passthrough")
                compressed_depth = self.compress_image(depth_image)
                sensor_dict["depth"] = compressed_depth
                sensor_dict["depth_shape"] = depth_image.shape
            except Exception as e:
                self.get_logger().warn(f"Failed to extract depth: {e}")
        
        # 提取相机信息
        if sensor_data.left_camera_info:
            camera_info = sensor_data.left_camera_info[0]
            sensor_dict["camera_info"] = {
                "width": camera_info.width,
                "height": camera_info.height,
                "distortion_model": camera_info.distortion_model,
                "D": list(camera_info.d),
                "K": list(camera_info.k),
                "R": list(camera_info.r),
                "P": list(camera_info.p)
            }
        
        return sensor_dict if sensor_dict else None
    
    def compress_image(self, image: np.ndarray) -> bytes:
        """压缩图像数据"""
        # 将图像编码为JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        
        # 进一步压缩
        compressed = zlib.compress(encoded_img.tobytes())
        return compressed
    
    def extract_pose(self, pose: Pose) -> Dict[str, Any]:
        """提取位姿信息"""
        return {
            "position": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z)
            },
            "orientation": {
                "x": float(pose.orientation.x),
                "y": float(pose.orientation.y),
                "z": float(pose.orientation.z),
                "w": float(pose.orientation.w)
            }
        }
    
    def extract_transform(self, transform: Transform) -> Dict[str, Any]:
        """提取变换信息"""
        return {
            "translation": {
                "x": float(transform.translation.x),
                "y": float(transform.translation.y),
                "z": float(transform.translation.z)
            },
            "rotation": {
                "x": float(transform.rotation.x),
                "y": float(transform.rotation.y),
                "z": float(transform.rotation.z),
                "w": float(transform.rotation.w)
            }
        }
    
    def cleanup(self):
        """清理资源"""
        self.socket.close()
        self.context.term()
        self.get_logger().info("RTABMap ZMQ Server cleaned up")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='RTABMap ZMQ Server')
    parser.add_argument('--target_ip', default='127.0.0.1', help='Target IP address')
    parser.add_argument('--port', type=int, default=5501, help='ZMQ port')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    try:
        server = RTABMapZmqServer(target_ip=args.target_ip, port=args.port)
        rclpy.spin(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.cleanup()
        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
