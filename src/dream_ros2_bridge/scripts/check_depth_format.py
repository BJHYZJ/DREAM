#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查深度图数据格式的脚本
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class DepthFormatChecker(Node):
    def __init__(self):
        super().__init__('depth_format_checker')
        
        self.bridge = CvBridge()
        self.count = 0
        
        # 订阅深度图话题
        self.subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',  # 根据你的launch文件中的话题
            self.depth_callback,
            10
        )
        
        self.get_logger().info("Listening to depth topic: /camera/aligned_depth_to_color/image_raw")
        self.get_logger().info("Waiting for depth messages...")

    def depth_callback(self, msg):
        self.count += 1
        
        try:
            # 转换深度图
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            self.get_logger().info(f"=== Depth Message #{self.count} ===")
            self.get_logger().info(f"Original encoding: {msg.encoding}")
            self.get_logger().info(f"Image shape: {depth.shape}")
            self.get_logger().info(f"Image dtype: {depth.dtype}")
            self.get_logger().info(f"Value range: {depth.min()} - {depth.max()}")
            self.get_logger().info(f"Non-zero pixels: {np.count_nonzero(depth)}")
            
            # 检查数据类型
            if depth.dtype == np.uint8:
                self.get_logger().warn("⚠️  Depth is uint8 (8-bit)! This may not be true depth values.")
                self.get_logger().warn("   Expected: uint16 (16-bit) for true depth values")
            elif depth.dtype == np.uint16:
                self.get_logger().info("✅ Depth is uint16 (16-bit) - this looks correct!")
            elif depth.dtype == np.float32:
                self.get_logger().info("✅ Depth is float32 - high precision depth values")
            else:
                self.get_logger().warn(f"⚠️  Unexpected depth dtype: {depth.dtype}")
            
            # 保存一个样本用于检查
            if self.count == 1:
                sample_path = "/home/yanzj/dream_ws/depth_sample.npy"
                np.save(sample_path, depth)
                self.get_logger().info(f"Saved sample depth to: {sample_path}")
                
                # 也保存为PNG用于可视化
                if depth.dtype == np.uint16:
                    # 转换为8位用于可视化
                    depth_vis = (depth / 256).astype(np.uint8)
                    cv2.imwrite("/home/yanzj/dream_ws/depth_sample_vis.png", depth_vis)
                    self.get_logger().info("Saved visualization to: /home/yanzj/dream_ws/depth_sample_vis.png")
                elif depth.dtype == np.uint8:
                    cv2.imwrite("/home/yanzj/dream_ws/depth_sample.png", depth)
                    self.get_logger().info("Saved sample to: /home/yanzj/dream_ws/depth_sample.png")
            
            # 只检查前几个消息
            if self.count >= 3:
                self.get_logger().info("Checked enough samples. Exiting...")
                rclpy.shutdown()
                
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

def main():
    rclpy.init()
    node = DepthFormatChecker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
