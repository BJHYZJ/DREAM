#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import time
import threading
from collections import defaultdict, deque

class TFDiagnosticsNode(Node):
    """
    TF树诊断工具，用于监控TF树的稳定性和性能
    """
    
    def __init__(self):
        super().__init__('tf_diagnostics')
        
        # TF相关
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 监控的变换对 - 只检查实际存在的变换
        self.monitored_transforms = [
            ('map', 'base_link'),
            ('base_link', 'link_eef'),
            ('map', 'link_eef'),
            ('base_link', 'camera_color_optical_frame'),
        ]
        
        # 统计数据
        self.stats = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'last_success_time': None,
            'last_failure_time': None,
            'recent_errors': deque(maxlen=10),
            'timing_data': deque(maxlen=100)
        })
        
        # 创建定时器
        self.timer = self.create_timer(1.0, self.check_transforms)
        self.stats_timer = self.create_timer(10.0, self.print_stats)
        
        self.get_logger().info("TF Diagnostics Node started")
    
    def check_transforms(self):
        """检查所有监控的变换"""
        for parent_frame, child_frame in self.monitored_transforms:
            self.check_single_transform(parent_frame, child_frame)
    
    def check_single_transform(self, parent_frame, child_frame):
        """检查单个变换"""
        transform_name = f"{parent_frame} -> {child_frame}"
        start_time = time.time()
        
        try:
            # 尝试获取变换
            transform = self.tf_buffer.lookup_transform(
                parent_frame, 
                child_frame, 
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            
            # 计算查找时间
            lookup_time = time.time() - start_time
            
            # 更新成功统计
            stats = self.stats[transform_name]
            stats['success_count'] += 1
            stats['last_success_time'] = time.time()
            stats['timing_data'].append(lookup_time)
            
            # 检查变换是否合理
            if self.is_transform_valid(transform):
                self.get_logger().debug(f"✓ {transform_name}: OK (t={lookup_time:.3f}s)")
            else:
                self.get_logger().warn(f"⚠ {transform_name}: Invalid transform detected")
                stats['recent_errors'].append("Invalid transform")
            
        except Exception as e:
            # 更新失败统计
            stats = self.stats[transform_name]
            stats['failure_count'] += 1
            stats['last_failure_time'] = time.time()
            stats['recent_errors'].append(str(e))
            
            self.get_logger().warn(f"✗ {transform_name}: {e}")
    
    def is_transform_valid(self, transform: TransformStamped) -> bool:
        """验证变换是否合理"""
        # 检查位置
        pos = transform.transform.translation
        if abs(pos.x) > 1000 or abs(pos.y) > 1000 or abs(pos.z) > 1000:
            return False
        
        # 检查四元数
        quat = transform.transform.rotation
        norm = (quat.x**2 + quat.y**2 + quat.z**2 + quat.w**2)**0.5
        if abs(norm - 1.0) > 0.1:
            return False
        
        return True
    
    def print_stats(self):
        """打印统计信息"""
        self.get_logger().info("=== TF Diagnostics Report ===")
        
        for transform_name, stats in self.stats.items():
            total_attempts = stats['success_count'] + stats['failure_count']
            if total_attempts == 0:
                continue
                
            success_rate = stats['success_count'] / total_attempts * 100
            
            # 计算平均查找时间
            if stats['timing_data']:
                avg_time = sum(stats['timing_data']) / len(stats['timing_data'])
                max_time = max(stats['timing_data'])
                min_time = min(stats['timing_data'])
            else:
                avg_time = max_time = min_time = 0.0
            
            self.get_logger().info(
                f"{transform_name}:\n"
                f"  Success Rate: {success_rate:.1f}% ({stats['success_count']}/{total_attempts})\n"
                f"  Timing: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s"
            )
            
            # 显示最近的错误
            if stats['recent_errors']:
                self.get_logger().warn(f"  Recent errors: {list(stats['recent_errors'])[-3:]}")
            
            # 检查是否长时间失败
            if stats['last_success_time'] and stats['last_failure_time']:
                if stats['last_failure_time'] > stats['last_success_time']:
                    time_since_success = time.time() - stats['last_success_time']
                    if time_since_success > 30:  # 30秒没有成功
                        self.get_logger().error(f"  ⚠️  No successful lookup for {time_since_success:.1f}s")
        
        self.get_logger().info("=== End Report ===")

def main(args=None):
    rclpy.init(args=args)
    
    node = TFDiagnosticsNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
