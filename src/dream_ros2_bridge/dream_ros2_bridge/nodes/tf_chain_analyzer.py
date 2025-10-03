#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import time
from collections import defaultdict, deque
import numpy as np

class TFChainAnalyzer(Node):
    """
    TF链分析工具，专门分析TF树中变换链的完整性
    """
    
    def __init__(self):
        super().__init__('tf_chain_analyzer')
        
        # TF相关
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 分析目标变换链
        self.target_chains = [
            # 完整链条：map -> base_link -> link_eef
            {
                'name': 'map_to_eef',
                'source': 'map',
                'target': 'link_eef',
                'expected_path': ['map', 'base_link', 'link_eef']
            },
            # 基础变换：map -> base_link
            {
                'name': 'map_to_base',
                'source': 'map', 
                'target': 'base_link',
                'expected_path': ['map', 'base_link']
            },
            # 机械臂变换：base_link -> link_eef
            {
                'name': 'base_to_eef',
                'source': 'base_link',
                'target': 'link_eef', 
                'expected_path': ['base_link', 'link_eef']
            }
        ]
        
        # 统计数据
        self.chain_stats = defaultdict(lambda: {
            'success_count': 0,
            'failure_count': 0,
            'path_lengths': deque(maxlen=50),
            'timing_data': deque(maxlen=50),
            'last_success_time': None,
            'last_failure_time': None,
            'recent_errors': deque(maxlen=10)
        })
        
        # 创建定时器
        self.timer = self.create_timer(2.0, self.analyze_chains)
        self.stats_timer = self.create_timer(15.0, self.print_analysis)
        
        self.get_logger().info("TF Chain Analyzer started")
    
    def analyze_chains(self):
        """分析所有目标变换链"""
        for chain in self.target_chains:
            self.analyze_single_chain(chain)
    
    def analyze_single_chain(self, chain):
        """分析单个变换链"""
        chain_name = chain['name']
        start_time = time.time()
        
        try:
            # 尝试获取变换
            transform = self.tf_buffer.lookup_transform(
                chain['source'],
                chain['target'], 
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # 计算查找时间
            lookup_time = time.time() - start_time
            
            # 更新成功统计
            stats = self.chain_stats[chain_name]
            stats['success_count'] += 1
            stats['last_success_time'] = time.time()
            stats['timing_data'].append(lookup_time)
            
            # 分析变换质量
            quality_score = self.analyze_transform_quality(transform)
            
            self.get_logger().debug(
                f"✓ {chain_name}: OK (t={lookup_time:.3f}s, quality={quality_score:.2f})"
            )
            
        except Exception as e:
            # 更新失败统计
            stats = self.chain_stats[chain_name]
            stats['failure_count'] += 1
            stats['last_failure_time'] = time.time()
            stats['recent_errors'].append(str(e))
            
            # 分析失败原因
            failure_reason = self.analyze_failure_reason(chain, str(e))
            self.get_logger().warn(f"✗ {chain_name}: {failure_reason}")
    
    def analyze_transform_quality(self, transform: TransformStamped) -> float:
        """分析变换质量，返回0-1的分数"""
        score = 1.0
        
        # 检查位置合理性
        pos = transform.transform.translation
        if abs(pos.x) > 100 or abs(pos.y) > 100 or abs(pos.z) > 100:
            score -= 0.3
        
        # 检查四元数归一化
        quat = transform.transform.rotation
        norm = (quat.x**2 + quat.y**2 + quat.z**2 + quat.w**2)**0.5
        if abs(norm - 1.0) > 0.01:
            score -= 0.5
        
        # 检查时间戳新鲜度
        now = self.get_clock().now()
        transform_time = rclpy.time.Time.from_msg(transform.header.stamp)
        age = (now - transform_time).nanoseconds / 1e9  # 转换为秒
        
        if age > 1.0:  # 超过1秒认为过期
            score -= 0.2
        
        return max(0.0, score)
    
    def analyze_failure_reason(self, chain, error_msg: str) -> str:
        """分析失败原因"""
        if "LookupException" in error_msg:
            return f"变换链断裂: {chain['source']} -> {chain['target']}"
        elif "ExtrapolationException" in error_msg:
            return f"时间戳外推失败: {chain['source']} -> {chain['target']}"
        elif "timeout" in error_msg.lower():
            return f"查找超时: {chain['source']} -> {chain['target']}"
        else:
            return f"未知错误: {error_msg}"
    
    def print_analysis(self):
        """打印分析结果"""
        self.get_logger().info("=== TF Chain Analysis Report ===")
        
        for chain in self.target_chains:
            chain_name = chain['name']
            stats = self.chain_stats[chain_name]
            
            total_attempts = stats['success_count'] + stats['failure_count']
            if total_attempts == 0:
                continue
            
            success_rate = stats['success_count'] / total_attempts * 100
            
            # 计算平均查找时间
            if stats['timing_data']:
                avg_time = sum(stats['timing_data']) / len(stats['timing_data'])
                max_time = max(stats['timing_data'])
            else:
                avg_time = max_time = 0.0
            
            self.get_logger().info(
                f"\n{chain_name} ({chain['source']} -> {chain['target']}):\n"
                f"  Success Rate: {success_rate:.1f}% ({stats['success_count']}/{total_attempts})\n"
                f"  Timing: avg={avg_time:.3f}s, max={max_time:.3f}s"
            )
            
            # 显示最近的错误
            if stats['recent_errors']:
                self.get_logger().warn(f"  Recent errors: {list(stats['recent_errors'])[-3:]}")
            
            # 检查链条健康状态
            if stats['last_failure_time'] and stats['last_success_time']:
                if stats['last_failure_time'] > stats['last_success_time']:
                    time_since_success = time.time() - stats['last_success_time']
                    if time_since_success > 10:  # 10秒没有成功
                        self.get_logger().error(f"  ⚠️  Chain broken for {time_since_success:.1f}s")
        
        # 分析整体TF树健康状态
        self.analyze_overall_health()
        
        self.get_logger().info("=== End Analysis ===")
    
    def analyze_overall_health(self):
        """分析整体TF树健康状态"""
        total_chains = len(self.target_chains)
        healthy_chains = 0
        
        for chain in self.target_chains:
            stats = self.chain_stats[chain['name']]
            if stats['success_count'] > 0:
                success_rate = stats['success_count'] / (stats['success_count'] + stats['failure_count'])
                if success_rate > 0.8:  # 80%以上成功率认为健康
                    healthy_chains += 1
        
        health_percentage = (healthy_chains / total_chains) * 100
        
        if health_percentage >= 80:
            self.get_logger().info(f"🌱 Overall TF Health: {health_percentage:.1f}% (Good)")
        elif health_percentage >= 50:
            self.get_logger().warn(f"⚠️  Overall TF Health: {health_percentage:.1f}% (Degraded)")
        else:
            self.get_logger().error(f"🚨 Overall TF Health: {health_percentage:.1f}% (Critical)")

def main(args=None):
    rclpy.init(args=args)
    
    node = TFChainAnalyzer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
