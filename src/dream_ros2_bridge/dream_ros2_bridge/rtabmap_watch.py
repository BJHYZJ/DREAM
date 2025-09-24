#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rtabmap_msgs.msg import Info
from typing import Dict, List

class RtabmapMemoryMonitor(Node):
    def __init__(self):
        super().__init__('rtabmap_memory_monitor')
        self.create_subscription(Info, '/rtabmap/info', self.info_callback, 10)
        
        # 记录上一次的状态，用于显示变化
        self.prev_stats = {}
        
        self.get_logger().info("=== RTAB-Map 内存监控器启动 ===")
        self.get_logger().info("监控所有类型的内存节点数量...")

    def info_callback(self, msg):
        # 获取当前参考节点ID（总处理节点数）
        current_node_id = msg.ref_id
        
        # 获取工作内存中的节点ID列表
        wm_state = msg.wm_state
        wm_size = len(wm_state)
        
        # 从统计信息中提取详细信息
        stats_dict = dict(zip(msg.stats_keys, msg.stats_values))
        
        # 提取各种内存大小
        wm_size_stats = stats_dict.get('Memory/Working_memory_size', 0)
        stm_size = stats_dict.get('Memory/Short_time_memory_size', 0)
        local_graph_size = stats_dict.get('Memory/Local_graph_size', 0)
        db_memory_mb = stats_dict.get('Memory/Database_memory_used/MB', 0)
        
        # 计算总内存节点数（当前节点ID就是总处理过的节点数）
        total_nodes = current_node_id
        
        # 计算活跃内存节点数（WM + STM）
        active_memory_nodes = wm_size_stats + stm_size
        
        # 计算数据库中的节点数（总节点 - 活跃节点）
        db_nodes = total_nodes - active_memory_nodes
        
        # 显示内存统计
        self.get_logger().info("=" * 60)
        self.get_logger().info("📊 RTAB-Map 内存状态报告")
        self.get_logger().info("=" * 60)
        
        # 总体统计
        self.get_logger().info(f"🔢 总处理节点数 (Total Nodes): {total_nodes}")
        self.get_logger().info(f"💾 数据库内存使用: {db_memory_mb:.2f} MB")
        
        # 内存分层统计
        self.get_logger().info("")
        self.get_logger().info("🧠 内存分层统计:")
        self.get_logger().info(f"  📝 工作内存 (WM): {wm_size_stats} 个节点")
        self.get_logger().info(f"  ⚡ 短期内存 (STM): {stm_size} 个节点")
        self.get_logger().info(f"  🗄️  数据库存储 (LTM): {db_nodes} 个节点")
        self.get_logger().info(f"  🔄 活跃内存总计: {active_memory_nodes} 个节点")
        
        # 工作内存详细状态
        if wm_state:
            wm_ids = list(wm_state)
            self.get_logger().info("")
            self.get_logger().info(f"🎯 工作内存节点ID: {wm_ids}")
        
        # 局部图信息
        self.get_logger().info("")
        self.get_logger().info("🗺️  局部图信息:")
        self.get_logger().info(f"  📐 局部图大小: {local_graph_size} 个节点")
        
        # 视觉特征统计
        dict_size = stats_dict.get('Keypoint/Dictionary_size/words', 0)
        current_features = stats_dict.get('Keypoint/Current_frame/words', 0)
        indexed_features = stats_dict.get('Keypoint/Indexed_words/words', 0)
        
        self.get_logger().info("")
        self.get_logger().info("👁️  视觉特征统计:")
        self.get_logger().info(f"  📚 视觉词典大小: {dict_size} 个词汇")
        self.get_logger().info(f"  🎬 当前帧特征: {current_features} 个")
        self.get_logger().info(f"  📍 已索引特征: {indexed_features} 个")
        
        # 回环检测统计
        loop_id = stats_dict.get('Loop/Id', 0)
        last_loop_id = stats_dict.get('Loop/Last_id', 0)
        accepted_loops = stats_dict.get('Loop/Accepted_hypothesis_id', 0)
        
        self.get_logger().info("")
        self.get_logger().info("🔄 回环检测统计:")
        self.get_logger().info(f"  🔗 当前回环ID: {loop_id}")
        self.get_logger().info(f"  📍 最后回环ID: {last_loop_id}")
        self.get_logger().info(f"  ✅ 接受的回环: {accepted_loops}")

        rehearsal_merged = stats_dict.get('Memory/Rehearsal_merged', 0)
        rehearsal_sim = stats_dict.get('Memory/Rehearsal_sim', 0)
        rehearsal_id = stats_dict.get('Memory/Rehearsal_id', 0)

        self.get_logger().info("")
        self.get_logger().info(f"  🔄 rehearsal_merged: {rehearsal_merged}")
        self.get_logger().info(f"  🔄 rehearsal_merged: {rehearsal_merged}")
        self.get_logger().info(f"  🔄 rehearsal_id: {rehearsal_id}")    


        # 显示变化趋势
        self.show_changes(stats_dict)
        
        # 更新上一次的状态
        self.prev_stats = stats_dict.copy()
        
        self.get_logger().info("=" * 60)

    def show_changes(self, current_stats: Dict):
        """显示与上一次相比的变化"""
        if not self.prev_stats:
            return
            
        self.get_logger().info("")
        self.get_logger().info("📈 变化趋势:")
        
        # 检查关键指标的变化
        key_metrics = [
            'Memory/Working_memory_size',
            'Memory/Short_time_memory_size', 
            'Memory/Local_graph_size',
            'Keypoint/Dictionary_size/words'
        ]
        
        for metric in key_metrics:
            if metric in current_stats and metric in self.prev_stats:
                current_val = current_stats[metric]
                prev_val = self.prev_stats[metric]
                if current_val != prev_val:
                    change = current_val - prev_val
                    change_str = f"+{change}" if change > 0 else str(change)
                    self.get_logger().info(f"  {metric}: {prev_val} → {current_val} ({change_str})")

def main():
    rclpy.init()
    node = RtabmapMemoryMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断监控")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()