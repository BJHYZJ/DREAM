#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rtabmap_msgs.msg import MapData, MapGraph
import json
import time
from collections import defaultdict

class MapDataGraphComparator(Node):
    def __init__(self):
        super().__init__('mapdata_graph_comparator')
        
        self.declare_parameter('mapdata_topic', '/rtabmap_ranger_xarm/mapData')
        self.declare_parameter('mapgraph_topic', '/rtabmap_ranger_xarm/mapGraph')
        self.declare_parameter('output_file', './comparison_results.json')
        
        self.mapdata_topic = self.get_parameter('mapdata_topic').get_parameter_value().string_value
        self.mapgraph_topic = self.get_parameter('mapgraph_topic').get_parameter_value().string_value
        self.output_file = self.get_parameter('output_file').get_parameter_value().string_value
        
        # 存储最新的消息
        self.latest_mapdata = None
        self.latest_mapgraph = None
        self.comparison_results = []
        
        # QoS设置
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # 订阅两个话题
        self.mapdata_sub = self.create_subscription(
            MapData, self.mapdata_topic, self.mapdata_callback, qos)
        self.mapgraph_sub = self.create_subscription(
            MapGraph, self.mapgraph_topic, self.mapgraph_callback, qos)
        
        # 定时器用于比较
        self.timer = self.create_timer(1.0, self.compare_messages)
        
        self.get_logger().info(f"Comparing {self.mapdata_topic} and {self.mapgraph_topic}")
        self.get_logger().info(f"Results will be saved to {self.output_file}")
        
        self.msg_count = 0
        
    def mapdata_callback(self, msg):
        self.latest_mapdata = msg
        self.get_logger().debug(f"Received MapData: {len(msg.graph.poses_id)} poses, {len(msg.nodes)} nodes")
        
    def mapgraph_callback(self, msg):
        self.latest_mapgraph = msg
        self.get_logger().debug(f"Received MapGraph: {len(msg.poses_id)} poses, {len(msg.links)} links")
        
    def compare_messages(self):
        if self.latest_mapdata is None or self.latest_mapgraph is None:
            return
            
        self.msg_count += 1
        
        # 比较graph部分
        comparison = {
            'timestamp': time.time(),
            'msg_count': self.msg_count,
            'mapdata_header': {
                'stamp': {
                    'sec': self.latest_mapdata.header.stamp.sec,
                    'nanosec': self.latest_mapdata.header.stamp.nanosec
                },
                'frame_id': self.latest_mapdata.header.frame_id
            },
            'mapgraph_header': {
                'stamp': {
                    'sec': self.latest_mapgraph.header.stamp.sec,
                    'nanosec': self.latest_mapgraph.header.stamp.nanosec
                },
                'frame_id': self.latest_mapgraph.header.frame_id
            },
            'graph_comparison': self.compare_graphs()
        }
        
        self.comparison_results.append(comparison)
        
        # 每10次比较输出一次结果
        if self.msg_count % 10 == 0:
            self.print_comparison_summary()
            
    def compare_graphs(self):
        """比较MapData.graph和MapGraph的内容"""
        mapdata_graph = self.latest_mapdata.graph
        mapgraph = self.latest_mapgraph
        
        result = {
            'poses_id_identical': False,
            'poses_identical': False,
            'links_identical': False,
            'map_to_odom_identical': False,
            'poses_id_diff': {},
            'poses_diff': {},
            'links_diff': {},
            'mapdata_poses_count': len(mapdata_graph.poses_id),
            'mapgraph_poses_count': len(mapgraph.poses_id),
            'mapdata_links_count': len(mapdata_graph.links),
            'mapgraph_links_count': len(mapgraph.links)
        }
        
        # 比较poses_id
        if mapdata_graph.poses_id == mapgraph.poses_id:
            result['poses_id_identical'] = True
        else:
            result['poses_id_diff'] = {
                'mapdata_only': list(set(mapdata_graph.poses_id) - set(mapgraph.poses_id)),
                'mapgraph_only': list(set(mapgraph.poses_id) - set(mapdata_graph.poses_id)),
                'common': list(set(mapdata_graph.poses_id) & set(mapgraph.poses_id))
            }
            
        # 比较poses
        if mapdata_graph.poses == mapgraph.poses:
            result['poses_identical'] = True
        else:
            # 比较相同ID的poses
            common_ids = set(mapdata_graph.poses_id) & set(mapgraph.poses_id)
            pose_diffs = []
            for i, pose_id in enumerate(common_ids):
                mapdata_idx = mapdata_graph.poses_id.index(pose_id)
                mapgraph_idx = mapgraph.poses_id.index(pose_id)
                mapdata_pose = mapdata_graph.poses[mapdata_idx]
                mapgraph_pose = mapgraph.poses[mapgraph_idx]
                
                if mapdata_pose != mapgraph_pose:
                    pose_diffs.append({
                        'id': pose_id,
                        'mapdata_pose': {
                            'position': {'x': mapdata_pose.position.x, 'y': mapdata_pose.position.y, 'z': mapdata_pose.position.z},
                            'orientation': {'x': mapdata_pose.orientation.x, 'y': mapdata_pose.orientation.y, 'z': mapdata_pose.orientation.z, 'w': mapdata_pose.orientation.w}
                        },
                        'mapgraph_pose': {
                            'position': {'x': mapgraph_pose.position.x, 'y': mapgraph_pose.position.y, 'z': mapgraph_pose.position.z},
                            'orientation': {'x': mapgraph_pose.orientation.x, 'y': mapgraph_pose.orientation.y, 'z': mapgraph_pose.orientation.z, 'w': mapgraph_pose.orientation.w}
                        }
                    })
            result['poses_diff'] = pose_diffs
            
        # 比较links
        if mapdata_graph.links == mapgraph.links:
            result['links_identical'] = True
        else:
            # 比较links的详细信息
            mapdata_links = {(l.from_id, l.to_id, l.type) for l in mapdata_graph.links}
            mapgraph_links = {(l.from_id, l.to_id, l.type) for l in mapgraph.links}
            
            result['links_diff'] = {
                'mapdata_only': list(mapdata_links - mapgraph_links),
                'mapgraph_only': list(mapgraph_links - mapdata_links),
                'common': list(mapdata_links & mapgraph_links)
            }
            
        # 比较map_to_odom
        if mapdata_graph.map_to_odom == mapgraph.map_to_odom:
            result['map_to_odom_identical'] = True
        else:
            result['map_to_odom_diff'] = {
                'mapdata': {
                    'translation': {'x': mapdata_graph.map_to_odom.translation.x, 'y': mapdata_graph.map_to_odom.translation.y, 'z': mapdata_graph.map_to_odom.translation.z},
                    'rotation': {'x': mapdata_graph.map_to_odom.rotation.x, 'y': mapdata_graph.map_to_odom.rotation.y, 'z': mapdata_graph.map_to_odom.rotation.z, 'w': mapdata_graph.map_to_odom.rotation.w}
                },
                'mapgraph': {
                    'translation': {'x': mapgraph.map_to_odom.translation.x, 'y': mapgraph.map_to_odom.translation.y, 'z': mapgraph.map_to_odom.translation.z},
                    'rotation': {'x': mapgraph.map_to_odom.rotation.x, 'y': mapgraph.map_to_odom.rotation.y, 'z': mapgraph.map_to_odom.rotation.z, 'w': mapgraph.map_to_odom.rotation.w}
                }
            }
            
        return result
        
    def print_comparison_summary(self):
        """打印比较结果摘要"""
        if not self.comparison_results:
            return
            
        latest = self.comparison_results[-1]
        graph_comp = latest['graph_comparison']
        
        self.get_logger().info(f"=== Comparison Summary (Message {self.msg_count}) ===")
        self.get_logger().info(f"Poses ID identical: {graph_comp['poses_id_identical']}")
        self.get_logger().info(f"Poses identical: {graph_comp['poses_identical']}")
        self.get_logger().info(f"Links identical: {graph_comp['links_identical']}")
        self.get_logger().info(f"Map-to-odom identical: {graph_comp['map_to_odom_identical']}")
        
        self.get_logger().info(f"MapData poses: {graph_comp['mapdata_poses_count']}, MapGraph poses: {graph_comp['mapgraph_poses_count']}")
        self.get_logger().info(f"MapData links: {graph_comp['mapdata_links_count']}, MapGraph links: {graph_comp['mapgraph_links_count']}")
        
        if not graph_comp['poses_id_identical']:
            diff = graph_comp['poses_id_diff']
            self.get_logger().warn(f"Poses ID differences: MapData only: {diff['mapdata_only']}, MapGraph only: {diff['mapgraph_only']}")
            
        if not graph_comp['poses_identical']:
            self.get_logger().warn(f"Poses differences: {len(graph_comp['poses_diff'])} different poses")
            
        if not graph_comp['links_identical']:
            diff = graph_comp['links_diff']
            self.get_logger().warn(f"Links differences: MapData only: {len(diff['mapdata_only'])}, MapGraph only: {len(diff['mapgraph_only'])}")
            
    def save_results(self):
        """保存比较结果到文件"""
        with open(self.output_file, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        self.get_logger().info(f"Comparison results saved to {self.output_file}")

def main():
    rclpy.init()
    node = MapDataGraphComparator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Saving results before shutdown...")
        node.save_results()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()