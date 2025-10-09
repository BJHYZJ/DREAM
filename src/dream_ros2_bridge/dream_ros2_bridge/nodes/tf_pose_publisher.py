# Copyright (c) Hello Robot, Inc.
# All rights reserved.

from typing import Optional, Dict
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import tf2_ros
import sophuspy as sp
from dream.utils.pose import to_matrix, transform_to_list
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header


class TFPosePublisher(Node):
    """TF Pose Publisher Node that publishes TF transforms as PoseStamped messages"""
    
    def __init__(self, cache_sec: float = 5.0, use_sim_time: Optional[bool] = None):
        super().__init__('tf_pose_publisher')
        
        if use_sim_time is not None:
            self.set_parameters([rclpy.parameter.Parameter('use_sim_time',
                                                           rclpy.parameter.Parameter.Type.BOOL,
                                                           use_sim_time)])
        
        # TF Buffer and Listener
        self.buffer = tf2_ros.Buffer(cache_time=Duration(seconds=cache_sec))
        self.listener = tf2_ros.TransformListener(self.buffer, self, spin_thread=True)
        
        # TF pose configuration - maps pose keys to frame pairs
        self.tf_poses = {
            "base_pose": ["base_footprint", "map"],
            "camera_pose_in_arm": ["camera_color_optical_frame", "arm_base"],
            "camera_pose_in_base": ["camera_color_optical_frame", "base_footprint"],
            "camera_pose_in_map": ["camera_color_optical_frame", "map"],
            "ee_pose_in_map": ["link_eef", "map"],
        }
        
        # Create publishers for TF pose information
        self._create_pose_publishers()
        
        # Control variables
        self._tf_delay_t = 0.05  # 20Hz update frequency
        
        
        # Start TF update timer
        self._tf_timer = self.create_timer(self._tf_delay_t, self._tf_update_callback)
        
        self.get_logger().info("TF Pose Publisher initialized")
    
    def _create_pose_publishers(self):
        """Create pose publishers"""
        # QoS settings
        qos_profile = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        # Create publishers for each TF pose configuration
        self.pose_publishers = {}
        for pose_key in self.tf_poses.keys():
            topic_name = f"tf_pose/{pose_key}"
            self.pose_publishers[pose_key] = self.create_publisher(
                PoseStamped, topic_name, qos_profile
            )
    
    def _tf_update_callback(self):
        """TF update callback, periodically query TF transforms"""
        for pose_key, frames in self.tf_poses.items():
            frame, base = frames
            try:
                if not self.buffer.can_transform(base, frame, Time(), Duration(seconds=0.0)):
                    continue

                tr = self.buffer.lookup_transform(
                    base, frame, Time(), Duration(seconds=0.1)
                )
                trans, rot = transform_to_list(tr)
                
                # Publish pose to topic
                self._publish_pose_to_topic(pose_key, sp.SE3(to_matrix(trans, rot)), base, frame, tr.header.stamp)
                # if pose_key == "base_pose" and False:
                #     time_now = self.get_clock().now().to_msg()
                #     # Convert to seconds for delay calculation
                #     time_now_sec = time_now.sec + time_now.nanosec / 1e9
                #     tf_time_sec = tr.header.stamp.sec + tr.header.stamp.nanosec / 1e9
                #     delay = time_now_sec - tf_time_sec
                #     self.get_logger().info(f"TF delay: {delay:.3f}s")
                
                    
            except Exception as e:
                self.get_logger().debug(f"TF lookup failed {base}<-{frame}: {e!r}")
    
    def _publish_pose_to_topic(self, pose_key: str, se3_pose: sp.SE3, frame_id: str, child_frame_id: str, timestamp):
        """Publish pose to corresponding topic"""
        if pose_key in self.pose_publishers and se3_pose is not None:
            self._publish_pose(pose_key, se3_pose, frame_id, child_frame_id, timestamp)
    
    def _publish_pose(self, pose_key: str, se3_pose: sp.SE3, frame_id: str, child_frame_id: str, timestamp):
        """Publish pose to corresponding topic"""
        if pose_key in self.pose_publishers:
            msg = PoseStamped()
            msg.header = Header()
            msg.header.stamp = timestamp
            # Include both parent and child frame info in frame_id
            msg.header.frame_id = f"{frame_id}->{child_frame_id}"
            
            # Convert from SE3 to Pose
            matrix = se3_pose.matrix()
            msg.pose.position.x = float(matrix[0, 3])
            msg.pose.position.y = float(matrix[1, 3])
            msg.pose.position.z = float(matrix[2, 3])
            
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation as R
            rotation_matrix = matrix[:3, :3]
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()  # [x, y, z, w]
            msg.pose.orientation.x = float(quat[0])
            msg.pose.orientation.y = float(quat[1])
            msg.pose.orientation.z = float(quat[2])
            msg.pose.orientation.w = float(quat[3])
            
            self.pose_publishers[pose_key].publish(msg)

def main():
    rclpy.init()
    
    tf_pose_publisher = TFPosePublisher()
    rclpy.spin(tf_pose_publisher)

    tf_pose_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()