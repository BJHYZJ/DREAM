# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import threading
import numpy as np
import rclpy
import rclpy.time
from rclpy.time import Time
from livox_ros_driver2.msg import CustomMsg, CustomPoint


class Ros3DLidar(object):
    """Simple wrapper node for Livox 3D LiDAR (Mid-360) - Based on livox_ros_driver2 implementation"""

    def __init__(self, ros_client, name: str = "/livox/lidar", verbose: bool = False):
        self.name = name
        self._points = None
        self._reflectivity = None
        self._tags = None
        self._lines = None
        self._offset_times = None
        self.verbose = verbose
        self._lock = threading.Lock()
        self._t = Time()

        self._ros_client = ros_client
        self._subscriber = self._ros_client.create_subscription(
            CustomMsg, self.name, self._livox_callback, 10
        )
        print(f"Livox Lidar subscriber created for topic: {self.name}")

    def _livox_callback(self, livox_msg):
        """Process incoming Livox custom message - following livox_ros_driver2 format"""
        try:
            # Extract point data from Livox custom message
            # Following the exact format from lddc.cpp FillPointsToCustomMsg()
            points_list = []
            reflectivity_list = []
            tags_list = []
            lines_list = []
            offset_times_list = []
            
            # Process each point following the driver's format
            for point in livox_msg.points:
                # Extract coordinates (x, y, z are in meters)
                points_list.append([point.x, point.y, point.z])
                
                # Extract reflectivity (0-255, same as intensity in PointXyzlt)
                reflectivity_list.append(point.reflectivity)
                
                # Extract tag (Livox tag for point classification)
                tags_list.append(point.tag)
                
                # Extract line (laser line number, 0-5 for Mid-360)
                lines_list.append(point.line)
                
                # Extract offset_time (relative to base_time in microseconds)
                offset_times_list.append(point.offset_time)
            
            # Convert to numpy arrays
            if len(points_list) > 0:
                points_array = np.array(points_list, dtype=np.float32)
                reflectivity_array = np.array(reflectivity_list, dtype=np.uint8)
                tags_array = np.array(tags_list, dtype=np.uint8)
                lines_array = np.array(lines_list, dtype=np.uint8)
                offset_times_array = np.array(offset_times_list, dtype=np.uint32)
                
                # Filter out invalid points following FAST_LIO approach
                valid_mask = np.isfinite(points_array).all(axis=1)
                
                # Filter by distance (remove points too close or too far)
                distances = np.linalg.norm(points_array, axis=1)
                distance_mask = (distances > 0.1) & (distances < 100.0)  # 0.1m to 100m
                
                # Filter by line number (Mid-360 has 6 lines: 0-5)
                line_mask = lines_array < 6
                
                # Filter by tag (following FAST_LIO filtering)
                # Tag filtering: (tag & 0x30) == 0x10 or (tag & 0x30) == 0x00
                tag_mask = ((tags_array & 0x30) == 0x10) | ((tags_array & 0x30) == 0x00)
                
                # Combine all filters
                final_mask = valid_mask & distance_mask & line_mask & tag_mask
                
                if np.any(final_mask):
                    points_array = points_array[final_mask]
                    reflectivity_array = reflectivity_array[final_mask]
                    tags_array = tags_array[final_mask]
                    lines_array = lines_array[final_mask]
                    offset_times_array = offset_times_array[final_mask]
                else:
                    # No valid points
                    points_array = np.array([]).reshape(0, 3)
                    reflectivity_array = np.array([])
                    tags_array = np.array([])
                    lines_array = np.array([])
                    offset_times_array = np.array([])
            else:
                # Empty message
                points_array = np.array([]).reshape(0, 3)
                reflectivity_array = np.array([])
                tags_array = np.array([])
                lines_array = np.array([])
                offset_times_array = np.array([])
            
            if self.verbose:
                print(f"[LIVOX LIDAR] Received {len(points_array)} valid points from {livox_msg.point_num} total")
                print(f"[LIVOX LIDAR] Point cloud shape: {points_array.shape}")
                print(f"[LIVOX LIDAR] Time stamp: {livox_msg.header.stamp}")
                print(f"[LIVOX LIDAR] Timebase: {livox_msg.timebase}")
                print(f"[LIVOX LIDAR] Point num: {livox_msg.point_num}")
                print(f"[LIVOX LIDAR] Lidar ID: {livox_msg.lidar_id}")
                if len(reflectivity_array) > 0:
                    print(f"[LIVOX LIDAR] Reflectivity range: {reflectivity_array.min()} - {reflectivity_array.max()}")
                    print(f"[LIVOX LIDAR] Unique lines: {np.unique(lines_array)}")
                    print(f"[LIVOX LIDAR] Unique tags: {np.unique(tags_array)}")

            with self._lock:
                # Use the header timestamp for consistency (following driver implementation)
                self._t = rclpy.time.Time.from_msg(livox_msg.header.stamp)
                self._points = points_array
                self._reflectivity = reflectivity_array
                self._tags = tags_array
                self._lines = lines_array
                self._offset_times = offset_times_array
                
        except Exception as e:
            print(f"Error processing Livox message: {e}")

    def get_time(self) -> rclpy.time.Time:
        """Get time point cloud was received last"""
        return self._t

    def get(self) -> np.ndarray:
        """Return the contents of the lidar (the last point cloud)"""
        with self._lock:
            return self._points

    def get_with_metadata(self) -> dict:
        """Return point cloud with all metadata - following livox_ros_driver2 format"""
        with self._lock:
            return {
                'points': self._points,
                'reflectivity': self._reflectivity,  # Same as intensity in PointXyzlt
                'tags': self._tags,                  # Livox tag for point classification
                'lines': self._lines,                # Laser line number (0-5 for Mid-360)
                'offset_times': self._offset_times,  # Relative time in microseconds
                'timestamp': self._t
            }

    def get_2d_projection(self, height_threshold: float = 0.1) -> np.ndarray:
        """Get 2D projection of the 3D point cloud for compatibility with existing code"""
        with self._lock:
            if self._points is None or len(self._points) == 0:
                return None
            
            # Filter points within height threshold (remove ground and high objects)
            height_mask = np.abs(self._points[:, 2]) < height_threshold
            filtered_points = self._points[height_mask]
            
            # Project to 2D (x, y)
            if len(filtered_points) > 0:
                return filtered_points[:, :2]  # Return only x, y coordinates
            else:
                return None

    def get_ground_filtered(self, ground_height: float = -0.5, height_threshold: float = 0.1) -> np.ndarray:
        """Get points above ground level"""
        with self._lock:
            if self._points is None or len(self._points) == 0:
                return None
            
            # Filter points above ground
            ground_mask = self._points[:, 2] > (ground_height + height_threshold)
            filtered_points = self._points[ground_mask]
            
            return filtered_points

    def get_by_line(self, line_number: int) -> np.ndarray:
        """Get points from a specific laser line (0-5 for Mid-360)"""
        with self._lock:
            if self._points is None or len(self._points) == 0:
                return None
            
            if self._lines is None:
                return None
            
            # Filter points from specific line
            line_mask = self._lines == line_number
            filtered_points = self._points[line_mask]
            
            return filtered_points

    def get_by_tag(self, tag_value: int) -> np.ndarray:
        """Get points with specific tag value"""
        with self._lock:
            if self._points is None or len(self._points) == 0:
                return None
            
            if self._tags is None:
                return None
            
            # Filter points with specific tag
            tag_mask = self._tags == tag_value
            filtered_points = self._points[tag_mask]
            
            return filtered_points

    def wait_for_scan(self) -> None:
        """Wait for point cloud. Needs to be sort of slow, in order to make sure we give it time
        to update the point cloud in the backend."""
        rate = self._ros_client.create_rate(5)
        while rclpy.ok():
            with self._lock:
                if self._points is not None and len(self._points) > 0:
                    break
            rate.sleep()


if __name__ == "__main__":
    import time
    from dream_ros2_bridge.remote import StretchClient

    rclpy.init()

    client = StretchClient()

    print("Creating a Livox lidar object..")
    time.sleep(5)
    livox_lidar = Ros3DLidar(ros_client=client.get_ros_client(), verbose=True)
    livox_lidar.wait_for_scan()

    print("Livox Lidar ready!")
    points = livox_lidar.get()
    if points is not None:
        print(f"Point cloud shape: {points.shape}")
        
        # Test 2D projection
        points_2d = livox_lidar.get_2d_projection()
        if points_2d is not None:
            print(f"2D projection shape: {points_2d.shape}")
        
        # Test ground filtered
        points_ground = livox_lidar.get_ground_filtered()
        if points_ground is not None:
            print(f"Ground filtered shape: {points_ground.shape}")
        
        # Test metadata
        metadata = livox_lidar.get_with_metadata()
        if len(metadata['reflectivity']) > 0:
            print(f"Reflectivity range: {metadata['reflectivity'].min()} - {metadata['reflectivity'].max()}")
            print(f"Unique tags: {np.unique(metadata['tags'])}")
            print(f"Unique lines: {np.unique(metadata['lines'])}")
        
        # Test line-specific filtering
        for line in range(6):  # Mid-360 has 6 lines
            line_points = livox_lidar.get_by_line(line)
            if line_points is not None and len(line_points) > 0:
                print(f"Line {line} points: {len(line_points)}")

    print("Ending main loop")