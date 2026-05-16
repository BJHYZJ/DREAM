# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import numpy as np
import trimesh.transformations as tra

from dream.motion.robot import RobotModel

from .abstract import AbstractControlModule

MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001

# our robot just has one camera, so we use this class to control it
class DreamCamClient(AbstractControlModule):
    min_depth_val = 0.1
    max_depth_val = 4.0
    camera_frame = "camera_color_optical_frame"

    def __init__(
        self,
        ros_client,
        robot_model: RobotModel,
    ):
        super().__init__()

        self._ros_client = ros_client
        self._robot_model = robot_model

    # Interface methods

    def get_pose_in_map(self):
        """get matrix version of the camera pose"""
        return self._ros_client.get_camera_in_map_pose()  # matrix

    def get_pose_in_base(self):
        """get matrix version of the camera pose"""
        return self._ros_client.get_camera_in_base_pose()  # matrix


    def get_pose_in_arm(self):
        """get matrix version of the camera pose"""
        return self._ros_client.get_camera_in_arm_pose()  # matrix


    def intrinsics(self) -> np.ndarray:
        """Return 3x3 intrinsics matrics"""
        return self._ros_client.rgb_cam.K


    def get_images(self, compute_xyz=False):
        """helper logic to get images from the robot's camera feed"""
        rgb = self._ros_client.rgb_cam.get()
        if self._ros_client.filter_depth:
            dpt = self._ros_client.dpt_cam.get_filtered()
        else:
            dpt = self._process_depth(self._ros_client.dpt_cam.get())

        if rgb is None or dpt is None:
            return None

        # Compute point cloud from depth image
        if compute_xyz:
            xyz = self.depth_to_xyz(dpt)
            imgs = [rgb, dpt, xyz]
        else:
            imgs = [rgb, dpt, None]

        return imgs

    def get_timestamp(self) -> Optional[float]:
        """Return the latest RGB image ROS timestamp in seconds."""
        stamp = self._ros_client.rgb_cam.get_time()
        if stamp is None:
            return None
        return stamp.sec + stamp.nanosec / 1e9

    def depth_to_xyz(self, dpt: np.ndarray) -> np.ndarray:
        """Convert depth to xyz coordinates"""
        xyz = self._ros_client.dpt_cam.depth_to_xyz(self._ros_client.dpt_cam.fix_depth(dpt))
        return xyz

    # Helper methods

    def _process_depth(self, depth):
        # depth[depth < self.min_depth_val] = MIN_DEPTH_REPLACEMENT_VALUE
        # depth[depth > self.max_depth_val] = MAX_DEPTH_REPLACEMENT_VALUE
        return depth

    def _enable_hook(self) -> bool:
        """Dummy override for abstract method"""

    def _disable_hook(self) -> bool:
        """Dummy override for abstract method"""
