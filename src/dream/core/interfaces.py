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


from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union
import array
import numpy as np
import sophuspy as sp


class GeneralTaskState(Enum):
    NOT_STARTED = 0
    PREPPING = 1
    DOING_TASK = 2
    IDLE = 3
    STOP = 4


class Action:
    """Controls."""


class DiscreteNavigationAction(Action, Enum):
    """Discrete navigation controls."""

    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_OBJECT = 4
    PLACE_OBJECT = 5
    NAVIGATION_MODE = 6
    MANIPULATION_MODE = 7
    POST_NAV_MODE = 8
    # Arm extension to a fixed position and height
    EXTEND_ARM = 9
    EMPTY_ACTION = 10
    # Simulation only actions
    SNAP_OBJECT = 11
    DESNAP_OBJECT = 12
    # Discrete gripper commands
    OPEN_GRIPPER = 13
    CLOSE_GRIPPER = 14


class ContinuousNavigationAction(Action):
    xyt: np.ndarray

    def __init__(self, xyt: np.ndarray):
        if not len(xyt) == 3:
            raise RuntimeError("continuous navigation action space has 3 dimensions, x y and theta")
        self.xyt = xyt

    def __str__(self):
        return f"xyt={self.xyt}"


class ContinuousFullBodyAction:
    xyt: np.ndarray
    joints: np.ndarray

    def __init__(self, joints: np.ndarray, xyt: np.ndarray = None):
        """Create full-body continuous action"""
        if xyt is not None and not len(xyt) == 3:
            raise RuntimeError("continuous navigation action space has 3 dimensions, x y and theta")
        self.xyt = xyt
        # Joint states in robot action format
        self.joints = joints


class ContinuousEndEffectorAction:
    pos: np.ndarray
    ori: np.ndarray
    g: np.ndarray
    num_actions: int

    def __init__(
        self,
        pos: np.ndarray = None,
        ori: np.ndarray = None,
        g: np.ndarray = None,
    ):
        """Create end-effector continuous action; moves to 6D pose and activates gripper"""
        if (
            pos is not None
            and ori is not None
            and g is not None
            and not (pos.shape[1] + ori.shape[1] + g.shape[1]) == 8
        ):
            raise RuntimeError(
                "continuous end-effector action space has 8 dimensions: pos=3, ori=4, gripper=1"
            )
        self.pos = pos
        self.ori = ori
        self.g = g
        self.num_actions = pos.shape[0]


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS_NAVIGATION = 1
    CONTINUOUS_MANIPULATION = 2
    CONTINUOUS_EE_MANIPULATION = 3


class HybridAction(Action):
    """Convenience for supporting multiple action types - provides handling to make sure we have the right class at any particular time"""

    action_type: ActionType
    action: Action

    def __init__(
        self,
        action=None,
        xyt: np.ndarray = None,
        joints: np.ndarray = None,
        pos: np.ndarray = None,
        ori: np.ndarray = None,
        gripper: np.ndarray = None,
    ):
        """Make sure that we were passed a useful generic action here. Process it into something useful."""
        if action is not None:
            if isinstance(action, HybridAction):
                self.action_type = action.action_type
            if isinstance(action, DiscreteNavigationAction):
                self.action_type = ActionType.DISCRETE
            elif isinstance(action, ContinuousNavigationAction):
                self.action_type = ActionType.CONTINUOUS_NAVIGATION
            elif isinstance(action, ContinuousEndEffectorAction):
                self.action_type = ActionType.CONTINUOUS_EE_MANIPULATION
            else:
                self.action_type = ActionType.CONTINUOUS_MANIPULATION
        elif joints is not None:
            self.action_type = ActionType.CONTINUOUS_MANIPULATION
            action = ContinuousFullBodyAction(joints, xyt)
        elif xyt is not None:
            self.action_type = ActionType.CONTINUOUS_NAVIGATION
            action = ContinuousNavigationAction(xyt)
        elif pos is not None:
            self.action_type = ActionType.CONTINUOUS_EE_MANIPULATION
            action = ContinuousEndEffectorAction(pos, ori, gripper)
        else:
            raise RuntimeError("Cannot create HybridAction without any action!")
        if isinstance(action, HybridAction):
            # TODO: should we copy like this?
            self.action_type = action.action_type
            action = action.action
            # But more likely this was a mistake so let's actually throw an error
            raise RuntimeError("Do not pass a HybridAction when creating another HybridAction!")
        self.action = action

    def is_discrete(self):
        """Let environment know if we need to handle a discrete action"""
        return self.action_type == ActionType.DISCRETE

    def is_navigation(self):
        return self.action_type == ActionType.CONTINUOUS_NAVIGATION

    def is_manipulation(self):
        return self.action_type in [
            ActionType.CONTINUOUS_MANIPULATION,
            ActionType.CONTINUOUS_EE_MANIPULATION,
        ]

    def get(self):
        """Extract continuous component of the command and return it."""
        if self.action_type == ActionType.DISCRETE:
            return self.action
        elif self.action_type == ActionType.CONTINUOUS_NAVIGATION:
            return self.action.xyt
        elif self.action_type == ActionType.CONTINUOUS_EE_MANIPULATION:
            return self.action.pos, self.action.ori, self.action.g
        else:
            # Extract both the joints and the waypoint target
            return self.action.joints, self.action.xyt


@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray


class BaseObservations:
    """Base class for all observation types with common methods."""
    xyz: Optional[np.ndarray] = None
    camera_pose_in_map: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    camera_K: Optional[np.ndarray] = None

    
    def compute_xyz(self, scaling: float = 1e-3) -> Optional[np.ndarray]:
        """Compute xyz from depth and camera intrinsics."""
        if self.depth is not None and self.camera_K is not None:
            self.xyz = self.depth_to_xyz(self.depth * scaling, self.camera_K)
        return self.xyz

    def depth_to_xyz(self, depth, camera_K) -> np.ndarray:
        """Convert depth image to xyz point cloud."""
        # Get the camera intrinsics
        fx, fy, cx, cy = camera_K[0, 0], camera_K[1, 1], camera_K[0, 2], camera_K[1, 2]
        # Get the image size
        h, w = depth.shape
        # Create the grid
        x = np.tile(np.arange(w), (h, 1))
        y = np.tile(np.arange(h).reshape(-1, 1), (1, w))
        # Compute the xyz
        x = (x - cx) * depth / fx
        y = (y - cy) * depth / fy
        return np.stack([x, y, depth], axis=-1)

    def get_xyz_in_world_frame(self, scaling: float = 1.0) -> Optional[np.ndarray]:
        """Get the xyz in world frame.

        Args:
            scaling: scaling factor for xyz"""
        if self.xyz is None:
            self.compute_xyz(scaling=scaling)
        if self.xyz is not None and self.camera_pose_in_map is not None:
            return self.transform_points(self.xyz, self.camera_pose_in_map)
        return None

    def transform_points(self, points: np.ndarray, pose: Union[np.ndarray, sp.SE3]):
        """Transform points to world frame.
        Args:
            points: points in camera frame
            pose: pose of the camera"""
        if isinstance(pose, sp.SE3):
            pose = pose.matrix()
        assert points.shape[-1] == 3, "Points should be in 3D"
        assert pose.shape == (4, 4), "Pose should be a 4x4 matrix"
        return np.dot(points, pose[:3, :3].T) + pose[:3, 3]


@dataclass
class RtabmapData:
    timestamp: float
    compass: np.ndarray
    gps: np.ndarray
    node_id: int
    rgb_compressed: array.array
    depth_compressed: array.array
    laser_compressed: array.array
    camera_K: np.ndarray
    pose_graph: Dict[str, np.ndarray]
    camera_pose_in_map: np.ndarray


@dataclass
class StateObservations:
    """State observations."""
    gps: np.ndarray  # (x, y) where positive x is forward, positive y is translation to left in meters
    compass: np.ndarray  # positive theta is rotation to left in radians - consistent with robot
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_efforts: np.ndarray
    base_pose_in_map: np.ndarray
    ee_pose_in_map: np.ndarray
    at_goal: bool
    is_homed: bool
    is_runstopped: bool


@dataclass
class ServoObservations(BaseObservations):
    """Servo observations for visual servoing."""
    rgb: np.ndarray
    depth: np.ndarray
    camera_K: Optional[np.ndarray] = None
    depth_K: Optional[np.ndarray] = None
    image_scaling: Optional[float] = None
    depth_scaling: Optional[float] = None
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    ee_pose_in_map: Optional[np.ndarray] = None
    camera_pose_in_map: Optional[np.ndarray] = None
    is_simulation: bool = False


@dataclass
class Observations(BaseObservations):
    """Full sensor observations with all data."""
    # Core data
    gps: np.ndarray
    compass: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray
    xyz: Optional[np.ndarray] = None
    camera_K: Optional[np.ndarray] = None
    camera_pose_in_map: Optional[np.ndarray] = None
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    ee_pose_in_map: Optional[np.ndarray] = None
    lidar_points: Optional[np.ndarray] = None
    lidar_timestamp: Optional[int] = None
    seq_id: int = -1
    is_simulation: bool = False


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observations":
        """Create observations from dictionary."""
        return cls(
            gps=data.get("gps"),
            compass=data.get("compass"),
            rgb=data.get("rgb"),
            depth=data.get("depth"),
            xyz=data.get("xyz"),
            semantic=data.get("semantic"),
            camera_K=data.get("camera_K"),
            camera_pose=data.get("camera_pose"),
            image_scaling=data.get("image_scaling"),
            depth_scaling=data.get("depth_scaling"),
            camera_pose_in_arm=data.get("camera_pose_in_arm"),
            camera_pose_in_base=data.get("camera_pose_in_base"),
            camera_pose_in_map=data.get("camera_pose_in_map"),
            ee_rgb=data.get("ee_rgb"),
            ee_depth=data.get("ee_depth"),
            ee_xyz=data.get("ee_xyz"),
            ee_semantic=data.get("ee_semantic"),
            ee_camera_K=data.get("ee_camera_K"),
            ee_camera_pose=data.get("ee_camera_pose"),
            ee_pose=data.get("ee_pose"),
            ee_pose_in_map=data.get("ee_pose_in_map"),
            instance=data.get("instance"),
            third_person_image=data.get("third_person_image"),
            lidar_points=data.get("lidar_points"),
            lidar_timestamp=data.get("lidar_timestamp"),
            joint_positions=data.get("joint_positions"),
            joint_velocities=data.get("joint_velocities"),
            relative_resting_position=data.get("relative_resting_position"),
            is_holding=data.get("is_holding"),
            task_observations=data.get("task_observations"),
            seq_id=data.get("seq_id"),
            is_simulation=data.get("is_simulation"),
            is_pose_graph_node=data.get("is_pose_graph_node"),
            pose_graph_timestamp=data.get("pose_graph_timestamp"),
            initial_pose_graph_gps=data.get("initial_pose_graph_gps"),
            initial_pose_graph_compass=data.get("initial_pose_graph_compass"),
        )