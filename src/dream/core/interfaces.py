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
    # xyz: Optional[np.ndarray] = None
    # camera_in_map_pose: Optional[np.ndarray] = None
    # depth: Optional[np.ndarray] = None
    # camera_K: Optional[np.ndarray] = None

    
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
        if self.xyz is not None and self.camera_in_map_pose is not None:
            return self.transform_points(self.xyz, self.camera_in_map_pose)
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


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseObservations":
        """Create observations from dictionary.
        
        This method dynamically extracts only the fields that exist in the target class,
        making it compatible with all subclasses (ServoObservations, Observations, etc.).
        
        Args:
            data: Dictionary containing observation data
            
        Returns:
            Instance of the calling class with data from the dictionary
        """
        import inspect
        
        # Get the signature of the class constructor
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        
        # Extract only the parameters that exist in the target class
        for param_name in sig.parameters:
            if param_name == 'self':
                continue
            if param_name in data:
                kwargs[param_name] = data[param_name]
        
        return cls(**kwargs)


@dataclass
class RtabmapData(BaseObservations):
    timestamp: float
    compass: np.ndarray
    gps: np.ndarray
    node_id: int
    is_history_node: bool
    rgb_compressed: array.array
    depth_compressed: array.array
    # laser_compressed: array.array
    camera_K: np.ndarray
    pose_graph: Dict[str, np.ndarray]
    local_tf_graph: Dict[str, np.ndarray]
    base_in_map_pose: np.ndarray
    camera_in_map_pose: np.ndarray


@dataclass
class StateObservations(BaseObservations):
    """State observations."""
    gps: np.ndarray  # (x, y) where positive x is forward, positive y is translation to left in meters
    compass: np.ndarray  # positive theta is rotation to left in radians - consistent with robot
    base_in_map_pose: np.ndarray
    arm_base_in_map_pose: np.ndarray
    camera_in_arm_base_pose: np.ndarray
    camera_in_base_pose: np.ndarray
    camera_in_map_pose: np.ndarray
    ee_in_arm_base_pose: np.ndarray
    ee_in_base_pose: np.ndarray
    ee_in_map_pose: np.ndarray
    joint_states: np.ndarray  # joint radians or degrees 
    joint_velocities: np.ndarray
    joint_forces: np.ndarray
    joint_positions: np.ndarray
    at_goal: bool
    is_homed: bool
    is_runstopped: bool
    control_mode: Optional[str] = None
    step: Optional[int] = None
    is_simulation: Optional[bool] = False


@dataclass
class ServoObservations(BaseObservations):
    """Servo observations for visual servoing."""
    rgb: np.ndarray
    depth: np.ndarray
    xyz: Optional[np.ndarray] = None
    camera_K: Optional[np.ndarray] = None
    depth_K: Optional[np.ndarray] = None
    image_scaling: Optional[float] = None
    depth_scaling: Optional[float] = None
    color_shape: Optional[tuple] = None
    depth_shape: Optional[tuple] = None
    # task_observations: Optional[Dict[str, Any]] = None
    is_simulation: bool = False


@dataclass
class Observations(BaseObservations):
    """Full sensor observations with all data."""
    # Core data
    timestamp:float
    compass: np.ndarray
    gps: np.ndarray
    node_id: int
    is_history_node: bool
    rgb: array.array
    depth: array.array
    local_tf_graph: Dict[str, np.ndarray]
    base_in_map_pose: np.ndarray
    camera_in_map_pose: np.ndarray
    pose_graph: Dict[str, np.ndarray]
    camera_K: np.ndarray
    xyz: Optional[np.ndarray] = None
    depth_K: Optional[np.ndarray] = None
    image_scaling: Optional[float] = None
    depth_scaling: Optional[float] = None
    color_shape: Optional[tuple] = None
    depth_shape: Optional[tuple] = None
    recv_address: Optional[str] = None
    step: Optional[int] = None
    at_goal: Optional[bool] = None
    # task_observations: Optional[Dict[str, Any]] = None
    seq_id: int = -1
    is_simulation: Optional[bool] = False