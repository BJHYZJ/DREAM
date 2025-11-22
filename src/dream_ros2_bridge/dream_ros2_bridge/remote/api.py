# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, Iterable, List, Optional
from typing_extensions import Self
import time

import numpy as np
import torch
import trimesh.transformations as tra

from dream.core.interfaces import Observations, RtabmapData, ServoObservations, StateObservations
from dream.core.robot import AbstractRobotClient, ControlMode
from dream.motion import RobotModel
from dream.motion.constants import DreamIdx
from dream.motion.kinematics import RangerxARMKinematics
from dream.utils.geometry import xyt2sophus, posquat2sophus, sophus2xyt, pose2sophus
from dream.motion.constants import ROBOT_JOINTS
from dream.motion.constants import BASE_INDEX, ARM_INDEX, GRIPPER_INDEX

from .modules.cam import DreamCamClient
from .modules.manip import DreamManipulationClient
from .modules.mapping import DreamMappingClient
from .modules.nav import DreamNavigationClient
from .ros import DreamRosInterface

JOINT_POS_TOL = 0.009
JOINT_ANG_TOL = 0.03


def pose_to_dict(p):
    return {
        "tx": float(p.position.x), "ty": float(p.position.y), "tz": float(p.position.z),
        "qx": float(p.orientation.x), "qy": float(p.orientation.y),
        "qz": float(p.orientation.z), "qw": float(p.orientation.w)
    }

def pose_to_sophus(p):
    pos = np.array([float(p.position.x), float(p.position.y), float(p.position.z)])
    quat = np.array([float(p.orientation.x), float(p.orientation.y), 
                     float(p.orientation.z), float(p.orientation.w)])
    return posquat2sophus(pos, quat)

def transform_to_dict(tf):
    return {
        "tx": float(tf.translation.x), "ty": float(tf.translation.y), "tz": float(tf.translation.z),
        "qx": float(tf.rotation.x), "qy": float(tf.rotation.y),
        "qz": float(tf.rotation.z), "qw": float(tf.rotation.w)
    }

def transform_to_sophus(tf):
    pos = np.array([float(tf.translation.x), float(tf.translation.y), float(tf.translation.z)])
    quat = np.array([float(tf.rotation.x), float(tf.rotation.y), 
                     float(tf.rotation.z), float(tf.rotation.w)])
    return posquat2sophus(pos, quat)

def pose_to_list(stamp, p):
    return [
        stamp,
        float(p.position.x), float(p.position.y), float(p.position.z),
        float(p.orientation.x), float(p.orientation.y), float(p.orientation.z), float(p.orientation.w)
    ]


def camera_info_to_dict(ci):
    if ci is None:
        return None
    return {
        "width": int(ci.width), "height": int(ci.height),
        "distortion_model": ci.distortion_model,
        "D": [float(x) for x in ci.d],
        "K": [float(x) for x in ci.k],
        "R": [float(x) for x in ci.r],
        "P": [float(x) for x in ci.p],
        "binning_x": int(ci.binning_x), "binning_y": int(ci.binning_y),
    }


class DreamClient(AbstractRobotClient):
    """Defines a ROS-based interface to the real Stretch robot. Collect observations and command the robot."""

    head_camera_frame = "camera_color_optical_frame"
    world_frame = "map"

    def __init__(
        self,
        camera_overrides: Optional[Dict] = None,
    ):
        """Create an interface into ROS execution here. This one needs to connect to:
            - joint_states to read current position
            - tf for SLAM
            - FollowJointTrajectory for arm motions

        Based on this code:
        https://github.com/hello-robot/stretch_ros/blob/master/hello_helpers/src/hello_helpers/hello_misc.py
        """

        if camera_overrides is None:
            camera_overrides = {}
        self._ros_client = DreamRosInterface(init_lidar=True, **camera_overrides)

        self._robot_model = RangerxARMKinematics()

        # Interface modules
        self.nav = DreamNavigationClient(self._ros_client, self._robot_model)
        self.manip = DreamManipulationClient(self._ros_client, self._robot_model)
        self.cam = DreamCamClient(self._ros_client, self._robot_model)
        self.mapping = DreamMappingClient(self._ros_client)

        # Init control mode
        self._base_control_mode = ControlMode.IDLE

        # Initially start in navigation mode all the time - in order to make sure we are initialized into a decent state. Otherwise we need to check the different components and safely figure out control mode, which can be inaccurate.
        self.switch_to_navigation_mode()

        self._last_node_id = None
        self._last_poses_id = None
        self._uploaded_ids: set[int] = set()

    @property
    def model(self):
        return self._robot_model

    @property
    def is_homed(self) -> bool:
        return self._ros_client.is_homed

    @property
    def is_runstopped(self) -> bool:
        return self._ros_client.is_runstopped

    def at_goal(self) -> bool:
        """Returns true if we have up to date head info and are at goal position"""
        return self.nav.at_goal()

    # Mode interfaces

    def switch_to_navigation_mode(self):
        """Switch dream to navigation control
        Robot base is now controlled via continuous velocity feedback.
        """
        result_pre = True
        if self.manip.is_enabled:
            result_pre = self.manip.disable()

        result_post = self.nav.enable()

        self._base_control_mode = ControlMode.NAVIGATION
        return result_pre and result_post

    @property
    def base_control_mode(self) -> ControlMode:
        return self._base_control_mode

    def switch_to_busy_mode(self) -> bool:
        """Switch to a mode that says we are occupied doing something blocking"""
        self._base_control_mode = ControlMode.BUSY
        return True

    def switch_to_manipulation_mode(self):
        """Switch dream to manipulation control
        Robot base is now controlled via position control.
        Base rotation is locked.
        """
        result_pre = True
        if self.nav.is_enabled:
            result_pre = self.nav.disable()

        result_post = self.manip.enable()

        self._base_control_mode = ControlMode.MANIPULATION

        return result_pre and result_post

    # General control methods

    def wait(self):
        self.nav.wait()
        self.manip.wait()
        self.cam.wait()

    def reset(self):
        self.stop()
        self.switch_to_manipulation_mode()
        self.manip.home()
        self.switch_to_navigation_mode()
        self.nav.home()
        self.stop()

    def stop(self):
        """Stop the robot"""
        self.nav.disable()
        self.manip.disable()
        self._base_control_mode = ControlMode.IDLE

    def pause_slam(self, timeout: float = 2.0) -> bool:
        """Pause SLAM updates via RTAB-Map."""
        return self._ros_client.pause_rtabmap(timeout=timeout)

    def resume_slam(self, timeout: float = 2.0) -> bool:
        """Resume SLAM updates via RTAB-Map."""
        return self._ros_client.resume_rtabmap(timeout=timeout)
    
    def shutdown(self):
        if hasattr(self, "_ros_client") and hasattr(self._ros_client, "shutdown"):
            self._ros_client.shutdown()

    # Other interfaces

    def get_robot_model(self) -> RobotModel:
        """return a model of the robot for planning. Overrides base class method"""
        return self._robot_model

    def get_ros_client(self) -> DreamRosInterface:
        """return the internal ROS client"""
        return self._ros_client

    @property
    def robot_joint_pos(self):
        return self._ros_client.pos

    @property
    def rgb_cam(self):
        return self._ros_client.rgb_cam

    @property
    def dpt_cam(self):
        return self._ros_client.dpt_cam

    @property
    def lidar(self):
        return self._ros_client._lidar


    def move_to_manip_posture(self):
        """Move the arm and head into manip mode posture: gripper down, head facing the gripper."""
        self.switch_to_manipulation_mode()
        self.manip.reset()
        print("- Robot switched to Manipulation mode.")

    def move_to_nav_posture(self):
        """Move the arm and head into nav mode. The head will be looking front."""
        self.switch_to_manipulation_mode()
        self.manip.reset()
        self.switch_to_navigation_mode()
        print("- Robot switched to navigation mode.")


    def get_arm_state(self):
        arm_state, arm_velocity, arm_force = self.manip._arm.get_joint_state()
        return arm_state, arm_velocity, arm_force
    
    def get_gripper_state(self):
        gripper_pos = self.manip._arm.get_gripper_state()
        return gripper_pos

    def get_arm_position(self):
        arm_position = self.manip._arm.get_current_pose()
        return arm_position

    def get_base_in_map_pose(self) -> np.ndarray:
        return self._ros_client.get_base_in_map_pose()
    
    def get_base_in_map_xyt(self) -> np.ndarray:
        """Get the robot's base pose as XYT (required by AbstractRobotClient)."""
        base_in_map_pose = self.get_base_in_map_pose()
        return sophus2xyt(pose2sophus(base_in_map_pose))

    def get_arm_base_in_map_pose(self):
        return self._ros_client.get_arm_base_in_map_pose()
    
    def get_camera_in_arm_base_pose(self):
        return self._ros_client.get_camera_in_arm_base_pose()
    
    def get_camera_in_base_pose(self):
        return self._ros_client.get_camera_in_base_pose()

    def get_camera_in_map_pose(self):
        return self._ros_client.get_camera_in_map_pose()

    def get_ee_in_arm_base_pose(self):
        return self._ros_client.get_ee_in_arm_base_pose()

    def get_ee_in_base_pose(self):
        return self._ros_client.get_ee_in_base_pose()
    
    def get_ee_in_map_pose(self):
        return self._ros_client.get_ee_in_map_pose()

    def get_join_information(self):
        joint_states = np.zeros(len(ROBOT_JOINTS))
        joint_velocities = np.zeros(len(ROBOT_JOINTS))
        joint_forces = np.zeros(len(ROBOT_JOINTS))
        joint_positions = np.zeros(len(ROBOT_JOINTS))

        base_pose = self.get_base_in_map_pose()

        arm_state, arm_velocity, arm_force = self.get_arm_state()
        vel_base = self._ros_client.get_vel_base()
        
        arm_position = self.get_arm_position()
        gripper_position = self.get_gripper_state()

        # if base_pose is None or arm_state is None or arm_position is None or gripper_position is None:
        #     return None
        if not all(v is not None for v in [base_pose, arm_state, arm_position, gripper_position]):
            return None

        joint_states[ARM_INDEX] = arm_state

        joint_velocities[BASE_INDEX] = vel_base
        joint_velocities[ARM_INDEX] = arm_velocity

        joint_forces[ARM_INDEX] = arm_force

        joint_positions[BASE_INDEX] = sophus2xyt(base_pose)
        joint_positions[ARM_INDEX] = arm_position
        joint_positions[GRIPPER_INDEX] = gripper_position
        
        # # If we are in manipulation mode...
        # if self._base_control_mode == ControlMode.MANIPULATION:
        #     # ...we need to get the joint positions from the manipulator
        #     joint_positions[DreamIdx.BASE_X] = self.manip.get_base_x()

        return joint_states, joint_velocities, joint_forces, joint_positions


    def get_pose_graph(self) -> np.ndarray:  # TODO (zhijie, may need edit)
        """Get SLAM pose graph as a numpy array"""
        graph = self._ros_client.get_pose_graph()
        for i in range(len(graph)):
            relative_pose = xyt2sophus(np.array(graph[i][1:4]))
            euler_angles = relative_pose.so3().log()
            theta = euler_angles[-1]

            # GPS in robot coordinates
            gps = relative_pose.translation()[:2]

            graph[i] = np.array([graph[i][0], gps[0], gps[1], theta])

        return graph

    def load_map(self, filename: str):
        self.mapping.load_map(filename)

    def save_map(self, filename: str):
        self.mapping.save_map(filename)

    def execute_trajectory(self, *args, **kwargs):
        """Open-loop trajectory execution wrapper. Executes a multi-step trajectory; this is always blocking since it waits to reach each one in turn."""
        return self.nav.execute_trajectory(*args, **kwargs)

    def base_to(
        self,
        xyt: Iterable[float],
        relative: bool = False,
        blocking: bool = True,
    ):
        """
        Move to xyt in global coordinates or relative coordinates. Cannot be used in manipulation mode.
        """
        return self.nav.base_to(xyt, relative=relative, blocking=blocking)

    def get_full_observation(self) -> RtabmapData:
        
        rtabmap_data = self._ros_client.get_rtabmapdata()
        base_in_map_pose = self.get_base_in_map_pose()
        if rtabmap_data is None or base_in_map_pose is None:
            return None
        
        timestamp = (
            rtabmap_data.header.stamp.sec
            + rtabmap_data.header.stamp.nanosec / 1e9
        )

        node = rtabmap_data.nodes[0]
        graph = rtabmap_data.graph
        node_id = node.id

        # Always use the latest pose graph
        # note that the latest node in the pose graph cannot be included in this judgment, 
        # as it may only be an intermediate node.
        assert max(graph.poses_id) == graph.poses_id[-1]
        pose_graph = {
            nid: pose_to_sophus(pose).matrix()
            for nid, pose in zip(
                graph.poses_id, graph.poses
            )
        }

        rgb = node.data.left_compressed
        depth = node.data.right_compressed

        # if history node, don't need to delete pose_graph[node_id], cause it pose is new
        if (self._last_node_id is not None and node_id <= self._last_node_id) or \
            (len(rgb) == 0) or (len(depth) == 0):
            print("[warning] 🛑 received history node or empty node")
            return RtabmapData(
                timestamp=timestamp,
                pose_graph=pose_graph,
                just_pose_graph=True,
            )
        else:
            self._last_node_id = node_id

        current_pose = pose2sophus(pose_graph[node_id])
        del pose_graph[node_id]
        
        # local_tf = transform_to_sophus(node.data.local_transform[0])
        left_ci = camera_info_to_dict(node.data.left_camera_info[0])
        camera_K = np.array(left_ci['K']).reshape(3, 3)

        euler_angles = current_pose.so3().log()
        compass = np.array([euler_angles[-1]])
        # GPS in robot coordinates
        gps = current_pose.translation()[:2]

        return RtabmapData(
            timestamp=timestamp,
            compass=compass,
            gps=gps,
            obs_id=node_id,
            just_pose_graph=False,
            rgb_compressed=rgb,
            depth_compressed=depth,
            camera_K=camera_K,
            pose_graph=pose_graph,
            camera_in_map_pose=current_pose.matrix(),
            base_in_map_pose=base_in_map_pose.matrix(),
        )


    # def get_full_observation(
    #     self,
    #     start_pose: Optional[np.ndarray] = None,
    # ) -> RtabmapData:
        
    #     rtabmap_data = self._ros_client.get_rtabmapdata()
    #     if rtabmap_data is None:
    #         return None
        
    #     timestamp = (
    #         rtabmap_data.header.stamp.sec
    #         + rtabmap_data.header.stamp.nanosec / 1e9
    #     )

    #     current_node = rtabmap_data.nodes[0]
    #     current_node_id = current_node.id

    #     # Always use the latest pose graph
    #     pose_graph = {
    #         nid: pose_to_sophus(pose)
    #         for nid, pose in zip(
    #             rtabmap_data.graph.poses_id, rtabmap_data.graph.poses
    #         )
    #     }

    #     if self._last_node_id is not None and current_node_id <= self._last_node_id:
    #         print("[warning] 🛑 received history node")
    #         return RtabmapData(
    #             timestamp=timestamp,
    #             pose_graph=pose_graph,
    #             just_pose_graph=True,
    #         )
    #     else:
    #         self._last_node_id = current_node_id

    #     assert len(rtabmap_data.nodes) == 1, "nodes has more than one node not happend!"

    #     # ================ send the latest node added to the pose graph. ================
    #     pose_ids = set(rtabmap_data.graph.poses_id)
    #     pose_ids.discard(current_node_id)
    #     new_pose_ids = set()

    #     if self._last_poses_id is not None:
    #         new_pose_ids = pose_ids - self._last_poses_id
    #     self._last_poses_id = pose_ids

    #     if new_pose_ids:
    #         # Only send the latest node added to the pose graph.
    #         newest_pose_id = max(new_pose_ids)  # 
    #         diff = current_node_id - newest_pose_id
    #         max_uploaded_id = max(self._uploaded_ids) if self._uploaded_ids else None
    #         is_newer_than_uploaded = (
    #             max_uploaded_id is None or newest_pose_id > max_uploaded_id
    #         )
    #         if newest_pose_id not in self._uploaded_ids and is_newer_than_uploaded:
    #             print(
    #                 f"now node id is {current_node_id}, added id: {new_pose_ids}, "
    #                 f"newest: {newest_pose_id}, diff: {diff}"
    #             )
    #             node_data_list = self._ros_client.get_node_data(
    #                 node_ids=[newest_pose_id],
    #             )
    #             self._uploaded_ids.add(newest_pose_id)

    #             if node_data_list:
    #                 latest_node = node_data_list[0]
    #                 latest_node_id = latest_node.id
    #                 rgb = latest_node.data.left_compressed or None
    #                 depth = latest_node.data.right_compressed or None
                    
    #                 if rgb is None or len(rgb) == 0:
    #                     print("get_full_observation: rgb is None or len(rgb) == 0")
    #                 if depth is None or len(depth) == 0:
    #                     print("get_full_observation: depth is None or len(depth) == 0")

    #                 local_tf = transform_to_sophus(
    #                     latest_node.data.local_transform[0]
    #                 )
    #                 assert len(latest_node.data.left_camera_info) > 0
    #                 left_ci = camera_info_to_dict(
    #                     latest_node.data.left_camera_info[0]
    #                 )
    #                 camera_K = np.array(left_ci["K"]).reshape(3, 3)


    #                 current_pose = pose_graph[latest_node_id]

    #                 if start_pose is not None:
    #                     # use sophus to get the relative translation
    #                     relative_pose = start_pose.inverse() * current_pose
    #                 else:
    #                     relative_pose = current_pose

    #                 euler_angles = relative_pose.so3().log()
    #                 compass = np.array([euler_angles[-1]])
    #                 gps = relative_pose.translation()[:2]  # GPS in robot coordinates

    #                 return RtabmapData(
    #                     timestamp=timestamp,
    #                     compass=compass,
    #                     gps=gps,
    #                     node_id=latest_node_id,
    #                     just_pose_graph=False,
    #                     rgb_compressed=rgb,
    #                     depth_compressed=depth,
    #                     camera_K=camera_K,
    #                     pose_graph=pose_graph,
    #                     base_in_map_pose=current_pose.matrix(),
    #                     camera_in_base_pose=local_tf.matrix(),
    #                 )

    #     return RtabmapData(
    #         timestamp=timestamp,
    #         pose_graph=pose_graph,
    #         just_pose_graph=True,
    #     )


    def get_state_observation(
        self,
        start_pose: Optional[np.ndarray] = None
    ) -> StateObservations:
        joint_information = self.get_join_information()
        base_in_map_pose = self.get_base_in_map_pose()
        arm_base_in_map_pose = self.get_arm_base_in_map_pose()
        camera_in_arm_base_pose = self.get_camera_in_arm_base_pose()
        camera_in_base_pose = self.get_camera_in_base_pose()
        camera_in_map_pose = self.get_camera_in_map_pose()
        ee_in_arm_base_pose = self.get_ee_in_arm_base_pose()
        ee_in_base_pose = self.get_ee_in_base_pose()
        ee_in_map_pose = self.get_ee_in_map_pose()
        
        if not all(v is not None for v in (
            joint_information,
            base_in_map_pose, arm_base_in_map_pose,
            camera_in_arm_base_pose, camera_in_base_pose, camera_in_map_pose,
            ee_in_arm_base_pose, ee_in_base_pose, ee_in_map_pose,
        )):
            print("get_state_observation: missing required state or tf pose")
            return None
        
        joint_states, joint_velocities, joint_forces, joint_positions = joint_information
        if start_pose is not None:
            relative_pose = start_pose.inverse() * base_in_map_pose
        else: 
            relative_pose = base_in_map_pose
        euler_angles = relative_pose.so3().log()
        theta = euler_angles[-1]
        gps = relative_pose.translation()[:2]
        return StateObservations(
            gps=gps,
            compass=np.array([theta]),
            base_in_map_pose=base_in_map_pose.matrix(),
            arm_base_in_map_pose=arm_base_in_map_pose.matrix(),
            camera_in_arm_base_pose=camera_in_arm_base_pose.matrix(),
            camera_in_base_pose=camera_in_base_pose.matrix(),
            camera_in_map_pose=camera_in_map_pose.matrix(),
            ee_in_arm_base_pose=ee_in_arm_base_pose.matrix(),
            ee_in_base_pose=ee_in_base_pose.matrix(),
            ee_in_map_pose=ee_in_map_pose.matrix(),
            joint_states=joint_states,
            joint_velocities=joint_velocities,
            joint_forces=joint_forces,
            joint_positions=joint_positions,
            at_goal=self.at_goal(),
            is_homed=self.is_homed,
            is_runstopped=self.is_runstopped,
        )

    def get_servo_observation(self) -> ServoObservations:
        images = self.cam.get_images(compute_xyz=False)
        camera_in_arm_base_pose = self.get_camera_in_arm_base_pose()

        if images is None or camera_in_arm_base_pose is None:
            return None

        rgb, depth = images[0], images[1]
        
        return ServoObservations(
            rgb=rgb,
            depth=depth,
            camera_in_arm_base_pose=camera_in_arm_base_pose.matrix(),
        )


    def get_camera_intrinsics(self) -> torch.Tensor:
        """Get 3x3 matrix of camera intrisics K"""
        return torch.from_numpy(self.cam._ros_client.rgb_cam.K).float()

    def get_servo_angle(self):
        return self.manip.get_servo_angle()

    # def move_to_positions(self, positions: List[np.ndarray], wait: bool = True):
    #     return self.manip.move_to_positions(positions, wait=wait)


    def get_has_wrist(self) -> bool:
        return self._ros_client.get_has_wrist()


if __name__ == "__main__":
    import rclpy

    rclpy.init()
    client = DreamClient()
    
    print("DreamClient initialized. Running without sleep for performance testing.")
    print("Press Ctrl+C to exit.")
    
    try:
        # Performance testing mode - no sleep for maximum performance
        while rclpy.ok():
            pass  # Busy waiting loop - no sleep for performance testing
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        rclpy.shutdown()
