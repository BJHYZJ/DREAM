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
    # ee_camera_frame = "gripper_camera_color_optical_frame"
    # ee_frame = "link_grasp_center"
    ee_frame = "link_eef"
    world_frame = "map"

    def __init__(
        self,
        init_node: bool = True,
        camera_overrides: Optional[Dict] = None,
        urdf_path: str = "",
        ik_type: str = "pinocchio",
        visualize_ik: bool = False,
        grasp_frame: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        manip_mode_controlled_joints: Optional[List[str]] = None,

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

        # Robot model
        # self._robot_model = HelloStretchKinematics(
        #     urdf_path=urdf_path,
        #     ik_type=ik_type,
        #     visualize=visualize_ik,
        #     grasp_frame=grasp_frame,
        #     ee_link_name=ee_link_name,
        #     manip_mode_controlled_joints=manip_mode_controlled_joints,
        # )

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
    
    def shutdown(self):
        """关闭ROS客户端"""
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

    # @property
    # def camera_pose(self):
    #     return self.head_camera_pose

    # @property
    # def head_camera_pose(self):
    #     p0 = self._ros_client.get_frame_pose(
    #         self.head_camera_frame, base_frame=self.world_frame, timeout_s=5.0
    #     )
    #     if p0 is not None:
    #         p0 = p0 @ tra.euler_matrix(0, 0, -np.pi / 2)
    #     return p0

    # @property
    # def ee_camera_pose(self):
    #     p0 = self._ros_client.get_frame_pose(
    #         self.ee_camera_frame, base_frame=self.world_frame, timeout_s=5.0
    #     )
    #     return p0

    # @property
    # def ee_pose(self):
    #     p0 = self._ros_client.get_frame_pose(self.ee_frame, base_frame=self.world_frame)
    #     if p0 is not None:
    #         p0 = p0 @ tra.euler_matrix(0, 0, 0)
    #     return p0

    @property
    def rgb_cam(self):
        return self._ros_client.rgb_cam

    @property
    def dpt_cam(self):
        return self._ros_client.dpt_cam

    # @property
    # def ee_dpt_cam(self):
    #     return self._ros_client.ee_dpt_cam

    # @property
    # def ee_rgb_cam(self):
    #     return self._ros_client.ee_rgb_cam

    @property
    def lidar(self):
        return self._ros_client._lidar


    def move_to_manip_posture(self):
        """Move the arm and head into manip mode posture: gripper down, head facing the gripper."""
        self.switch_to_manipulation_mode()
        # pos = self.manip._extract_joint_pos(STRETCH_PREGRASP_Q)
        # # pan, tilt = self._robot_model.look_at_ee
        # print("- go to configuration:", pos, "pan =", pan, "tilt =", tilt)
        # self.manip.goto_joint_positions(pos, head_pan=pan, head_tilt=tilt, blocking=True)
        # print("- Robot switched to manipulation mode.")
        assert 1 == 2

    def move_to_nav_posture(self):
        """Move the arm and head into nav mode. The head will be looking front."""

        # First retract the robot's joints
        self.switch_to_manipulation_mode()
        # pan, tilt = self._robot_model.look_close
        # pos = self.manip._extract_joint_pos(STRETCH_NAVIGATION_Q)
        # print("- go to configuration:", pos, "pan =", pan, "tilt =", tilt)
        # self.manip.goto_joint_positions(pos, head_pan=pan, head_tilt=tilt, blocking=True)
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
        
        # If we are in manipulation mode...
        if self._base_control_mode == ControlMode.MANIPULATION:
            # ...we need to get the joint positions from the manipulator
            joint_positions[DreamIdx.BASE_X] = self.manip.get_base_x()

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

    def get_full_observation(
        self,
        start_pose: Optional[np.ndarray] = None,
    ) -> RtabmapData:
        
        rtabmap_data = self._ros_client.get_rtabmapdata()
        if rtabmap_data is None:
            return None
        
        timestamp = rtabmap_data.header.stamp.sec + rtabmap_data.header.stamp.nanosec / 1e9
        last_timestamp = getattr(self, 'last_rtabmap_timestamp', None)
        if last_timestamp is not None and timestamp <= last_timestamp:
            # print("rtabmap data timestamp is not updated, Skipping...")
            return

        self.last_rtabmap_timestamp = timestamp
        
        nid = rtabmap_data.nodes[0].id
        is_history_node = False
        if getattr(self, '_last_node_id', None) is not None and nid <= self._last_node_id:
            print("[warning] 🛑 received history node")
            is_history_node = True

        self._last_node_id = nid


        node = rtabmap_data.nodes[0]
        node_id = node.id

        rgb = node.data.left_compressed
        depth = node.data.right_compressed
        laser = node.data.laser_scan_compressed
        if rgb is None or len(rgb) == 0:
            print("get_full_observation: rgb is None or len(rgb) == 0")
        if depth is None or len(depth) == 0:
            print("get_full_observation: depth is None or len(depth) == 0")
        if laser is None or len(laser) == 0:
            print("get_full_observation: laser is None or len(laser) == 0")
        
        if (rgb is None or len(rgb) == 0) or (depth is None or len(depth) == 0) or (laser is None or len(laser) == 0):
            print("=" * 32)

        pose_graph_now = {nid: pose_to_sophus(p) for nid, p in zip(rtabmap_data.graph.poses_id, rtabmap_data.graph.poses)}

        # Thread-safe update of pose graph
        self._ros_client.update_pose_graph(pose_graph_now)
        pose_graph = self._ros_client.get_pose_graph()
        
        local_tf = transform_to_sophus(node.data.local_transform[0])
        self._ros_client.update_local_tf_graph(node_id, local_tf)
        local_tf_graph = self._ros_client.get_local_tf_graph()

        current_pose = pose_graph[node_id]

        assert len(node.data.left_camera_info) > 0
        left_ci = camera_info_to_dict(node.data.left_camera_info[0])
        # right_ci = camera_info_to_dict(node.data.right_camera_info[0]) if len(node.data.right_camera_info) > 0 else None
        camera_K = np.array(left_ci['K']).reshape(3, 3)

        if start_pose is not None:
            # use sophus to get the relative translation
            relative_pose = start_pose.inverse() * current_pose
        else:
            relative_pose = current_pose
        euler_angles = relative_pose.so3().log()

        compass = np.array([euler_angles[-1]])
        # GPS in robot coordinates
        gps = relative_pose.translation()[:2]

        full_observation = RtabmapData(
            timestamp=timestamp,
            compass=compass,
            gps=gps,
            node_id=node_id,
            is_history_node=is_history_node,
            rgb_compressed=rgb,
            depth_compressed=depth,
            # laser_compressed=laser,
            camera_K=camera_K,
            pose_graph=pose_graph,
            local_tf_graph=local_tf_graph,
            base_in_map_pose=current_pose.matrix(),
            camera_in_map_pose=current_pose.matrix() @ local_tf.matrix(),
        )
        return full_observation

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
        # ee_in_map_pose = self.get_ee_in_map_pose()
        # camera_in_map_pose = self.get_camera_in_map_pose()

        # if images is None or ee_in_map_pose is None or camera_in_map_pose is None:
        #     print("get_servo_observation: images is None or ee_in_map_pose is None or camera_in_map_pose is None")
        #     return None

        rgb, depth = images[0], images[1]
        
        return ServoObservations(
            rgb=rgb,
            depth=depth,
            # ee_in_map_pose=ee_in_map_pose,
            # camera_in_map_pose=camera_in_map_pose,
        )


    def get_camera_intrinsics(self) -> torch.Tensor:
        """Get 3x3 matrix of camera intrisics K"""
        return torch.from_numpy(self.cam._ros_client.rgb_cam.K).float()

    def set_servo_angle(self, angle, is_radian=False, wait=True):
        self.manip.set_servo_angle(angle, is_radian=is_radian, wait=wait)

    def get_servo_angle(self):
        return self.manip.get_servo_angle()

    def move_to_positions(self, positions: List[np.ndarray], wait: bool = True):
        return self.manip.move_to_positions(positions, wait=wait)


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
