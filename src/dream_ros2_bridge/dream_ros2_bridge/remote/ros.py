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

from typing import Tuple
import os
import time
import atexit, signal

import sys
import threading
from typing import Dict, Optional

import numpy as np
import cv2
# from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import rclpy
import sophuspy as sp
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Twist
from hello_helpers.joint_qpos_conversion import get_Idx

# from nav2_msgs.srv import LoadMap, SaveMap
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32, Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

# from dream.motion.constants import (
#     ROS_ARM_JOINTS,
#     ROS_GRIPPER_FINGER,
#     ROS_HEAD_PAN,
#     ROS_HEAD_TILT,
#     ROS_LIFT_JOINT,
#     ROS_WRIST_PITCH,
#     ROS_WRIST_ROLL,
#     ROS_WRIST_YAW,
#     STRETCH_HEAD_CAMERA_ROTATIONS,
# )
from dream.motion.kinematics import DreamIdx
from dream.utils.pose import to_matrix, transform_to_list
from dream.utils.geometry import angle_difference, sophus2xyt, xyt2sophus, xyt_base_to_global
from dream.motion.constants import BASE_JOINTS, ARM_JOINTS, GRIPPER_JOINTS, ROBOT_JOINTS
from dream_ros2_bridge.ros.camera import RosCamera
# from dream_ros2_bridge.ros.lidar import RosLidar
from dream_ros2_bridge.ros.lidar3d import Ros3DLidar
from dream_ros2_bridge.ros.streaming_activator import StreamingActivator
from dream_ros2_bridge.ros.utils import matrix_from_pose_msg
from dream_ros2_bridge.ros.visualizer import Visualizer
from rtabmap_msgs.msg import MapData, Info
from dream_ros2_bridge.xarm6_controller import XARM6
from geometry_msgs.msg import PoseStamped

DEFAULT_COLOR_TOPIC = "/camera/color"
DEFAULT_DEPTH_TOPIC = "/camera/aligned_depth_to_color"
DEFAULT_LIDAR_TOPIC = '/livox/lidar'
# DEFAULT_LIDAR_TOPIC = "/scan"
# DEFAULT_EE_COLOR_TOPIC = "/gripper_camera/color"
# DEFAULT_EE_DEPTH_TOPIC = "/gripper_camera/aligned_depth_to_color"

def pose_to_dict(p):
    return {
        "tx": float(p.position.x), "ty": float(p.position.y), "tz": float(p.position.z),
        "qx": float(p.orientation.x), "qy": float(p.orientation.y),
        "qz": float(p.orientation.z), "qw": float(p.orientation.w)
    }

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


class DreamRosInterface(Node):
    """Interface object with ROS topics and services"""

    # Base of the robot
    base_link = "base_link"

    goal_time_tolerance = 1.0
    msg_delay_t = 0.1

    # 3 for base position + rotation, 2 for lift + extension, 3 for rpy, 1 for gripper, 2 for head
    # dof = 3 + 2 + 3 + 1 + 2
    dof = 3 + 6 + 1 # 6 for arm joints, 1 for gripper, no base joints, just x, y, theta
    # Joint names in the ROS joint trajectory server
    # BASE_TRANSLATION_JOINT = "translate_mobile_base"
    # ARM_JOINT = "joint_arm"
    # LIFT_JOINT = ROS_LIFT_JOINT
    # WRIST_YAW = ROS_WRIST_YAW
    # WRIST_PITCH = ROS_WRIST_PITCH
    # WRIST_ROLL = ROS_WRIST_ROLL
    # GRIPPER_FINGER = ROS_GRIPPER_FINGER
    # HEAD_PAN = ROS_HEAD_PAN
    # HEAD_TILT = ROS_HEAD_TILT
    # ARM_JOINTS_ACTUAL = ROS_ARM_JOINTS

    def __init__(
        self,
        init_cameras: bool = True,
        color_topic: Optional[str] = None,
        depth_topic: Optional[str] = None,
        depth_buffer_size: Optional[int] = None,
        init_lidar: bool = True,
        lidar_topic: Optional[str] = None,
        verbose: bool = False,

        ee_color_topic: Optional[str] = None,
        ee_depth_topic: Optional[str] = None,
    ):
        super().__init__("robot_user_client")
        # Verbosity for the ROS client
        self.verbose = verbose

        self._arm_client = XARM6()

        # QoS设置
        self.reliable_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.best_effort_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._executor = MultiThreadedExecutor(num_threads=8)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._spin_forever, daemon=False)
        self._spin_thread.start()
        # Start the thread  # publish pose visualization
        # self._thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        # self._thread.start()

        self.cb_rtabmapdata_group = MutuallyExclusiveCallbackGroup()
        # self.cb_rtabmapinfo_group = MutuallyExclusiveCallbackGroup()
        self.cb_grasp_group = MutuallyExclusiveCallbackGroup()
        self.cb_place_group = MutuallyExclusiveCallbackGroup()
        self.cb_tf_group = ReentrantCallbackGroup()
        self.cb_state_group = ReentrantCallbackGroup()
        self.cb_odom_group = ReentrantCallbackGroup()
        self.cb_srv_group = ReentrantCallbackGroup()

        # Initialize caches
        self.current_mode: Optional[str] = None

        # self.arm_pos = np.zeros(self.dof)
        # self.arm_vel = np.zeros(self.dof)
        # self.arm_frc = np.zeros(self.dof)
        # self.arm_joint_status: Dict[str, float] = {}

        self.arm_pos = np.zeros(len(ARM_JOINTS))
        self.arm_vel = np.zeros(len(ARM_JOINTS))
        self.arm_frc = np.zeros(len(ARM_JOINTS))
        self.gripper_pos = 830  # 830 is the open position

        self.pos = np.zeros(self.dof)
        self.vel = np.zeros(self.dof)
        self.frc = np.zeros(self.dof)
        self.joint_status: Dict[str, float] = {}

        self.vel_base = np.zeros(len(BASE_JOINTS))

        self.rtabmapdata = None

        # self.se3_base_filtered: Optional[sp.SE3] = None
        # self.se3_base_odom: Optional[sp.SE3] = None
        # self.se3_camera_pose: Optional[sp.SE3] = None

        self.se3_base: Optional[sp.SE3] = None
        self.se3_camera_in_map_pose: Optional[sp.SE3] = None
        self.se3_camera_in_base_pose: Optional[sp.SE3] = None
        self.se3_camera_in_arm_pose: Optional[sp.SE3] = None
        self.se3_ee_in_map_pose: Optional[sp.SE3] = None

        self.at_goal: bool = False

        # self.last_odom_update_timestamp = Time(clock_type=ClockType.ROS_TIME)
        # self.last_base_update_timestamp = Time(clock_type=ClockType.ROS_TIME)
        self._goal_reset_t = Time(clock_type=ClockType.ROS_TIME)

        # Create visualizers for pose information
        self.goal_visualizer = Visualizer("command_pose", rgba=[1.0, 0.0, 0.0, 0.5])
        self.curr_visualizer = Visualizer("current_pose", rgba=[0.0, 0.0, 1.0, 0.5])

        # self._is_homed = None
        # self._is_runstopped = None
        self._is_homed = True
        self._is_runstopped = False

        self._pose_graph = None
        self._local_tf_graph = None


        self._lock_js = threading.Lock()
        self._lock_tf = threading.Lock()
        self._lock_odom = threading.Lock()
        self._lock_rtab = threading.Lock() 
        self._lock_goal = threading.Lock()

        # Initialize ros communication
        self._create_pubs_subs()
        self._create_services()

        # self._ros_joint_names = []
        # for i in range(3, self.dof):
        #     self._ros_joint_names += CONFIG_TO_ROS[i]

        # Get fleet ID from environment
        # fleet_id = os.environ["HELLO_FLEET_ID"]
        # Parse from format dream-reX-XX to just reX
        # robot_model = fleet_id.split("-")[1]
        # print(f"Using fleet ID: {fleet_id}")
        # print(f"Using robot model: {robot_model}")
        # self._fleet_id = fleet_id
        # self._robot_model = robot_model

        # Initialize cameras
        self._color_topic = DEFAULT_COLOR_TOPIC if color_topic is None else color_topic
        self._depth_topic = DEFAULT_DEPTH_TOPIC if depth_topic is None else depth_topic
        # self._ee_color_topic = DEFAULT_EE_COLOR_TOPIC if ee_color_topic is None else ee_color_topic
        # self._ee_depth_topic = DEFAULT_EE_DEPTH_TOPIC if ee_depth_topic is None else ee_depth_topic
        self._lidar_topic = '/livox/lidar'
        self._depth_buffer_size = depth_buffer_size

        # self._streaming_activator = StreamingActivator(self)
        # streaming_ok = self._streaming_activator.activate_streaming()
        # if not streaming_ok:
        #     self.get_logger().error("Failed to activate streaming")
        #     raise RuntimeError(
        #         "Could not start joint state streaming service; make sure you have the correct and up-to-date version of stretch_ros2."
        #     )

        self.rgb_cam: RosCamera = None
        self.dpt_cam: RosCamera = None
        self.lidar: Ros3DLidar = None
        if init_cameras:
            # self._create_cameras(use_d405=d405)
            self._create_cameras()
            self._wait_for_cameras()
        if init_lidar:
            self._create_lidar()
            self._wait_for_lidar()

        print("..done.")
        # if self.get_has_wrist():
        #     # Get indexer
        #     self._has_wrist = True
        #     self.Idx = get_Idx("eoa_wrist_dw3_tool_sg3")
        # else:
        #     self._has_wrist = False
        #     self.Idx = get_Idx("tool_stretch_gripper")

    def _spin_forever(self):
        try:
            self._executor.spin()
        except RuntimeError as e:
            self.get_logger().warn(f"spin stopped: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in spin: {e}")

    # def shutdown(self):
    #     """关闭ROS接口"""
    #     if hasattr(self, "_spin_thread") and self._spin_thread.is_alive():
    #         self._spin_thread.join(timeout=3.0)
    #     if hasattr(self, "_executor"):
    #         self._executor.shutdown(timeout_sec=2.0)

    def shutdown(self):
        if hasattr(self, "_executor"):
            try:
                self._executor.shutdown(timeout_sec=2.0)
            except Exception:
                pass
        if hasattr(self, "_spin_thread") and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=3.0)


    def __del__(self):
        self.shutdown()


    def get_joint_state(self):
        with self._lock_js:
            self.arm_pos, self.arm_vel, self.arm_frc = self._arm_client.get_joint_state()
            self.gripper_pos = self._arm_client.get_gripper_state()

        base_pose = self.get_base_in_map_pose()

        if base_pose is None:
            return None
            
        self.pos[:3] = sophus2xyt(base_pose)
        self.pos[3:9] = self.arm_pos[:6]
        self.pos[9] = self.gripper_pos
        self.vel[:3] = self.vel_base
        self.vel[3:9] = self.arm_vel[:6]
        self.frc[3:9] = self.arm_frc[:6]
        return [self.pos, self.vel, self.frc]
    

    # def _process_joint_status(self, j_status) -> np.ndarray:
    #     """Get joint status from ROS joint state message and convert it into the form we use for streaming position commands."""
    #     pose = np.zeros(self.Idx.num_joints)
    #     pose[self.Idx.LIFT] = j_status[ROS_LIFT_JOINT]
    #     pose[self.Idx.ARM] = (
    #         j_status[ROS_ARM_JOINTS[0]]
    #         + j_status[ROS_ARM_JOINTS[1]]
    #         + j_status[ROS_ARM_JOINTS[2]]
    #         + j_status[ROS_ARM_JOINTS[3]]
    #     )
    #     # Set the gripper
    #     pose[self.Idx.GRIPPER] = j_status[ROS_GRIPPER_FINGER]

    #     # Check if we have wrist joints
    #     if ROS_WRIST_ROLL in j_status:
    #         pose[self.Idx.WRIST_ROLL] = j_status[ROS_WRIST_ROLL]
    #         pose[self.Idx.WRIST_PITCH] = j_status[ROS_WRIST_PITCH]
    #     # else:
    #     #    pose[self.Idx.WRIST_ROLL] = 0
    #     #    pose[self.Idx.WRIST_PITCH] = -np.pi / 4
    #     # We always have yaw
    #     pose[self.Idx.WRIST_YAW] = j_status[ROS_WRIST_YAW]

    #     # Head joints
    #     pose[self.Idx.HEAD_PAN] = j_status[ROS_HEAD_PAN]
    #     pose[self.Idx.HEAD_TILT] = j_status[ROS_HEAD_TILT]
    #     return pose

    # def get_has_wrist(self) -> bool:
    #     """Check if the robot has a wrist joint."""
    #     # Wait until self.joint_status is populated
    #     rate = self.create_rate(10)
    #     while rclpy.ok():
    #         with self._lock_js:
    #             print("Waiting for joint status...", self.joint_status.keys())
    #             if ROS_LIFT_JOINT in self.joint_status:
    #                 break
    #         rate.sleep()
    #     print("Done waiting for joint status...", self.joint_status.keys())
    #     return ROS_WRIST_ROLL in self.joint_status

    # def send_joint_goals(
    #     self, joint_goals: Dict[str, float], velocities: Optional[Dict[str, float]] = None
    # ):
    #     """Send joint goals to the robot. Goals are a dictionary of joint names and strings. Can optionally provide velicities as well."""

    #     # with self._lock_js:
    #         # joint_pose = self._process_joint_status(self.joint_status)
    #     joint_pose = self._process_joint_status(self.joint_status)

    #     # Use Idx to convert
    #     if self.LIFT_JOINT in joint_goals:
    #         joint_pose[self.Idx.LIFT] = joint_goals[self.LIFT_JOINT]
    #     if self.ARM_JOINT in joint_goals:
    #         joint_pose[self.Idx.ARM] = joint_goals[self.ARM_JOINT]
    #     if self.WRIST_ROLL in joint_goals and ROS_WRIST_ROLL in self.joint_status:
    #         # Only set wrist roll if we have it
    #         joint_pose[self.Idx.WRIST_ROLL] = joint_goals[self.WRIST_ROLL]
    #     if self.WRIST_PITCH in joint_goals and ROS_WRIST_PITCH in self.joint_status:
    #         # Only set wrist pitch if we have it
    #         joint_pose[self.Idx.WRIST_PITCH] = joint_goals[self.WRIST_PITCH]
    #     # We always have yaw
    #     if self.WRIST_YAW in joint_goals:
    #         joint_pose[self.Idx.WRIST_YAW] = joint_goals[self.WRIST_YAW]
    #     if self.GRIPPER_FINGER in joint_goals:
    #         joint_pose[self.Idx.GRIPPER] = joint_goals[self.GRIPPER_FINGER]
    #     if self.HEAD_PAN in joint_goals:
    #         joint_pose[self.Idx.HEAD_PAN] = joint_goals[self.HEAD_PAN]
    #     if self.HEAD_TILT in joint_goals:
    #         joint_pose[self.Idx.HEAD_TILT] = joint_goals[self.HEAD_TILT]
    #     if self.BASE_TRANSLATION_JOINT in joint_goals:
    #         joint_pose[self.Idx.BASE_TRANSLATE] = joint_goals[self.BASE_TRANSLATION_JOINT]

    #     # Create the message now that it's been computed
    #     msg = Float64MultiArray()
    #     msg.data = list(joint_pose)
    #     self._joint_goal_publisher.publish(msg)
    #     self.get_logger().info('Publishing: "%s"' % msg.data)

    def send_trajectory_goals(
        self, joint_goals: Dict[str, float], velocities: Optional[Dict[str, float]] = None
    ):
        """Send trajectory goals to the robot. Goals are a dictionary of joint names and strings. Can optionally provide velicities as well."""

        # Preprocess arm joints (arm joints are actually 4 joints in one)
        if self.ARM_JOINT in joint_goals:
            arm_joint_goal = joint_goals.pop(self.ARM_JOINT)

            for arm_joint_name in self.ARM_JOINTS_ACTUAL:
                joint_goals[arm_joint_name] = arm_joint_goal / len(self.ARM_JOINTS_ACTUAL)

        # Preprocess base translation joint (stretch_driver errors out if translation value is 0)
        if self.BASE_TRANSLATION_JOINT in joint_goals:
            if joint_goals[self.BASE_TRANSLATION_JOINT] == 0:
                joint_goals.pop(self.BASE_TRANSLATION_JOINT)

        # Parse input
        joint_names = []
        joint_values = []
        for name, val in joint_goals.items():
            joint_names.append(name)
            joint_values.append(val)

        # Construct goal positions
        point_msg = JointTrajectoryPoint()
        point_msg.positions = joint_values
        if velocities is not None:
            point_msg.velocities = velocities

        # Construct goal msg
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.goal_time_tolerance = Duration(seconds=self.goal_time_tolerance).to_msg()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points = [point_msg]
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        # self.action_done_event = Event()
        # Send goal
        self.goal_handle = None
        self.goal_handle_future = self.trajectory_client.send_goal_async(goal_msg)
        self.goal_handle_future.add_done_callback(self.trajectory_done_callback)
        # self.get_logger().info(f"")

    def trajectory_done_callback(self, future):
        self.goal_handle = future.result()

    def wait_for_trajectory_action(self):
        rate = self.create_rate(100)
        while self.goal_handle is None:
            rate.sleep()
        self.goal_handle.get_result()

    def recent_depth_image(self, seconds, print_delay_timers: bool = False):
        """Return true if we have up to date depth."""
        # Make sure we have a goal and our poses and depths are synced up - we need to have
        # received depth after we stopped moving
        if print_delay_timers:
            print(
                " - 1",
                (self.get_clock().now() - self._goal_reset_t) * 1e-9,
                self.msg_delay_t,
            )
            print(" - 2", (self.dpt_cam.get_time() - self._goal_reset_t) * 1e-9, seconds)
        if (
            self._goal_reset_t is not None
            and (self.get_clock().now() - self._goal_reset_t).nanoseconds * 1e-9 > self.msg_delay_t
        ):
            # self.get_logger().info(
            #     f"Two Times {self.dpt_cam.get_time()}, {self._goal_reset_t}, {seconds}"
            # )
            return (self.dpt_cam.get_time().sec - self._goal_reset_t.nanoseconds * 1e-9) > seconds
        else:
            return False

    def config_to_ros_trajectory_goal(
        self, q: np.ndarray, dq: np.ndarray = None, ddq: np.ndarray = None
    ) -> FollowJointTrajectory.Goal:
        """Create a joint trajectory goal to move the arm."""
        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.goal_time_tolerance = Duration(seconds=self.goal_time_tolerance).to_msg()
        trajectory_goal.trajectory.joint_names = self.ros_joint_names
        trajectory_goal.trajectory.points = [self._config_to_ros_msg(q, dq, ddq)]
        trajectory_goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        return trajectory_goal

    # Helper functions

    def _create_services(self):
        """Create services to activate/deactivate robot modes"""
        # self.nav_mode_service = self.create_client(Trigger, "switch_to_navigation_mode")
        # self.pos_mode_service = self.create_client(Trigger, "switch_to_position_mode")

        # self.save_map_service = self.create_client(SaveMap, "save_map")
        # self.load_map_service = self.create_client(LoadMap, "load_map")

        self.goto_on_service = self.create_client(Trigger, "goto_controller/enable", callback_group=self.cb_srv_group)
        self.goto_off_service = self.create_client(Trigger, "goto_controller/disable", callback_group=self.cb_srv_group)
        self.set_yaw_service = self.create_client(SetBool, "goto_controller/set_yaw_tracking", callback_group=self.cb_srv_group)
        # print("Wait for mode service...")
        # self.pos_mode_service.wait_for_service()

        # print("Wait for map services...")
        # self.save_map_service.wait_for_service()
        # self.load_map_service.wait_for_service()

    def _is_homed_cb(self, msg) -> None:
        """Update this variable"""
        self._is_homed = bool(msg.data)

    @property
    def is_homed(self) -> bool:
        return self._is_homed

    def _is_runstopped_cb(self, msg) -> None:
        """Update this variable"""
        self._is_runstopped = bool(msg.data)

    @property
    def is_runstopped(self) -> bool:
        return self._is_runstopped

    def _create_pubs_subs(self):
        """create ROS publishers and subscribers - only call once"""

        # Create command publishers
        self.goal_pub = self.create_publisher(Pose, "goto_controller/goal", 1)
        self.velocity_pub = self.create_publisher(Twist, "/ranger/cmd_vel", 1)
        self._at_goal_sub = self.create_subscription(Bool, "goto_controller/at_goal", self._at_goal_callback, 1, callback_group=self.cb_state_group)

        # Check if robot is homed and runstopped
        self._is_homed_sub = self.create_subscription(Bool, "/is_homed", self._is_homed_cb, 1, callback_group=self.cb_state_group)
        self._is_runstopped_sub = self.create_subscription(Bool, "/is_runstopped", self._is_runstopped_cb, 1, callback_group=self.cb_state_group)
        # self._mode_sub = self.create_subscription(String, "mode", self._mode_callback, 1, callback_group=self.cb_state_group)

        self.grasp_ready = None
        self.grasp_complete = None
        self.grasp_enable_pub = self.create_publisher(Empty, "grasp_point/enable", 1)
        self.grasp_disable_pub = self.create_publisher(Empty, "grasp_point/disable", 1)
        self.grasp_trigger_pub = self.create_publisher(PointStamped, "grasp_point/trigger_grasp_point", 1)
        self.grasp_ready_sub = self.create_subscription(Empty, "grasp_point/ready", self._grasp_ready_callback, 10, callback_group=self.cb_grasp_group)  # Had to check qos_profile
        self.grasp_result_sub = self.create_subscription(Float32, "grasp_point/result", self._grasp_result_callback, 10, callback_group=self.cb_grasp_group)  # Had to check qos_profile

        self.place_ready = None
        self.place_complete = None
        self.location_above_surface_m = None
        self.place_enable_pub = self.create_publisher(Float32, "place_point/enable", 1)
        self.place_disable_pub = self.create_publisher(Empty, "place_point/disable", 1)
        self.place_trigger_pub = self.create_publisher(PointStamped, "place_point/trigger_place_point", 1)
        self.place_ready_sub = self.create_subscription(Empty, "place_point/ready", self._place_ready_callback, 10, callback_group=self.cb_place_group)  # Had to check qos_profile
        self.place_result_sub = self.create_subscription(Empty, "place_point/result", self._place_result_callback, 10, callback_group=self.cb_place_group)  # Had to check qos_profile

        # Create subscribers
        self._odom_sub = self.create_subscription(Odometry, "/ranger/odom", self._odom_callback, self.best_effort_qos, callback_group=self.cb_odom_group)

        # self._base_state_sub = self.create_subscription(
        #     PoseStamped, "state_estimator/pose_filtered", self._base_state_callback, 1
        # )

        # self._camera_pose_sub = self.create_subscription(
        #     PoseStamped, "camera_pose", self._camera_pose_callback, 1
        # )


        # self._pose_graph_sub = self.create_subscription(
        #     Path, "slam_toolbox/pose_graph", self._pose_graph_callback, 1
        # )
        # zhijie added
        self._rtabmapdata_sub = self.create_subscription(MapData, "/rtabmap/mapData", self._rtabmapdata_callback, self.reliable_qos, callback_group=self.cb_rtabmapdata_group)

        # self._rtabmapinfo_sub = self.create_subscription(Info, "/rtabmap/info", self._rtabmapinfo_callback, self.reliable_qos, callback_group=self.cb_rtabmapinfo_group)

        # tf pose publisher
        self._tf_base_pose_sub = self.create_subscription(PoseStamped, "tf_pose/base_pose", self._tf_base_pose_callback, self.best_effort_qos, callback_group=self.cb_tf_group)
        self._tf_camera_pose_in_map_sub = self.create_subscription(PoseStamped, "tf_pose/camera_pose_in_map", self._tf_camera_pose_in_map_callback, self.best_effort_qos, callback_group=self.cb_tf_group)
        self._tf_camera_pose_in_base_sub = self.create_subscription(PoseStamped, "tf_pose/camera_pose_in_base", self._tf_camera_pose_in_base_callback, self.best_effort_qos, callback_group=self.cb_tf_group)
        self._tf_camera_pose_in_arm_sub = self.create_subscription(PoseStamped, "tf_pose/camera_pose_in_arm", self._tf_camera_pose_in_arm_callback, self.best_effort_qos, callback_group=self.cb_tf_group)
        self._tf_ee_pose_in_map_sub = self.create_subscription(PoseStamped, "tf_pose/ee_pose_in_map", self._tf_ee_pose_in_map_callback, self.best_effort_qos, callback_group=self.cb_tf_group)

        # Create trajectory client with which we can control the robot
        # self.trajectory_client = ActionClient(
        #     self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory"
        # )  # Doubt action type


        # This callback group is used to ensure that the joint goal publisher is reentrant
        # TODO: notes on what this is for?
        # self._reentrant_cb = ReentrantCallbackGroup()

        # self._arm_joint_state_subscriber = self.create_subscription(
        #     JointState,
        #     "/xarm/joint_states",
        #     self._js_callback,
        #     100,
        #     callback_group=self._reentrant_cb,
        # )

        # Create joint goal publisher for streaming joint goals
        # self._joint_goal_publisher = self.create_publisher(
        #     Float64MultiArray, "joint_pose_cmd", 10, callback_group=self._reentrant_cb
        # )

        # print("Waiting for trajectory server...")
        # server_reached = self.trajectory_client.wait_for_server(timeout_sec=30.0)
        # if not server_reached:
        #     print("ERROR: Failed to connect to arm action server.")
        #     # rclpy.shutdown("Unable to connect to arm action server. Timeout exceeded.")
        #     rclpy.shutdown()
        #     sys.exit()
        # print("... connected to arm action server.")

        # self.ros_joint_names = []
        # for i in range(3, self.dof):
        #     self.ros_joint_names += CONFIG_TO_ROS[i]

        # self.ros_joint_names = ROS_ARM_JOINTS
        self.joint_names = ROBOT_JOINTS

    # def _create_cameras(self, use_d405: bool = True):
    def _create_cameras(self):
        if self.rgb_cam is not None or self.dpt_cam is not None:
            raise RuntimeError("Already created cameras")
        print("Creating cameras...")
        self.rgb_cam = RosCamera(self, self._color_topic)
        self.dpt_cam = RosCamera(self, self._depth_topic)
        # if use_d405:
        #     self.ee_rgb_cam = RosCamera(
        #         self,
        #         self._ee_color_topic,
        #         rotations=0,
        #         image_ext="/image_rect_raw",
        #     )
        #     self.ee_dpt_cam = RosCamera(
        #         self,
        #         self._ee_depth_topic,
        #         rotations=0,
        #         image_ext="/image_raw",
        #     )
        # else:
        #     self.ee_rgb_cam = None
        #     self.ee_dpt_cam = None
        self.filter_depth = self._depth_buffer_size is not None
    
    def _create_lidar(self):
        if self.lidar is not None:
            raise RuntimeError("Already created lidar")
        print("Creating lidar...")
        self.lidar = Ros3DLidar(self, self._lidar_topic)

    def _wait_for_lidar(self):
        """wait until lidar has a message"""
        print("Waiting for lidar scan...")
        self.lidar.wait_for_scan()

    def _wait_for_cameras(self):
        if self.rgb_cam is None or self.dpt_cam is None:
            raise RuntimeError("cameras not initialized")
        print("Waiting for rgb camera images...")
        self.rgb_cam.wait_for_image()
        print("Waiting for depth camera images...")
        self.dpt_cam.wait_for_image()
        # if self.ee_rgb_cam is not None:
        #     print("Waiting for end effector rgb camera images...")
        #     self.ee_rgb_cam.wait_for_image()
        # if self.ee_dpt_cam is not None:
        #     print("Waiting for end effector depth camera images...")
        #     self.ee_dpt_cam.wait_for_image()
        
        if self.verbose:
            print("rgb frame =", self.rgb_cam.get_frame())
            print("dpt frame =", self.dpt_cam.get_frame())
        # if self.rgb_cam.get_frame() != self.dpt_cam.get_frame():
        #     raise RuntimeError("issue with camera setup; depth and rgb not aligned")

    def _config_to_ros_msg(self, q, dq=None, ddq=None):
        """convert into a joint state message"""
        msg = JointTrajectoryPoint()
        msg.positions = [0.0] * len(self._ros_joint_names)
        if dq is not None:
            msg.velocities = [0.0] * len(self._ros_joint_names)
        if ddq is not None:
            msg.accelerations = [0.0] * len(self._ros_joint_names)
        idx = 0
        for i in range(3, self.dof):
            names = CONFIG_TO_ROS[i]
            for _ in names:
                # Only for arm - but this is a dumb way to check
                if "arm" in names[0]:
                    msg.positions[idx] = q[i] / len(names)
                else:
                    msg.positions[idx] = q[i]
                if dq is not None:
                    msg.velocities[idx] = dq[i]
                if ddq is not None:
                    msg.accelerations[idx] = ddq[i]
                idx += 1
        return msg

    # Rostopic callbacks

    def _at_goal_callback(self, msg):
        """Is the velocity controller done moving; is it at its goal?"""
        # self.get_logger().info(f"at goal listennign {self.at_goal}")
        with self._lock_goal:
            self.at_goal = msg.data
        if not self.at_goal:
            self._goal_reset_t = None
        elif self._goal_reset_t is None:
            self._goal_reset_t = self.get_clock().now()

    def _rtabmapdata_callback(self, msg):

        with self._lock_rtab:
            self.rtabmapdata = msg
        
        # timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        # now_timestamp = self.get_clock().now().to_msg()
        # now_timestamp = now_timestamp.sec + now_timestamp.nanosec / 1e9
        # # self.get_logger().info(f"rtabmapdata timestamp: {timestamp}, now_timestamp: {now_timestamp}, diff: {now_timestamp - timestamp}")
        # print(f"rtabmapdata timestamp: {timestamp}, now_timestamp: {now_timestamp}, diff: {now_timestamp - timestamp}")

        # nid = msg.nodes[0].id

        # timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        # last_timestamp = getattr(self, 'last_rtabmap_timestamp', None)
        # if last_timestamp is not None and timestamp <= last_timestamp:
        #     print("rtabmap data timestamp is not updated, Skipping...")
        #     return
        # # else:
        # #     print(f"{rtabmap_data.nodes[0].id} {timestamp} rtabmap data timestamp is updated, Updating...")
        # self.last_rtabmap_timestamp = timestamp
        
        # if getattr(self, '_last_node_id', None) is not None and nid <= self._last_node_id:
        #     self.get_logger().warn(f"RTABMap Node ID out-of-order: {nid} <= {self._last_node_id}")
        #     return
        #     # raise RuntimeError(f"RTABMap Node ID sequence error: received {nid} but expected > {self._last_node_id}")
        # # timestamp_now = self.get_clock().now().to_msg()
        # # timestamp_tf = msg.header.stamp
        # # delay = timestamp_now.sec + timestamp_now.nanosec / 1e9 - timestamp_tf.sec - timestamp_tf.nanosec / 1e9
        # # self.get_logger().info(f"RTABMap data received: Node ID {nid}, Timestamp {msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9}, Delay {delay}")
        # # self.get_logger().info(f"RTABMap data received: Node ID {nid}, Timestamp {msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9}")
        # self._last_node_id = nid
        # with self._lock_rtab:
        #     self.rtabmapdata = msg

    def _rtabmapinfo_callback(self, msg):
        """get position or navigation mode from dream ros"""
        # self._rtabmapinfo = msg
        assert 1 == 1

    # def _mode_callback(self, msg):
    #     """get position or navigation mode from dream ros"""
    #     self._current_mode = msg.data

    # def _pose_graph_callback(self, msg):
    #     self._pose_graph = []

    #     for pose in msg.poses:
    #         p = [
    #             pose.header.stamp.sec + pose.header.stamp.nanosec / 1e9,
    #             pose.pose.position.x,
    #             pose.pose.position.y,
    #             pose.pose.position.z,
    #         ]
    #         self._pose_graph.append(p)

    def _odom_callback(self, msg: Odometry):
        """odometry callback"""
        # self.vel_base = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
        with self._lock_odom:
            self.vel_base = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z])
        # self._last_odom_update_timestamp = msg.header.stamp
        # self.se3_base_odom = sp.SE3(matrix_from_pose_msg(msg.pose.pose))

    def _tf_base_pose_callback(self, msg: PoseStamped):
        """base pose callback"""
        # timestamp_now = self.get_clock().now().to_msg()
        # timestamp_tf = msg.header.stamp
        # delay = timestamp_now.sec + timestamp_now.nanosec / 1e9 - timestamp_tf.sec - timestamp_tf.nanosec / 1e9
        # self.get_logger().info(f"TF delay: {delay}")
        se3 = sp.SE3(matrix_from_pose_msg(msg.pose))
        with self._lock_tf:
            self.se3_base = se3

    def _tf_camera_pose_in_map_callback(self, msg: PoseStamped):
        """camera pose in map callback"""
        se3 = sp.SE3(matrix_from_pose_msg(msg.pose))
        # timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        # now_timestamp = time.time()
        # self.get_logger().info(f"tf_camera_pose_in_map timestamp: {timestamp}, now_timestamp: {now_timestamp}, diff: {now_timestamp - timestamp}")
        with self._lock_tf:
            self.se3_camera_in_map_pose = se3

    def _tf_camera_pose_in_base_callback(self, msg: PoseStamped):
        """camera pose in base callback"""
        se3 = sp.SE3(matrix_from_pose_msg(msg.pose))
        with self._lock_tf:
            self.se3_camera_in_base_pose = se3
    
    def _tf_camera_pose_in_arm_callback(self, msg: PoseStamped):
        """camera pose in arm callback"""
        se3 = sp.SE3(matrix_from_pose_msg(msg.pose))
        with self._lock_tf:
            self.se3_camera_in_arm_pose = se3
    
    def _tf_ee_pose_in_map_callback(self, msg: PoseStamped):
        """ee pose in map callback"""
        se3 = sp.SE3(matrix_from_pose_msg(msg.pose))
        with self._lock_tf:
            self.se3_ee_in_map_pose = se3

    def get_base_in_map_pose(self):
        with self._lock_tf:
            return self.se3_base

    def get_camera_in_map_pose(self):
        with self._lock_tf:
            return self.se3_camera_in_map_pose

    def get_camera_in_base_pose(self):
        with self._lock_tf:
            return self.se3_camera_in_base_pose

    def get_camera_in_arm_pose(self):
        with self._lock_tf:
            return self.se3_camera_in_arm_pose
    
    def get_ee_in_map_pose(self):
        with self._lock_tf:
            return self.se3_ee_in_map_pose

    def get_rtabmapdata(self):
        with self._lock_rtab:
            return self.rtabmapdata


    # def _camera_pose_callback(self, msg: PoseStamped):
    #     """camera pose from CameraPosePublisher, which reads from tf"""
    #     self._last_camera_update_timestamp = msg.header.stamp
    #     self.se3_camera_pose = sp.SE3(matrix_from_pose_msg(msg.pose))

    # def _js_callback(self, msg):
    #     """Read in current joint information from ROS topics and update state"""
    #     n = len(ARM_JOINTS)
    #     pos, vel, trq = np.zeros(n), np.zeros(n), np.zeros(n)
    #     joint_status = {}
    #     for name, p, v, e in zip(msg.name, msg.position, msg.velocity, msg.effort):
    #         # Update joint status dictionary with name and position only
    #         joint_status[name] = p
            
    #         if name in ARM_JOINTS:
    #             joint_idx = ARM_JOINTS.index(name)
    #             pos[joint_idx] = p
    #             vel[joint_idx] = v
    #             trq[joint_idx] = e
        
    #     # with self._lock_js:
    #         # self.arm_pos, self.arm_vel, self.arm_frc = pos, vel, trq
    #         # self.arm_joint_status = joint_status
    #     self.arm_pos, self.arm_vel, self.arm_frc = pos, vel, trq
    #     self.arm_joint_status = joint_status
            

    def _construct_single_joint_ros_goal(self, joint_name, position, goal_time_tolerance=1):
        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.goal_time_tolerance = Duration(seconds=goal_time_tolerance)
        trajectory_goal.trajectory.joint_names = [
            joint_name,
        ]
        msg = JointTrajectoryPoint()
        msg.positions = [position]
        trajectory_goal.trajectory.points = [msg]
        trajectory_goal.trajectory.header.stamp = self.get_clock().now().to_msg()
        return trajectory_goal

    def _interp(self, x1, x2, num_steps=10):
        diff = x2 - x1
        rng = np.arange(num_steps + 1) / num_steps
        rng = rng[:, None].repeat(3, axis=1)
        diff = diff[None].repeat(num_steps + 1, axis=0)
        x1 = x1[None].repeat(num_steps + 1, axis=0)
        return x1 + (rng * diff)

    def goto(self, q, move_base=False, wait=False, max_wait_t=10.0, verbose=False):
        """some of these params are unsupported"""
        goal = self.config_to_ros_trajectory_goal(q)
        if wait:
            self.trajectory_client.send_goal(goal)
        else:
            self.trajectory_client.send_goal_async(goal)
        # self.trajectory_client.send_goal(goal)
        # if wait:
        #     self.trajectory_client.wait_for_result()
        return True

    def _grasp_ready_callback(self, empty_msg):
        self.grasp_ready = True

    def _grasp_result_callback(self, float_msg):
        self.location_above_surface_m = float_msg.data
        self.grasp_complete = True

    def trigger_grasp(self, x, y, z):
        """Calls FUNMAP based grasping"""
        # 1. Enable the grasp node
        assert self.grasp_ready is None
        assert self.grasp_complete is None
        assert self.location_above_surface_m is None
        self.grasp_enable_pub.publish(Empty())
        self.place_disable_pub.publish(Empty())

        # 2. Wait until grasp node ready
        rate = self.create_rate(5)
        while self.grasp_ready is None:
            rate.sleep()

        # 3. Call the trigger topic
        goal_point = PointStamped()
        goal_point.header.stamp = self.get_clock().now().to_msg()
        goal_point.header.frame_id = "map"
        goal_point.point.x = x
        goal_point.point.y = y
        goal_point.point.z = z
        self.grasp_trigger_pub.publish(goal_point)

        # 4. Wait for grasp to complete
        print(" - Waiting for grasp to complete")
        rate = self.create_rate(5)
        while self.grasp_complete is None:
            rate.sleep()
        assert self.location_above_surface_m is not None

        # 5. Disable the grasp node
        self.grasp_disable_pub.publish(Empty())

        self.grasp_ready = None
        self.grasp_complete = None
        return

    def _place_ready_callback(self, empty_msg):
        self.place_ready = True

    def _place_result_callback(self, msg):
        self.place_complete = True

    def get_pose_graph(self):
        """Get robot's pose graph"""
        return self._pose_graph

    def update_pose_graph(self, new_poses):
        """Thread-safe update of pose graph
        
        Args:
            new_poses: Dictionary of node_id -> pose_dict
        """
        if self._pose_graph is None:
            self._pose_graph = {}
        for nid, p in new_poses.items():
            self._pose_graph[nid] = p

    def get_local_tf_graph(self):
        return self._local_tf_graph

    def update_local_tf_graph(self, node_id, new_tf):
        if self._local_tf_graph is None:
            self._local_tf_graph = {}
        self._local_tf_graph[node_id] = new_tf



    def trigger_placement(self, x, y, z):
        """Calls FUNMAP based placement"""
        # 1. Enable the place node
        assert self.place_ready is None
        assert self.place_complete is None
        assert self.location_above_surface_m is not None
        self.grasp_disable_pub.publish(Empty())
        msg = Float32()
        msg.data = self.location_above_surface_m
        self.place_enable_pub.publish(msg)

        # 2. Wait until place node ready
        rate = self.create_rate(5)
        while self.place_ready is None:
            rate.sleep()

        # 3. Call the trigger topic
        goal_point = PointStamped()
        goal_point.header.stamp = self.get_clock().now().to_msg()
        goal_point.header.frame_id = "map"
        goal_point.point.x = x
        goal_point.point.y = y
        goal_point.point.z = z
        self.place_trigger_pub.publish(goal_point)

        # 4. Wait for grasp to complete
        print(" - Waiting for place to complete")
        rate = self.create_rate(5)
        while self.place_complete is None:
            rate.sleep()

        # 5. Disable the place node
        self.place_disable_pub.publish(Empty())

        self.location_above_surface_m = None
        self.place_ready = None
        self.place_complete = None
        return


if __name__ == "__main__":
    rclpy.init()
    ros_interface = DreamRosInterface()
    rclpy.spin(ros_interface)
    ros_interface.destroy_node()
    rclpy.shutdown()