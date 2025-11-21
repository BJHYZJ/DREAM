# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 Hello Robot under MIT license

import sys
import threading
import time
import timeit
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import click
import numpy as np
import zmq
from termcolor import colored

import dream.motion.constants as constants
import dream.motion.conversions as conversions
import dream.utils.compression as compression
from dream.core.interfaces import ContinuousNavigationAction, Observations, StateObservations, ServoObservations
from dream.core.parameters import Parameters, get_parameters
from dream.core.robot import AbstractRobotClient
from dream.motion import PlanResult
from dream.motion.kinematics import RangerxARMKinematics
from dream.motion.constants import DreamIdx
from dream.utils.geometry import (
    angle_difference,
    posquat2sophus,
    sophus2posquat,
    xyt_base_to_global,
)
from dream.utils.image import Camera
from dream.utils.logger import Logger
from dream.utils.memory import lookup_address
from dream.utils.point_cloud import show_point_cloud
from dream.utils.geometry import pose2sophus, sophus2xyt

logger = Logger(__name__)

# TODO: debug code - remove later if necessary
# import faulthandler
# faulthandler.enable()


class DreamRobotZmqClient(AbstractRobotClient):
    num_state_report_steps: int = 10000

    _head_pan_min = -np.pi
    _head_pan_max = np.pi / 4
    _head_tilt_min = -np.pi
    _head_tilt_max = 0

    def _create_recv_socket(
        self,
        port: int,
        robot_ip: str,
        use_remote_computer: bool,
        message_type: Optional[str] = "observations",
    ):
        # Receive state information
        recv_socket = self.context.socket(zmq.SUB)
        recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        recv_socket.setsockopt(zmq.SNDHWM, 1)
        recv_socket.setsockopt(zmq.RCVHWM, 1)
        recv_socket.setsockopt(zmq.CONFLATE, 1)

        ip_address = lookup_address(robot_ip, use_remote_computer)
        if ip_address is None:
            print()
            logger.error("No robot IP address found. Please provide a robot IP address.")
            logger.error("You can do so with:")
            logger.error("    dream.app.<this app> --robot_ip <robot_ip>")
            logger.error("Or with:")
            logger.error("    ./scripts/set_robot_ip.sh <robot_ip>")
            print()
            sys.exit(1)
        recv_address = ip_address + ":" + str(port)
        print(f"Connecting to {recv_address} to receive {message_type}...")
        recv_socket.connect(recv_address)

        return recv_socket

    def get_zmq_context(self) -> zmq.Context:
        """Get the ZMQ context for the client.

        Returns:
            zmq.Context: The ZMQ context
        """
        return self.context

    def __init__(
        self,
        robot_ip: str = "",
        recv_port: int = 4401,
        send_port: int = 4402,
        recv_state_port: int = 4403,
        recv_servo_port: int = 4404,
        pub_obs_port: int = 4450,
        output_path: Path = None,
        parameters: Parameters = None,
        use_remote_computer: bool = True,
        urdf_path: str = "",
        ik_type: str = "pinocchio",
        visualize_ik: bool = False,
        grasp_frame: Optional[str] = None,
        ee_link_name: Optional[str] = None,
        manip_mode_controlled_joints: Optional[List[str]] = None,
        start_immediately: bool = True,
        enable_rerun_server: bool = True,
        resend_all_actions: bool = False,
        publish_observations: bool = False,
    ):
        """
        Create a client to communicate with the robot over ZMQ.

        Args:
            robot_ip: The IP address of the robot
            recv_port: The port to receive observations on
            send_port: The port to send actions to on the robot
            use_remote_computer: Whether to use a remote computer to connect to the robot
            urdf_path: The path to the URDF file for the robot
            ik_type: The type of IK solver to use
            visualize_ik: Whether to visualize the IK solution
            grasp_frame: The frame to use for grasping
            ee_link_name: The name of the end effector link
            manip_mode_controlled_joints: The joints to control in manipulation mode
        """
        self.recv_port = recv_port
        self.send_port = send_port
        self.reset()

        # Load parameters
        if parameters is None:
            parameters = get_parameters("default_planner.yaml")
        self._parameters = parameters

        # Variables we set here should not change
        self._iter = -1  # Tracks number of actions set, never reset this

        self._started = False

        # Resend all actions immediately - helps if we are losing packets or something?
        self._resend_all_actions = resend_all_actions
        self._publish_observations = (
            publish_observations or self.parameters["agent"]["use_realtime_updates"]
        )
        self._warning_on_out_of_date_state = -1

        self._moving_threshold = parameters["motion"]["moving_threshold"]
        self._angle_threshold = parameters["motion"]["angle_threshold"]
        self._min_steps_not_moving = parameters["motion"]["min_steps_not_moving"]

        # Read in joint tolerances from config file
        # self._head_pan_tolerance = float(parameters["motion"]["joint_tolerance"]["head_pan"])
        # self._head_tilt_tolerance = float(parameters["motion"]["joint_tolerance"]["head_tilt"])
        # self._head_not_moving_tolerance = float(
        #     parameters["motion"]["joint_thresholds"]["head_not_moving_tolerance"]
        # )
        self._arm_joint_tolerance = float(parameters["motion"]["joint_tolerance"]["joint"])
        self._gripper_tolerance = float(parameters["motion"]["joint_tolerance"]["gripper"])
        # self._lift_joint_tolerance = float(parameters["motion"]["joint_tolerance"]["lift"])
        # self._base_x_joint_tolerance = float(parameters["motion"]["joint_tolerance"]["base_x"])
        # self._wrist_roll_joint_tolerance = float(
        #     parameters["motion"]["joint_tolerance"]["wrist_roll"]
        # )
        # self._wrist_pitch_joint_tolerance = float(
        #     parameters["motion"]["joint_tolerance"]["wrist_pitch"]
        # )
        # self._wrist_yaw_joint_tolerance = float(
        #     parameters["motion"]["joint_tolerance"]["wrist_yaw"]
        # )

        # Robot model
        self._robot_model = RangerxARMKinematics(
            # urdf_path=urdf_path,
            # ik_type=ik_type,
            # visualize=visualize_ik,
            # grasp_frame=grasp_frame,
            # ee_link_name=ee_link_name,
            # manip_mode_controlled_joints=manip_mode_controlled_joints,
        )

        # Create ZMQ sockets
        self.context = zmq.Context()

        print("-------- HOME-ROBOT ROS2 ZMQ CLIENT --------")
        self.recv_socket = self._create_recv_socket(
            self.recv_port, robot_ip, use_remote_computer, message_type="observations"
        )
        self.recv_state_socket = self._create_recv_socket(
            recv_state_port, robot_ip, use_remote_computer, message_type="low level state"
        )
        self.recv_servo_socket = self._create_recv_socket(
            recv_servo_port, robot_ip, use_remote_computer, message_type="visual servoing data"
        )

        # Send actions back to the robot for execution
        self.send_socket = self.context.socket(zmq.PUB)
        self.send_socket.setsockopt(zmq.SNDHWM, 1)
        self.send_socket.setsockopt(zmq.RCVHWM, 1)

        self.send_address = (
            lookup_address(robot_ip, use_remote_computer) + ":" + str(self.send_port)
        )

        print(f"Connecting to {self.send_address} to send action messages...")
        self.send_socket.connect(self.send_address)
        print("...connected.")

        self._obs_lock = Lock()
        self._act_lock = Lock()
        self._state_lock = Lock()
        self._servo_lock = Lock()
        self._send_lock = Lock()

        if enable_rerun_server:
            from dream.visualization.rerun import RerunVisualizer
            
            if output_path is None:
                current_datetime = datetime.now()
                output_path = Path("rerun_log/debug_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S"))

            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)

            self._rerun = RerunVisualizer(output_path=output_path, footprint=self._robot_model.get_footprint())
        else:
            self._rerun = None
            self._rerun_thread = None

        if start_immediately:
            self.start()

    @property
    def parameters(self) -> Parameters:
        return self._parameters


    def get_head_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the RGB and depth images from camera.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The RGB and depth images
        """
        with self._servo_lock:
            if self._servo is None:
                return None, None
            rgb = self._servo["head_rgb"]
            depth = self._servo["head_depth"]
        return rgb, depth

    def get_joint_state(self, timeout: float = 5.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the current joint positions, velocities, and efforts"""
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None, None, None
            joint_states = self._state.joint_states
            joint_velocities = self._state.joint_velocities
            joint_forces = self._state.joint_forces
        return joint_states, joint_velocities, joint_forces

    def get_joint_position(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current joint positions"""
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            joint_positions = self._state.joint_positions
        return joint_positions

    def get_arm_joint_state(self, timeout: float = 5.0) -> np.ndarray:
        joint_states, _, _ = self.get_joint_state(timeout=timeout)
        return np.array(self._extract_joint_state(joint_states))

    def get_joint_velocities(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current joint velocities"""
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            joint_velocities = self._state.joint_velocities
        return joint_velocities

    def get_joint_efforts(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current joint efforts from the robot.

        Args:
            timeout: How long to wait for the observation

        Returns:
            np.ndarray: The joint efforts as an array of floats
        """

        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            joint_efforts = self._state.joint_efforts
        return joint_efforts


    def get_base_in_map_xyt(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current pose of the base.

        Args:
            timeout: How long to wait for the observation

        Returns:
            np.ndarray: The base pose as [x, y, theta]
        """
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            base_in_map_pose = self._state.base_in_map_pose
            xyt = sophus2xyt(pose2sophus(base_in_map_pose))

        return xyt


    def get_arm_base_in_map_xyt(self, timeout: float = 5.0) -> np.ndarray:
        """Get the current pose of the base.

        Args:
            timeout: How long to wait for the observation

        Returns:
            np.ndarray: The base pose as [x, y, theta]
        """
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            arm_base_in_map_pose = self._state.arm_base_in_map_pose
            xyt = sophus2xyt(pose2sophus(arm_base_in_map_pose))
        return xyt


    def get_gripper_position(self):
        """Get the current position of the gripper.

        Returns:
            float: The position of the gripper
        """
        joint_state = self.get_joint_position()
        return joint_state[DreamIdx.GRIPPER]



    def _extract_joint_state(self, joint_states):
        """Helper to convert from the general-purpose config including full robot state, into the command space used in just the manip controller. Extracts just lift/arm/wrist information."""
        return joint_states[constants.ARM_INDEX]


    def look_front(self, speed=50, blocking: bool=True, timeout: float = 10.0):
        """Let robot look to its front."""
        self.arm_to(
            angle=constants.look_front,
            speed=speed,
            blocking=blocking,
            timeout=timeout,
            reliable=True,
        )

    # def get_camera_in_ee(self, timeout: float = 5.0) -> np.ndarray:
    #     t0 = timeit.default_timer()
    #     with self._state_lock:
    #         while self._state is None:
    #             time.sleep(1e-4)
    #             if timeit.default_timer() - t0 > timeout:
    #                 logger.error("Timeout waiting for state message")
    #                 return None
    #         camera_in_arm_base_pose = self._state.camera_in_arm_base_pose
    #         ee_in_arm_base_pose = self._state.ee_in_arm_base_pose
    #     return np.linalg.inv(ee_in_arm_base_pose) @ camera_in_arm_base_pose

    def get_camera_in_arm_base(self, timeout: float = 5.0) -> np.ndarray:
        return self.get_transform(transfrom_name="camera_in_arm_base_pose", timeout=timeout)

    def get_ee_in_arm_base(self, timeout: float = 5.0) -> np.ndarray:
        return self.get_transform(transfrom_name="ee_in_arm_base_pose", timeout=timeout)

    def get_transform(self, transfrom_name: str, timeout: float = 5.0) -> np.ndarray:
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            if not hasattr(self._state, transfrom_name):
                logger.error(f"State message does not contain transform '{transfrom_name}'")
                return None
            transform = getattr(self._state, transfrom_name)
        return transform


    def look_at_target(
        self, 
        tar_in_map: np.ndarray,
        blocking: bool=True, 
        timeout: float=10.0
    ):
        self.look_at_target_tilt(tar_in_map, blocking=blocking, timeout=timeout)
        self.look_at_target_pan(tar_in_map, blocking=blocking, timeout=timeout)
        # sleep for a while ensure image is newest
        time.sleep(1)
        print(f"look at target_point in map frame: {tar_in_map}")


    def look_at_target_tilt(
        self, 
        tar_in_map: np.ndarray, 
        blocking: bool=True,
        timeout: float=10.0
    ):
        """Let robot look at the target point by rotate TILT."""
        if isinstance(tar_in_map, list):
            tar_in_map = np.array(tar_in_map)
        assert len(tar_in_map) == 3, "target point must be a vector of size 3"
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            joint_states = self._state.joint_states
            # arm_base_in_map_pose = self._state.arm_base_in_map_pose
            camera_in_arm_base_pose = self._state.camera_in_arm_base_pose
            camera_in_map_pose = self._state.camera_in_map_pose
            ee_in_arm_base_pose = self._state.ee_in_arm_base_pose

        camera_K = self.get_servo_camera_K()

        tar_in_map_h = np.array([tar_in_map[0], tar_in_map[1], tar_in_map[2], 1.0], dtype=float)
        tar_in_cam= (np.linalg.inv(camera_in_map_pose) @ tar_in_map_h)[:3]

        arm_angles_deg = self._robot_model.compute_look_at_target_tilt(
            arm_angles_deg=joint_states[3:9],
            target_in_camera=tar_in_cam,
            camera_in_arm_base_pose=camera_in_arm_base_pose,
            ee_in_arm_base_pose=ee_in_arm_base_pose,
            camera_K=camera_K,
        )
        self.arm_to(arm_angles_deg, blocking=blocking, timeout=timeout)


    def look_at_target_pan(
        self, 
        tar_in_map: np.ndarray,
        blocking: bool=True,
        timeout: float=10.0
    ):
        """Let robot look at the target point by rotate PAN."""
        if isinstance(tar_in_map, list):
            tar_in_map = np.array(tar_in_map)
        assert len(tar_in_map) == 3, "target point must be a vector of size 3"
        t0 = timeit.default_timer()
        with self._state_lock:
            while self._state is None:
                time.sleep(1e-4)
                if timeit.default_timer() - t0 > timeout:
                    logger.error("Timeout waiting for state message")
                    return None
            joint_states = self._state.joint_states
            # arm_base_in_map_pose = self._state.arm_base_in_map_pose
            camera_in_arm_base_pose = self._state.camera_in_arm_base_pose
            camera_in_map_pose = self._state.camera_in_map_pose
            ee_in_arm_base_pose = self._state.ee_in_arm_base_pose
        
        camera_K = self.get_servo_camera_K()

        tar_in_map_h = np.array([tar_in_map[0], tar_in_map[1], tar_in_map[2], 1.0], dtype=float)
        tar_in_cam= (np.linalg.inv(camera_in_map_pose) @ tar_in_map_h)[:3]

        arm_angles_deg = self._robot_model.compute_look_at_target_pan(
            arm_angles_deg=joint_states[3:9],
            target_in_camera=tar_in_cam,
            camera_in_arm_base_pose=camera_in_arm_base_pose,
            ee_in_arm_base_pose=ee_in_arm_base_pose,
            camera_K=camera_K,
        )
        self.arm_to(arm_angles_deg, blocking=blocking, timeout=timeout)


    def base_to(
        self,
        xyt: Union[ContinuousNavigationAction, np.ndarray],
        relative: bool = False,
        timeout: float = 10.0,
        verbose: bool = False,
        blocking: bool = True,
        reliable: bool = True,
    ):
        """Move to xyt in global coordinates or relative coordinates.

        Args:
            xyt: The xyt position to move to
            relative: Whether the position is relative to the current position
            blocking: Whether to block until the motion is complete
            timeout: How long to wait for the motion to complete
            verbose: Whether to print out debug information
            reliable: Whether to resend the action if it is not received
        """
        if not self.in_navigation_mode():
            self.switch_to_navigation_mode()

        if isinstance(xyt, ContinuousNavigationAction):
            _xyt = xyt.xyt
        else:
            _xyt = xyt
        assert len(_xyt) == 3, "xyt must be a vector of size 3"
        # If it's relative, compute the relative position right now - this helps handle network issues
        if relative:
            current_xyt = self.get_base_in_map_xyt()
            if verbose:
                print("Current pose", current_xyt)
            _xyt = xyt_base_to_global(_xyt, current_xyt)
            if verbose:
                print("Goal pose in global coordinates", _xyt)

        if blocking and not reliable:
            logger.warning("Sending blocking commands without reliable is not recommended")

        # We never send a relative motion over wireless - this is because we can run into timing issues.
        # Instead, we always send the absolute position and let the robot handle the motions itself.
        next_action = {"xyt": _xyt, "nav_relative": False, "nav_blocking": blocking}
        if self._rerun:
            self._rerun.update_nav_goal(_xyt)

        # If we are not in navigation mode, switch to it
        # Send an action to the robot
        # Resend it to make sure it arrives, if we are not making a relative motion
        # If we are blocking, wait for the action to complete with a timeout
        action = self.send_action(next_action, timeout=timeout, verbose=verbose, reliable=reliable)

        # Make sure we had time to read
        if blocking:
            block_id = action["step"]
            time.sleep(0.1)
            # Now, wait for the command to finish
            self._wait_for_base_motion(
                block_id,
                goal_angle=_xyt[2],
                verbose=verbose,
                timeout=timeout,
                # resend_action=action,
                # resend_action=current_action,
            )
            

    def arm_to(
        self,
        angle: np.ndarray,
        speed: int=20,
        blocking: bool=True,
        timeout: float=10.0,
        reliable: bool=True,
        # sleep_time: int=1,
    ):
        """Move the arm to a particular configuration. servo angle control

        Args:
            angle: The angle of the xarm6
            blocking: Whether to block until the motion is complete
            timeout: How long to wait for the motion to complete
            reliable: Whether to resend the action if it is not received
        """
        if not self.in_manipulation_mode():
            self.switch_to_manipulation_mode()
        next_action = {"servo_angle": angle, "speed": speed, "wait": blocking}
        self.send_action(next_action, timeout=timeout, reliable=reliable)

        if blocking:
            t0 = timeit.default_timer()
            steps = 0
            while not self._finish:
                if steps % 40 == 39:
                    self.send_action(next_action, timeout=timeout, reliable=True)
                
                # Get current joint state
                joint_states, _, _ = self.get_joint_state()
                if joint_states is None:
                    time.sleep(0.01)
                    steps += 1
                    continue
                
                # Compute error for all 6 joints
                error = np.linalg.norm(joint_states[3:9] - angle)
                
                if error < 0.1:  # 1 degree threshold
                    # print(f"[Camera Aim] ✅ Reached target")
                    # waiting for 1 second to make sure data transport over
                    # time.sleep(sleep_time)
                    return True
                
                steps += 1
                time.sleep(0.01)
                
                t1 = timeit.default_timer()
                if t1 - t0 > timeout:
                    logger.error("Timeout waiting for camera to aim at target")
                    return False
            
            return False

    def gripper_to(
        self, 
        position: float,
        blocking: bool=False, 
        reliable: bool=True,
    ):
        """Set the position of the gripper."""
        next_action = {"gripper": position, "wait": blocking}
        self.send_action(next_action, reliable=reliable)
        if blocking:
            time.sleep(0.1)

    def set_velocity(self, v: float, w: float):
        """Move to xyt in global coordinates or relative coordinates.

        Args:
            v: The velocity in the x direction
            w: The angular velocity
        """
        next_action = {"v": v, "w": w}
        self.send_action(next_action)

    def reset(self):
        """Reset everything in the robot's internal state"""
        self._control_mode = None
        self._obs = None  # Full observation includes high res images and camera pose, no EE camera
        self._pose_graph = None
        self._state = None  # Low level state includes joint angles and base XYT
        self._servo = None  # Visual servoing state includes smaller images
        self._thread = None
        self._state_thread = None
        self._finish = False
        self._last_step = -1  # just update by state
        # ensure obs and pose graph is newest (legacy seq)
        # Local versions based on content changes
        self._obs_version = -1
        self._delivered_obs_version = -1
        self._pose_graph_version = -1
        self._delivered_pose_graph_version = -1

    def open_gripper(
        self,
        blocking: bool=True,
        reliable: bool=True
    ) -> bool:
        """Open the gripper based on hard-coded presets."""
        gripper_target = self._robot_model.GRIPPER_OPEN
        print("[ZMQ CLIENT] Opening gripper to", gripper_target)
        self.gripper_to(gripper_target, blocking=blocking, reliable=reliable)

    def close_gripper(
        self,
        blocking: bool=False,
        reliable: bool=True
    ) -> bool:
        """Close the gripper based on hard-coded presets."""
        gripper_target = self._robot_model.GRIPPER_CLOSED
        print("[ZMQ CLIENT] Closing gripper to", gripper_target)
        self.gripper_to(gripper_target, blocking=blocking, reliable=reliable)

    def switch_to_navigation_mode(self):
        """Velocity control of the robot base."""
        next_action = {"control_mode": "navigation", "step": self._iter}
        action = self.send_action(next_action)
        self._wait_for_mode("navigation")
        assert self.in_navigation_mode()

    def switch_to_manipulation_mode(self, verbose: bool = False):
        """Move the robot to manipulation mode.

        Args:
            verbose: Whether to print out debug information
        """
        next_action = {"control_mode": "manipulation", "step": self._iter}
        action = self.send_action(next_action)
        if verbose:
            logger.info("Waiting for manipulation mode")
        self._wait_for_mode("manipulation", verbose=verbose)

    def move_to_nav_posture(self) -> None:
        """Move the robot to the navigation posture. This is where the head is looking forward and the arm is tucked in."""
        next_action = {"posture": "navigation", "step": self._iter}
        next_action = self.send_action(next_action)
        # self._wait_for_head(constants.STRETCH_NAVIGATION_Q, resend_action=next_action)
        self._wait_for_mode("navigation")
        # self._wait_for_arm(constants.STRETCH_NAVIGATION_Q)
        assert self.in_navigation_mode()

    def move_to_manip_posture(self):
        """This is the pregrasp posture where the head is looking down and right and the arm is tucked in."""
        next_action = {"posture": "manipulation", "step": self._iter}
        self.send_action(next_action)
        time.sleep(0.1)
        self._wait_for_head(constants.STRETCH_PREGRASP_Q, resend_action=next_action)
        self._wait_for_mode("manipulation")
        # self._wait_for_arm(constants.STRETCH_PREGRASP_Q)
        assert self.in_manipulation_mode()

    def pause_slam(self, timeout: float=2.0, reliable: bool=True) -> None:
        """Pause SLAM updates on the robot."""
        next_action = {"slam_pause": True, "slam_timeout": timeout}
        self.send_action(next_action, timeout=timeout, reliable=reliable)

    def resume_slam(self, timeout: float=2.0, reliable: bool=True) -> None:
        """Resume SLAM updates on the robot."""
        next_action = {"slam_resume": True, "slam_timeout": timeout}
        self.send_action(next_action, timeout=timeout, reliable=reliable)


    def _wait_for_mode(
        self,
        mode,
        resend_action: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        timeout: float = 20.0,
        time_required: float = 0.05,
    ) -> bool:
        """
        Wait for the robot to switch to a particular control mode. Will throw an exception if mode switch fails; probably means a packet was dropped.

        Args:
            mode(str): The mode to wait for
            resend_action(dict): The action to resend if the robot is not moving. If none, do not resend.
            verbose(bool): Whether to print out debug information
            timeout(float): How long to wait for the robot to switch modes

        Returns:
            bool: Whether the robot successfully switched to the target mode
        """
        t0 = timeit.default_timer()
        mode_t0 = None
        while True:
            with self._state_lock:
                if verbose:
                    print(f"Waiting for mode {mode} current mode {self._control_mode} {mode_t0}")
                if self._control_mode == mode and mode_t0 is None:
                    mode_t0 = timeit.default_timer()
                elif self._control_mode != mode:
                    mode_t0 = None
            # Make sure we are in the mode for at least time_required seconds
            # This is to handle network delays
            if mode_t0 is not None and timeit.default_timer() - mode_t0 > time_required:
                break
            if resend_action is not None:
                self.send_socket.send_pyobj(resend_action)
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > timeout:
                raise RuntimeError(f"Timeout waiting for mode {mode}: {t1 - t0} seconds")

        assert self._control_mode == mode
        return True

    def _wait_for_base_motion(
        self,
        block_id: int,
        verbose: bool = True,
        timeout: float = 10.0,
        moving_threshold: Optional[float] = None,
        angle_threshold: Optional[float] = None,
        min_steps_not_moving: Optional[int] = None,
        goal_angle: Optional[float] = None,
        goal_angle_threshold: Optional[float] = 0.15,
        resend_action: Optional[dict] = None,
    ) -> None:
        """Wait for the navigation action to finish.

        Args:
            block_id(int): The unique, tracked integer id of the action to wait for
            verbose(bool): Whether to print out debug information
            timeout(float): How long to wait for the action to finish
            moving_threshold(float): How far the robot must move to be considered moving
            angle_threshold(float): How far the robot must rotate to be considered moving
            min_steps_not_moving(int): How many steps the robot must not move for to be considered stopped
            goal_angle(float): The goal angle to reach
            goal_angle_threshold(float): The threshold for the goal angle
            resend_action(dict): The action to resend if the robot is not moving. If none, do not resend.
        """
        print("=" * 20, f"Waiting for {block_id} at goal", "=" * 20)
        last_pos = None
        last_ang = None
        last_obs_t = None
        not_moving_count = 0
        if moving_threshold is None:
            moving_threshold = self._moving_threshold
        if angle_threshold is None:
            angle_threshold = self._angle_threshold
        if min_steps_not_moving is None:
            min_steps_not_moving = self._min_steps_not_moving
        t0 = timeit.default_timer()
        close_to_goal = False

        while not self._finish:

            # Minor delay at the end - give it time to get new messages
            time.sleep(0.01)

            if not self.is_state_up_to_date(min_step=block_id):
                if verbose:
                    print("Waiting for latest state message")
                continue

            with self._state_lock:
                if self._state is None:
                    print("waiting for state")
                    continue

            xyt = self.get_base_in_map_xyt()
            pos = xyt[:2]
            ang = xyt[2]
            obs_t = timeit.default_timer()

            if not self.at_goal():
                t0 = timeit.default_timer()
                continue

            moved_dist = np.linalg.norm(pos - last_pos) if last_pos is not None else float("inf")
            angle_dist = angle_difference(ang, last_ang) if last_ang is not None else float("inf")
            if goal_angle is not None:
                angle_dist_to_goal = angle_difference(ang, goal_angle)
                at_goal = angle_dist_to_goal < goal_angle_threshold
            else:
                at_goal = True

            moved_speed = (
                moved_dist / (obs_t - last_obs_t) if last_obs_t is not None else float("inf")
            )
            angle_speed = (
                angle_dist / (obs_t - last_obs_t) if last_obs_t is not None else float("inf")
            )

            not_moving = (
                last_pos is not None
                and moved_speed < moving_threshold
                and angle_speed < angle_threshold
            )
            if not_moving:
                not_moving_count += 1
            else:
                not_moving_count = 0

            # Check if we are at the goal
            # If we are at the goal, we can stop if we are not moving
            last_pos = pos
            last_ang = ang
            last_obs_t = obs_t
            close_to_goal = at_goal
            if verbose:
                print(
                    f"Waiting for step={block_id} {self._last_step} prev={self._last_step} at {pos} moved {moved_dist:0.04f} angle {angle_dist:0.04f} not_moving {not_moving_count} at_goal {self._state.at_goal}"
                )
                print(min_steps_not_moving, self._last_step, at_goal)
                if goal_angle is not None:
                    print(f"Goal angle {goal_angle} angle dist to goal {angle_dist_to_goal}")
            if self._last_step >= block_id and at_goal and not_moving_count > min_steps_not_moving:
                if verbose:
                    print("---> At goal")
                break

            # Resend the action if we are not moving for some reason and it's been provided
            if resend_action is not None and not close_to_goal:
                # Resend the action
                self.send_action(resend_action)

            t1 = timeit.default_timer()
            if t1 - t0 > timeout:
                print(f"Timeout waiting for block with step id = {block_id}")
                break
                # raise RuntimeError(f"Timeout waiting for block with step id = {block_id}")

    def in_manipulation_mode(self) -> bool:
        """is the robot ready to grasp"""
        return self._control_mode == "manipulation"

    def in_navigation_mode(self) -> bool:
        """Returns true if we are navigating (robot head forward, velocity control on)"""
        return self._control_mode == "navigation"

    def last_motion_failed(self) -> bool:
        """Override this if you want to check to see if a particular motion failed, e.g. it was not reachable and we don't know why."""
        return False

    def get_robot_model(self):
        """return a model of the robot for planning"""
        return self._robot_model

    def _latest_command_step(self) -> int:
        """Return the most recent command step we expect the robot to have executed."""
        return max(self._iter - 1, -1)

    def is_state_up_to_date(self, min_step: Optional[int] = None) -> bool:
        """Check if the low-level state stream has caught up to a requested step."""
        with self._state_lock:
            state_step = None if self._state is None else self._state.step

        if state_step is None:
            return False

        if min_step is None:
            min_step = self._latest_command_step()

        return state_step >= min_step

    def send_message(self, message: dict):
        """Send a message to the robot"""
        with self._send_lock:
            self.send_socket.send_pyobj(message)


    def at_goal(self) -> bool:
        """Check if the robot is at the goal.

        Returns:
            at_goal (bool): whether the robot is at the goal
        """
        with self._state_lock:
            if self._state is None:
                return False
            return self._state.at_goal

    def save_map(self, filename: str):
        """Save the current map to a file.

        Args:
            filename (str): the filename to save the map to
        """
        next_action = {"save_map": filename}
        self.send_action(next_action)

    def load_map(self, filename: str):
        """Load a map from a file.

        Args:
            filename (str): the filename to load the map from
        """
        next_action = {"load_map": filename}
        self.send_action(next_action)


    def execute_trajectory(
        self,
        trajectory: List[np.ndarray],
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.1,
        spin_rate: int = 10,
        verbose: bool = False,
        per_waypoint_timeout: float = 10.0,
        final_timeout: float = 10.0,
        relative: bool = False,
        blocking: bool = False,
    ):
        """Execute a multi-step trajectory; this is always blocking since it waits to reach each one in turn."""

        if isinstance(trajectory, PlanResult):
            trajectory = [pt.state for pt in trajectory.trajectory]

        if relative:
            raise NotImplementedError("Relative trajectories not yet supported")

        for i, pt in enumerate(trajectory):
            assert (
                len(pt) == 3 or len(pt) == 2
            ), "base trajectory needs to be 2-3 dimensions: x, y, and (optionally) theta"
            self.base_to(pt, relative, blocking=False, reliable=False)
            print("Moving to", pt)
            last_waypoint = i == len(trajectory) - 1
            self.base_to(
                pt,
                relative=False,
                blocking=last_waypoint,
                timeout=final_timeout if last_waypoint else per_waypoint_timeout,
                verbose=verbose,
                reliable=True if last_waypoint else False,
            )
            if not last_waypoint:
                self.wait_for_waypoint(
                    pt,
                    pos_err_threshold=pos_err_threshold,
                    rot_err_threshold=rot_err_threshold,
                    rate=spin_rate,
                    verbose=verbose,
                    timeout=per_waypoint_timeout,
                )

    def wait_for_waypoint(
        self,
        xyt: np.ndarray,
        rate: int = 10,
        pos_err_threshold: float = 0.2,
        rot_err_threshold: float = 0.75,
        verbose: bool = False,
        timeout: float = 20.0,
    ) -> bool:
        """Wait until the robot has reached a configuration... but only roughly. Used for trajectory execution.

        Parameters:
            xyt: se(2) base pose in world coordinates to go to
            rate: rate at which we should check to see if done
            pos_err_threshold: how far robot can be for this waypoint
            verbose: prints extra info out
            timeout: aborts at this point

        Returns:
            success: did we reach waypoint in time"""
        _delay = 1.0 / rate
        xy = xyt[:2]
        if verbose:
            print(f"Waiting for {xyt}, threshold = {pos_err_threshold}")
        # Save start time for exiting trajectory loop
        t0 = timeit.default_timer()
        while not self._finish:
            # Loop until we get there (or time out)
            t1 = timeit.default_timer()
            curr = self.get_base_in_map_xyt()
            pos_err = np.linalg.norm(xy - curr[:2])
            rot_err = np.abs(angle_difference(curr[-1], xyt[2]))
            # TODO: code for debugging slower rotations
            # if pos_err < pos_err_threshold and rot_err > rot_err_threshold:
            #     print(f"{curr[-1]}, {xyt[2]}, {rot_err}")
            if verbose:
                logger.info(f"- {curr=} target {xyt=} {pos_err=} {rot_err=}")
            if pos_err < pos_err_threshold and rot_err < rot_err_threshold:
                # We reached the goal position
                return True
            t2 = timeit.default_timer()
            dt = t2 - t1
            if t2 - t0 > timeout:
                logger.warning(
                    "[WAIT FOR WAYPOINT] WARNING! Could not reach goal in time: "
                    + str(xyt)
                    + " "
                    + str(curr)
                )
                return False
            time.sleep(max(0, _delay - (dt)))
        return False

    def set_base_velocity(self, forward: float, rotational: float) -> None:
        """Set the velocity of the robot base.

        Args:
            forward (float): forward velocity
            rotational (float): rotational velocity
        """
        next_action = {"base_velocity": {"v": forward, "w": rotational}}
        self.send_action(next_action)

    def send_action(
        self,
        next_action: Dict[str, Any],
        timeout: float = 5.0,
        verbose: bool = False,
        reliable: bool = True,
    ) -> Dict[str, Any]:
        """Send the next action to the robot. Increment the step counter and wait for the action to finish if it is blocking.

        Args:
            next_action (dict): the action to send
            timeout (float): how long to wait for the action to finish
            verbose (bool): whether to print out debug information
            reliable (bool): whether to resend the action if it is not received

        Returns:
            dict: copy of the action that was sent to the robot.
        """
        if verbose:
            logger.info("-> sending", next_action)
            cur_joints = self.get_joint_position()
            print("Current robot states")
            print(" - base: ", cur_joints[0])
            print(" - base_theta: ", cur_joints[1])
            print(" - joint1: ", cur_joints[2])
            print(" - joint2: ", cur_joints[3])
            print(" - joint3: ", cur_joints[4])
            print(" - joint4: ", cur_joints[5])
            print(" - joint5: ", cur_joints[6])
            print(" - joint6: ", cur_joints[7])
            print(" - gripper: ", cur_joints[8])

        blocking = False
        block_id = None
        with self._act_lock:

            # Send it
            block_id = max(self._iter, self._last_step + 1)
            next_action["step"] = block_id
            self._iter = block_id + 1

            self.send_message(next_action)

            while reliable and self._last_step < block_id:
                # print(next_action)
                self.send_message(next_action)
                time.sleep(0.01)

            # For tracking goal
            if "xyt" in next_action:
                goal_angle = next_action["xyt"][2]
            else:
                goal_angle = None

            # Empty it out for the next one
            current_action = next_action

        # Returns the current action in case we want to do something with it like resend
        return current_action

    def blocking_spin(self, verbose: bool = False, visualize: bool = False):
        """Listen for incoming observations and update internal state"""
        sum_time = 0.0
        steps = 0
        t0 = timeit.default_timer()
        camera = None
        shown_point_cloud = visualize

        while not self._finish:

            output = self.recv_socket.recv_pyobj()  # RtabmapData
            if output is None:
                continue

            # For history nodes that do not have RGB values, both depth and RGB values ​​should be set to None / np.array(B).
            if not output["just_pose_graph"]:
                output["rgb"] = compression.from_array(output["rgb"], is_rgb=True)
                output["depth"] = compression.from_array(output["depth"], is_rgb=False) / 1000
                rgb_height, rgb_width = output["rgb"].shape[:2]

                if camera is None:
                    camera = Camera.from_K(
                        output["camera_K"], width=rgb_width, height=rgb_height
                    )
                output["xyz"] = camera.depth_to_xyz(output["depth"])

                if visualize and not shown_point_cloud:
                    show_point_cloud(output["xyz"], output["rgb"] / 255.0, orig=np.zeros(3))
                    shown_point_cloud = True
                
                self._update_obs(output)

            self._update_pose_graph(output)

            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            if verbose:
                print("Control mode:", self._control_mode)
                print(f"time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}")
            t0 = timeit.default_timer()

    def blocking_spin_state(self, verbose: bool = False):
        """Listen for incoming observations and update internal state"""

        sum_time = 0.0
        steps = 0
        t0 = timeit.default_timer()

        while not self._finish:
            output = self.recv_state_socket.recv_pyobj()
            self._update_state(output)

            t1 = timeit.default_timer()
            dt = t1 - t0
            sum_time += dt
            steps += 1
            if verbose and steps % self.num_state_report_steps == 1:
                print("[STATE] Control mode:", self._control_mode)
                print(
                    f"[STATE] time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}"
                )
            t0 = timeit.default_timer()

    def blocking_spin_servo(self, verbose: bool = False):
        """Listen for servo messages coming from the robot, i.e. low res images for ML state. This is intended to be run in a separate thread.

        Args:
            verbose (bool): whether to print out debug information
        """
        sum_time = 0.0
        steps = 0
        t0 = timeit.default_timer()
        while not self._finish:
            t1 = timeit.default_timer()
            dt = t1 - t0
            output = self.recv_servo_socket.recv_pyobj()
            self._update_servo(output)
            sum_time += dt
            steps += 1
            if verbose and steps % self.num_state_report_steps == 1:
                print(
                    f"[SERVO] time taken = {dt} avg = {sum_time/steps} keys={[k for k in output.keys()]}"
                )
            t0 = timeit.default_timer()

    def blocking_spin_rerun(self) -> None:
        """Use the rerun server so that we can visualize what is going on as the robot takes actions in the world."""
        while not self._finish:
            self._rerun.step(self._obs, self._state, self._servo)


    def get_servo_observation(self):
        """Get the current servo observation.

        Returns:
            ServoObservations: the current servo observation
        """
        with self._servo_lock:
            return self._servo
    
    def get_servo_images(self, compute_xyz=False):
        """Get the current RGB and depth images from the robot.

        Args:
            compute_xyz (bool): whether to compute the XYZ image

        Returns:
            rgb (np.ndarray): the RGB image
            depth (np.ndarray): the depth image
            xyz (np.ndarray): the XYZ image if compute_xyz is True
        """
        servo_obs = self.get_servo_observation()
        if compute_xyz:
            return servo_obs.rgb, servo_obs.depth, servo_obs.camera_in_arm_base_pose, servo_obs.xyz
        else:
            return servo_obs.rgb, servo_obs.depth, servo_obs.camera_in_arm_base_pose      

    def get_servo_camera_K(self):
        """Get the camera intrinsics.

        Returns:
            camera_K (np.ndarray): the camera intrinsics
        """
        servo_obs = self.get_servo_observation()
        return servo_obs.camera_K


    # def get_observation(self, wait_for_new: bool=False):
    def get_observation(self):
        """Get the current observation. This uses the FULL observation track. Expected to be syncd with RGBD.

        Args:
            wait_for_new: if True, block until an observation arrives that has not been delivered yet.
        """
        while True:
            with self._obs_lock:
                if self._obs is not None:
                    # if not wait_for_new:
                    return self._obs
                    # # use local content-based versioning
                    # if (
                    #     self._obs_version is not None
                    #     and self._obs_version != self._delivered_obs_version
                    # ):
                    #     self._delivered_obs_version = self._obs_version
                    #     return self._obs
            time.sleep(0.05)

    # def get_pose_graph(self, *, wait_for_new: bool = False) -> np.ndarray:
    def get_pose_graph(self) -> np.ndarray:
        """Get the robot's SLAM pose graph.

        Args:
            wait_for_new: if True, block until a new pose graph arrives that has not been delivered yet.
        """
        while True:
            with self._obs_lock:
                if self._pose_graph is not None:
                    # if not wait_for_new:
                    return self._pose_graph
                    # # use local content-based versioning
                    # if (
                    #     self._pose_graph_version is not None
                    #     and self._pose_graph_version != self._delivered_pose_graph_version
                    # ):
                    #     self._delivered_pose_graph_version = self._pose_graph_version
                    #     return self._pose_graph
            time.sleep(0.05)

    def _update_obs(self, obs: dict):
        """Update observation internally with lock"""
        with self._obs_lock:
            self._obs = Observations.from_dict(obs)
            self._obs_version = self._obs_version + 1 if self._obs_version is not None else 0

    def _update_pose_graph(self, obs):
        """Update internal pose graph"""
        with self._obs_lock:
            self._pose_graph = {
                "timestamp": obs["timestamp"],
                "pose_graph": obs["pose_graph"],
            }
            self._pose_graph_version = self._pose_graph_version + 1 if self._pose_graph_version is not None else 0

    def _update_state(self, state: dict) -> None:
        """Update state internally with lock. This is expected to be much more responsive than using full observations, which should be reserved for higher level control.

        Args:
            state (dict): state message from the robot
        """
        with self._state_lock:
            if "step" in state:
                self._last_step = max(self._last_step, state["step"])
                if state["step"] < self._last_step:
                    if self._warning_on_out_of_date_state < state["step"]:
                        logger.warning(
                            f"Dropping out-of-date state message: {state['step']} < {self._last_step}"
                        )
                        self._warning_on_out_of_date_state = state["step"]
            # self._state = state
            self._state = StateObservations.from_dict(state)
            self._control_mode = self._state.control_mode
            self._at_goal = self._state.at_goal

            if self._iter <= 0:
                self._iter = max(self._last_step, self._iter)

    def _update_servo(self, message):
        """Servo messages"""
        if self._state is None:
            # self._servo is using with self._state, must ensure self._state is ok
            return

        # Get head information from the message as well
        rgb = compression.from_jpg(message["rgb"])
        depth = compression.from_jp2(message["depth"]) / 1000
        
        with self._servo_lock:
            self._servo = ServoObservations.from_dict(message)
            self._servo.rgb = rgb
            self._servo.depth = depth


    @property
    def running(self) -> bool:
        """Is the client running? Best practice is to check this during while loops.

        Returns:
            bool: whether the client is running
        """
        return not self._finish

    def is_running(self) -> bool:
        """Is the client running? Best practice is to check this during while loops.

        Returns:
            bool: whether the client is running
        """
        return not self._finish

    def say(self, text: str):
        """Send a text message to the robot to say. Will be spoken by the robot's text-to-speech system asynchronously."""
        next_action = {"say": text}
        self.send_action(next_action)

    def say_sync(self, text: str):
        """Send a text message to the robot to say. Will be spoken by the robot's text-to-speech system synchronously."""
        next_action = {"say_sync": text}
        self.send_action(next_action)


    @property
    def is_homed(self) -> bool:
        """Is the robot homed?

        Returns:
            bool: whether the robot is homed
        """
        # This is not really thread safe
        with self._state_lock:
            return self._state is not None and self._state.is_homed

    @property
    def is_runstopped(self) -> bool:
        """Is the robot runstopped?

        Returns:
            bool: whether the robot is runstopped
        """
        with self._state_lock:
            return self._state is not None and self._state.is_runstopped

    def start(self) -> bool:
        """Start running blocking thread in a separate thread. This will wait for observations to come in and update internal state.

        Returns:
            bool: whether the client was started successfully
        """
        if self._started:
            # Already started
            return True

        self._thread = threading.Thread(target=self.blocking_spin)
        self._state_thread = threading.Thread(target=self.blocking_spin_state)
        self._servo_thread = threading.Thread(target=self.blocking_spin_servo)
        if self._rerun:
            self._rerun_thread = threading.Thread(target=self.blocking_spin_rerun)  # type: ignore
        self._finish = False
        self._thread.start()
        self._state_thread.start()
        self._servo_thread.start()
        if self._rerun:
            self._rerun_thread.start()

        t0 = timeit.default_timer()
        # while self._obs is None or self._state is None or self._servo is None:
        # self._obs just update when robot is moving
        while self._state is None or self._servo is None:
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > 10.0:
                logger.error(
                    colored(
                        "Timeout waiting for observations; are you connected to the robot? Check the network.",
                        "red",
                    )
                )
                logger.info(
                    "Try making sure that the server on the robot is publishing, and that you can ping the robot IP address."
                )
                logger.info("Robot IP:", self.send_address)
                return False

        # Separately wait for state messages
        while self._state is None:
            time.sleep(0.1)
            t1 = timeit.default_timer()
            if t1 - t0 > 10.0:
                logger.error(
                    colored(
                        "Timeout waiting for state information; are you connected to the robot? Check the network.",
                        "red",
                    )
                )

        if not self.is_homed:
            self.stop()
            raise RuntimeError(
                "Robot is not homed; please home the robot before running. You can do so by shutting down the server and running ./stretch_robot_home.py on the robot."
            )
        if self.is_runstopped:
            self.stop()
            raise RuntimeError(
                "Robot is runstopped; please release the runstop before running. You can do so by pressing and briefly holding the runstop button on the robot."
            )

        self._started = True
        return True

    def __del__(self):
        """Destructor to make sure we stop the client when it is deleted"""
        self.stop()

    def stop(self):
        """Stop the client and close all sockets"""
        self._finish = True
        if self._thread is not None:
            self._thread.join()
        if self._state_thread is not None:
            self._state_thread.join()
        if self._servo_thread is not None:
            self._servo_thread.join()
        if self._rerun_thread is not None:
            self._rerun_thread.join()

        # Close the sockets and context
        self.recv_socket.close()
        self.recv_state_socket.close()
        self.recv_servo_socket.close()
        self.send_socket.close()
        self.context.term()


@click.command()
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
@click.option("--recv_port", default=4401, help="Port to receive observations on")
@click.option("--send_port", default=4402, help="Port to send actions to on the robot")
@click.option("--robot_ip", default="10.33.140.226")
def main(
    local: bool = True,
    recv_port: int = 4401,
    send_port: int = 4402,
    robot_ip: str = "10.33.140.226",
):
    client = DreamRobotZmqClient(
        robot_ip=robot_ip,
        recv_port=recv_port,
        send_port=send_port,
        use_remote_computer=(not local),
    )
    # client.blocking_spin(verbose=True, visualize=False)


if __name__ == "__main__":
    main()
