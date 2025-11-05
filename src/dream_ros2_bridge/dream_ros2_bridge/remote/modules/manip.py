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
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger

import dream.motion.conversions as conversions
from dream.motion.constants import DreamIdx
from dream.motion.robot import RobotModel
from dream.utils.geometry import pose_global_to_base_xyt, posquat2sophus, sophus2posquat
from .abstract import AbstractControlModule, enforce_enabled

from xarm import version
from xarm.wrapper import XArmAPI
from dream.motion.constants import look_front, back_front
import numpy as np
import time
import traceback


# from std_srvs.srv import TriggerRequest


GRIPPER_MOTION_SECS = 2.2
JOINT_POS_TOL = 0.015
JOINT_ANG_TOL = 0.05



class XARM6:
    def __init__(
        self,
        interface="192.168.1.233",
        # The pose corresponds to the servo angle
        init_servo_angle=look_front,
    ):
        self.pprint("xArm-Python-SDK Version:{}".format(version.__version__))
        self.alive = True
        self._arm = XArmAPI(interface, baud_checkset=False)
        self.init_servo_angle = init_servo_angle
        self._robot_init()

    # Robot Init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        self._arm.set_gripper_enable(True)
        self._arm.set_gripper_mode(0)
        self._arm.clean_gripper_error()
        self._arm.set_collision_sensitivity(1)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(
            self._error_warn_changed_callback
        )
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, "register_count_changed_callback"):
            self._arm.register_count_changed_callback(self._count_changed_callback)
        self.reset()
        self.open_gripper()

    # Robot Contrl: here the pose is the end-effector pose [X, Y, Z, roll, pitch, yaw]
    def move_to_pose(self, pose, wait=True, ignore_error=False):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_position(
            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], wait=wait
        )
        if not ignore_error:
            if not self._check_code(code, "set_position"):
                raise ValueError("move_to_pose Error")
        return True

    def get_current_pose(self):
        if not self.is_alive:
            # raise ValueError("Robot is not alive!")
            return None
        code, pose = self._arm.get_position()
        if not self._check_code(code, "get_position"):
            # raise ValueError("get_current_pose Error")
            return None
        return pose

    def get_joint_state(self):
        if not self.is_alive:
            # raise ValueError("Robot is not alive!")
            return None, None, None
        code, state = self._arm.get_joint_states()
        if not self._check_code(code, "get_joint_state"):
            # raise ValueError("get_joint_state Error")
            return None, None, None
        # Return only the first 6 joints (excluding gripper)
        # state is typically [positions, velocities, torques] with 7 elements each
        return [
            state[0][:6],  # positions of first 6 joints
            state[1][:6],  # velocities of first 6 joints  
            state[2][:6]   # torques of first 6 joints
        ]


    def set_gripper(self, target: int = 830, wait: bool = True):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._arm.set_gripper_position(target, wait=wait)
        if not self._check_code(code, "set_gripper_position"):
            raise ValueError("set_gripper Error")
        return True

    def open_gripper(self, wait=True, half_open=False):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        if half_open:
            self.set_gripper(460, wait=wait)
        else:
            self.set_gripper(830, wait=wait)

    def close_gripper(self, wait=True):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        self.set_gripper(0, wait=wait)

    def get_gripper_state(self):
        if not self.is_alive:
            # raise ValueError("Robot is not alive!")
            return None
        code, state = self._arm.get_gripper_position()
        if not self._check_code(code, "get_gripper_position"):
            # raise ValueError("get_gripper_position Error")
            return None
        return state

    def reset(self):
        # This can proimise the initial position has the correct joint angle
        servo_pose = self.get_servo_angle()
        if servo_pose[0] > 135 or servo_pose[0] < -135:
            self._arm.set_servo_angle(
                angle=back_front, is_radian=False, wait=True
            )
        time.sleep(0.1)
        self._arm.set_servo_angle(
            angle=self.init_servo_angle, is_radian=False, wait=True
        )

    def set_servo_angle(self, angle, is_radian=False, wait=True):
        self._arm.set_servo_angle(angle=angle, is_radian=is_radian, wait=wait)

    def get_servo_angle(self, is_radian=False, is_real=False):
        if not self.is_alive:
            # raise ValueError("Robot is not alive!")
            return None
        code, angle = self._arm.get_servo_angle(is_radian=is_radian, is_real=is_real)
        if not self._check_code(code, "get_servo_angle"):
            # raise ValueError("get_servo_angle Error")
            return None
        return angle

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data["error_code"] != 0:
            self.alive = False
            self.pprint("err={}, quit".format(data["error_code"]))
            self._arm.release_error_warn_changed_callback(
                self._error_warn_changed_callback
            )

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data["state"] == 4:
            self.alive = False
            self.pprint("state=4, quit")
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint("counter val: {}".format(data["count"]))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint(
                "{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}".format(
                    label,
                    code,
                    self._arm.connected,
                    self._arm.state,
                    self._arm.error_code,
                    ret1,
                    ret2,
                )
            )
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print(
                "[{}][{}] {}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    stack_tuple[1],
                    " ".join(map(str, args)),
                )
            )
        except:
            print(*args, **kwargs)

    @property
    def is_alive(self):
        # if self.alive and self._arm.connected and self._arm.error_code == 0:
        if self._arm.connected and self._arm.error_code == 0:
            # print(self._arm.connected, self._arm.error_code, self._arm.state)
            if self._arm.state == 5:
                self._arm.set_state(0)
                print("set state to 0 when state is 5")
                # cnt = 0
                # while self._arm.state == 5 and cnt < 5:
                #     cnt += 1
                #     time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False



class DreamManipulationClient(AbstractControlModule):
    """Manages dream arm control and "manipulation mode" base motions (forward and backward)."""

    def __init__(self, ros_client, robot_model: RobotModel):
        super().__init__()

        self._ros_client = ros_client
        self._robot_model = robot_model

        self._init_base_pose = None
        self._arm = XARM6()

    # Enable / disable

    def _enable_hook(self) -> bool:
        """Called when interface is enabled."""
        # # Switch interface mode & print messages
        # result = self._ros_client.pos_mode_service.call(Trigger.Request())
        # self._ros_client.get_logger().info("Switching to manipulation mode")
        print("Switching to manipulation mode")
        self._init_base_pose = self._ros_client.get_base_in_map_pose()

        # return result.success
        return True

    def _disable_hook(self) -> bool:
        """Called when interface is disabled. This will set the manip base pose back to none."""
        self._init_base_pose = None
        # We do not need to call the service to disable the mode
        return True

    def get_base_x(self):
        """Get the current base x position"""
        if self._init_base_pose is None:
            return 0.0
        current_global_pose = self._ros_client.get_base_in_map_pose()
        relative_xyt = pose_global_to_base_xyt(current_global_pose, self._init_base_pose)
        return relative_xyt[0]

    def _disable_hook(self) -> bool:
        """Called when interface is disabled."""
        return True

    # Interface methods

    @enforce_enabled
    def get_ee_pose(self, world_frame=False, matrix=False):
        """Get current end-effector pose from xarm controller"""
        # Get current pose from xarm controller [x, y, z, roll, pitch, yaw]
        pose_data = self._arm.get_current_pose()
        pos = np.array([pose_data[0], pose_data[1], pose_data[2]])  # mm
        # Convert euler angles to quaternion
        # xarm returns [x, y, z, roll, pitch, yaw] where roll, pitch, yaw are in degrees
        r = R.from_euler('ZYX', [pose_data[5], pose_data[4], pose_data[3]], degrees=True)
        quat = r.as_quat()  # [x, y, z, w]
        
        # Adjust for base position
        pos[0] += self.get_base_x()

        if world_frame:
            pose_base2ee = posquat2sophus(pos, quat)
            pose_world2base = self._ros_client.get_base_in_map_pose()
            pose_world2ee = pose_world2base * pose_base2ee
            pos, quat = sophus2posquat(pose_world2ee)

        if matrix:
            pose = posquat2sophus(pos, quat)
            return pose.matrix()
        else:
            return pos, quat

    # def get_ee_pose(self, world_frame=False, matrix=False):
    #     q, _, _ = self._ros_client.get_joint_state()
    #     pos_base, quat_base = self._robot_model.manip_fk(q)
    #     pos_base[0] += self.get_base_x()

    #     if world_frame:
    #         pose_base2ee = posquat2sophus(pos_base, quat_base)
    #         pose_world2base = self._ros_client.se3_base_filtered
    #         pose_world2ee = pose_world2base * pose_base2ee

    #         pos, quat = sophus2posquat(pose_world2ee)

    #     else:
    #         pos, quat = pos_base, quat_base
    #     if matrix:
    #         pose = posquat2sophus(pos, quat)
    #         return pose.matrix()
    #     else:
    #         return pos, quat

    # def get_joint_positions(self):
    #     """Get current joint positions including base x position"""
    #     # Get joint states from xarm controller
    #     joint_state = self._arm.get_joint_state()
    #     gripper_state = self._arm.get_gripper_state()
    #     base_x = self.get_base_x()
        
    #     return [
    #         base_x,
    #         joint_state[0],  # joint1
    #         joint_state[1],  # joint2  
    #         joint_state[2],  # joint3
    #         joint_state[3],  # joint4
    #         joint_state[4],  # joint5
    #         joint_state[5],  # joint6
    #         gripper_state,   # gripper
    #     ]

    # def get_joint_positions(self):
    #     q, _, _ = self._ros_client.get_joint_state()
    #     base_x = self.get_base_x()
    #     return [
    #         base_x,
    #         q[DreamIdx.JOINT1],
    #         q[DreamIdx.JOINT2],
    #         q[DreamIdx.JOINT3],
    #         q[DreamIdx.JOINT4],
    #         q[DreamIdx.JOINT5],
    #         q[DreamIdx.JOINT6],
    #         q[DreamIdx.GRIPPER],
    #     ]
    
    @enforce_enabled
    def get_gripper_position(self) -> float:
        """get current gripper position as a float"""
        gripper_state = self._arm.get_gripper_state()
        return gripper_state

    # def get_gripper_position(self) -> float:
    #     """get current gripper position as a float"""
    #     q, _, _ = self._ros_client.get_joint_state()
    #     return q[DreamIdx.GRIPPER]

    # @enforce_enabled
    # def goto(self, q, wait=True):
    #     """Directly command the robot using generalized coordinates
    #     For xarm, we'll convert to end-effector pose and move directly
    #     """
    #     # For xarm, we expect q to be [base_x, joint1, joint2, joint3, joint4, joint5, joint6, gripper]
    #     # We'll use the joint positions to move the arm
    #     if len(q) >= 6:
    #         joint_positions = q[1:7]  # Skip base_x, get joint positions
    #         return self._arm.move_to_pose(joint_positions, wait=wait)
        
    #     return True


    @enforce_enabled
    def move_to_positions(self, positions: List[np.ndarray], wait: bool = True):
        """Move robot to specified positions"""
        self._arm.move_to_pose(positions, wait=wait)
        return True

    @enforce_enabled
    def get_servo_angle(self, is_radian=False, is_real=False):
        return self._arm.get_servo_angle(is_radian=is_radian, is_real=is_real)

    @enforce_enabled
    def set_servo_angle(
        self,
        angle: np.ndarray,
        is_radian: bool = False,
        wait: bool = True,
    ):
        self._arm.set_servo_angle(angle, is_radian=is_radian, wait=wait)

    @enforce_enabled
    def set_gripper(self, target: int = 830, wait: bool = True):
        self._arm.set_gripper(target, wait=wait)

    @enforce_enabled
    def home(self):
        """Move robot to home position"""
        self._arm.reset()

    @enforce_enabled
    def reset(self):
        """Move robot to reset position"""
        self._arm.reset()

    # @enforce_enabled
    # def home(self):
    #     self.goto(STRETCH_HOME_Q, wait=True)

    @enforce_enabled
    def goto_joint_positions(
        self,
        joint_positions: List[float],
        relative: bool = False,
        blocking: bool = True,
        debug: bool = False,
        gripper: float = None,
    ):
        """
        Move robot to specified joint positions
        
        Args:
            joint_positions: List of length 6 containing desired joint positions [base_x, joint1, joint2, joint3, joint4, joint5, joint6]
            relative: Whether the joint positions are relative to current position
            blocking: Whether command blocks until completion
            debug: Whether to print debug information
            move_base: Whether to move the base (not implemented for xarm)
            velocities: Velocity parameter (not used for xarm)
            head_tilt: Head tilt angle (not used for xarm)
            head_pan: Head pan angle (not used for xarm)
            gripper: Gripper position (handled separately)
        """
        assert len(joint_positions) >= 6, "Joint position vector must be of length 6 or more."
        joint_positions = [float(x) for x in joint_positions]

        # Get current joint states for relative movement
        if relative:
            joint_pos_init = self.get_joint_positions()
            joint_pos_goal = np.array(joint_positions) + np.array(joint_pos_init)
        else:
            joint_pos_goal = np.array(joint_positions)

        # Convert to pose format for xarm [x, y, z, roll, pitch, yaw]
        # For xarm, we'll use the joint positions directly as pose coordinates
        pose = [
            joint_pos_goal[1],  # joint1 -> x
            joint_pos_goal[2],  # joint2 -> y  
            joint_pos_goal[3],  # joint3 -> z
            joint_pos_goal[4],  # joint4 -> roll (degrees)
            joint_pos_goal[5],  # joint5 -> pitch (degrees)
            joint_pos_goal[6],  # joint6 -> yaw (degrees)
        ]

        # Move to pose
        success = self._arm.move_to_pose(pose, wait=blocking)

        # Handle gripper separately if specified
        if gripper is not None:
            if gripper > 0.5:  # Threshold for open/close
                self._arm.open_gripper(wait=blocking)
            else:
                self._arm.close_gripper(wait=blocking)

        # Debug print
        if debug:
            print("-- joint goto cmd --")
            if relative:
                print("Initial joint pos: [", *(f"{x:.3f}" for x in joint_pos_init), "]")
            print("Desired joint pos: [", *(f"{x:.3f}" for x in joint_pos_goal), "]")
            print("Pose command: [", *(f"{x:.3f}" for x in pose), "]")
            print("--------------------")

        return success


    @enforce_enabled
    def goto_ee_pose(
        self,
        pos: List[float],
        quat: Optional[List[float]] = None,
        blocking: bool = True,
        debug: bool = False,

    ) -> bool:
        """Command gripper to pose using xarm controller
        
        Args:
            pos: Desired position
            quat: Desired orientation in quaternion (xyzw)
            blocking: Whether command blocks until completion
            debug: Whether to print debug information
            initial_cfg: Preferred (initial) joint state configuration
        """
        # For xarm, we can directly use the pose without complex IK
        # Convert the desired pose to xarm format
        pose = [
            pos[0],  # x position
            pos[1],  # y position
            pos[2],  # z position
        ]
        
        # Add orientation if provided
        if quat is not None:
            r = R.from_quat(quat)
            euler_angles = r.as_euler('ZYX', degrees=True)  # Use ZYX order
            pose.extend([euler_angles[2], euler_angles[1], euler_angles[0]])  # [roll, pitch, yaw]
        else:
            # Use current orientation if not specified
            current_pose = self._arm.get_current_pose()
            pose.extend(current_pose[3:6])
        
        # Move to pose
        success = self._arm.move_to_pose(pose, wait=blocking)

        # Debug print
        if debug and blocking:
            achieved_pos, achieved_quat = self.get_ee_pose()
            print(f"Achieved EE pose: pos={achieved_pos}; quat={achieved_quat}")
            print("=======================")

        return success


    @enforce_enabled
    def rotate_ee(self, axis: int, angle: float, **kwargs) -> bool:
        """Rotates the gripper by one of 3 principal axes (X, Y, Z)"""
        assert axis in [0, 1, 2], "'axis' must be 0, 1, or 2! (x, y, z)"

        r = np.zeros(3)
        r[axis] = angle
        quat_desired = R.from_rotvec(r).as_quat().tolist()

        return self.goto_ee_pose([0, 0, 0], quat_desired, relative=True, **kwargs)
