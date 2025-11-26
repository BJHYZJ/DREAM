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
from re import S
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
from dream.motion import constants
import numpy as np
import time
import traceback

from dream_ros2_bridge.remote.ros import DreamRosInterface

# from std_srvs.srv import TriggerRequest


GRIPPER_MOTION_SECS = 2.2
JOINT_POS_TOL = 0.015
JOINT_ANG_TOL = 0.05



class XARM6:
    def __init__(
        self,
        interface="192.168.1.233",
        # The pose corresponds to the servo angle
        init_servo_angle=constants.look_front,
        back_front_angle=constants.back_front,
    ):
        self.pprint("xArm-Python-SDK Version:{}".format(version.__version__))
        self.alive = True
        self._xarm = XArmAPI(interface, baud_checkset=False)
        self.init_servo_angle = init_servo_angle
        self.back_front_angle = back_front_angle
        self._robot_init()

    # Robot Init
    def _robot_init(self):
        self._xarm.clean_warn()
        self._xarm.clean_error()
        self._xarm.motion_enable(True)
        self._xarm.set_mode(0)
        self._xarm.set_state(0)
        self._xarm.set_gripper_enable(True)
        self._xarm.set_gripper_mode(0)
        self._xarm.clean_gripper_error()
        self._xarm.set_collision_sensitivity(1)
        time.sleep(1)
        self._xarm.register_error_warn_changed_callback(
            self._error_warn_changed_callback
        )
        self._xarm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._xarm, "register_count_changed_callback"):
            self._xarm.register_count_changed_callback(self._count_changed_callback)
        self.reset()
        self.open_gripper()

    # Robot Contrl: here the pose is the end-effector pose [X, Y, Z, roll, pitch, yaw]
    def move_to_pose(self, pose, wait=True, ignore_error=False):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._xarm.set_position(
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
        code, pose = self._xarm.get_position()
        if not self._check_code(code, "get_position"):
            # raise ValueError("get_current_pose Error")
            return None
        return pose

    def get_joint_state(self):
        if not self.is_alive:
            # raise ValueError("Robot is not alive!")
            return None, None, None
        code, state = self._xarm.get_joint_states()
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

    def set_servo_angle(self, angle, speed=20, is_radian=False, wait=True):
        self._xarm.set_servo_angle(angle=angle, speed=speed, is_radian=is_radian, wait=wait)

    def _sync_servo_command_with_real_state(self):
        """Align controller command with actual encoder state to avoid TF jumps."""
        real_angles = self.get_servo_angle(is_radian=False, is_real=True)
        if real_angles is None:
            return
        self.set_servo_angle(angle=real_angles, speed=20, is_radian=False, wait=False)

    def get_servo_angle(self, is_radian=False, is_real=False):
        if not self.is_alive:
            # raise ValueError("Robot is not alive!")
            return None
        code, angle = self._xarm.get_servo_angle(is_radian=is_radian, is_real=is_real)
        if not self._check_code(code, "get_servo_angle"):
            # raise ValueError("get_servo_angle Error")
            return None
        return angle


    def set_gripper(self, target: int = 830, wait: bool = True):
        if not self.is_alive:
            raise ValueError("Robot is not alive!")
        code = self._xarm.set_gripper_position(target, wait=wait)
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
        code, state = self._xarm.get_gripper_position()
        if not self._check_code(code, "get_gripper_position"):
            # raise ValueError("get_gripper_position Error")
            return None
        return state

    def reset(self):
        # This can proimise the initial position has the correct joint angle
        servo_pose = self.get_servo_angle()
        if servo_pose and (servo_pose[0] > 135 or servo_pose[0] < -135):
            back_front = self.back_front_angle.copy()
            if servo_pose[0] < 0:
                back_front[0] = -back_front[0]
            self._xarm.set_servo_angle(
                angle=back_front, speed=20, is_radian=False, wait=True
            )
        time.sleep(0.1)
        self._xarm.set_servo_angle(
            angle=self.init_servo_angle, speed=20, is_radian=False, wait=True
        )

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data["error_code"] != 0:
            self.alive = False
            self.pprint("err={}, quit".format(data["error_code"]))
            self._xarm.release_error_warn_changed_callback(
                self._error_warn_changed_callback
            )

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data["state"] == 4:
            self.alive = False
            self.pprint("state=4, quit")
            self._xarm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint("counter val: {}".format(data["count"]))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._xarm.get_state()
            ret2 = self._xarm.get_err_warn_code()
            self.pprint(
                "{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}".format(
                    label,
                    code,
                    self._xarm.connected,
                    self._xarm.state,
                    self._xarm.error_code,
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
        if self._xarm.connected and self._xarm.error_code == 0:
            # print(self._arm.connected, self._arm.error_code, self._arm.state)
            if self._xarm.state == 5:
                self._xarm.set_state(0)
                print("set state to 0 when state is 5")
                # cnt = 0
                # while self._arm.state == 5 and cnt < 5:
                #     cnt += 1
                #     time.sleep(0.1)
            return self._xarm.state < 4
        else:
            return False



class DreamManipulationClient(AbstractControlModule):
    """Manages dream arm control and "manipulation mode" base motions (forward and backward)."""

    def __init__(self, ros_client: DreamRosInterface, robot_model: RobotModel):
        super().__init__()

        self._ros_client = ros_client
        self._robot_model = robot_model

        self._init_base_pose = None
        with self._ros_client.rtabmap_paused():
            self._arm = XARM6()
            time.sleep(1)


    # Enable / disable

    def _enable_hook(self) -> bool:
        """Called when interface is enabled."""
        print("Switching to manipulation mode")
        # self._init_base_pose = self._ros_client.get_base_in_map_pose()

        # return result.success
        return True

    def _disable_hook(self) -> bool:
        """Called when interface is disabled. This will set the manip base pose back to none."""
        self._init_base_pose = None
        # We do not need to call the service to disable the mode
        return True

    # def get_base_x(self):
    #     """Get the current base x position"""
    #     if self._init_base_pose is None:
    #         return 0.0
    #     current_global_pose = self._ros_client.get_base_in_map_pose()
    #     relative_xyt = pose_global_to_base_xyt(current_global_pose, self._init_base_pose)
    #     return relative_xyt[0]

    def _disable_hook(self) -> bool:
        """Called when interface is disabled."""
        return True

    # Interface methods
    @enforce_enabled
    def get_servo_angle(self, is_radian=False, is_real=False):
        return self._arm.get_servo_angle(is_radian=is_radian, is_real=is_real)

    @enforce_enabled
    def set_servo_angle(
        self,
        angle: np.ndarray,
        speed: int,
        is_radian: bool = False,
        wait: bool = True,
    ):
        with self._ros_client.rtabmap_paused():  # paused when arm moving, it can ensure better image quality
            self._arm.set_servo_angle(angle, speed=speed, is_radian=is_radian, wait=wait)

    @enforce_enabled
    def get_gripper_position(self) -> float:
        """get current gripper position as a float"""
        gripper_state = self._arm.get_gripper_state()
        return gripper_state

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
