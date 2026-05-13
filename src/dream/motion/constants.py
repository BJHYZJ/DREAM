import math
import numpy as np
from dream.utils.config import get_full_config_path

look_front = [0, -45, -90, 0, 105, 0]
look_ahead = [0, -45, -90, 0, 75, 0]
look_down = [0, -45, -90, 0, 135, 0]
look_left_1 = [45, -45, -90, 0, 135, 0]
look_left_2 = [45, -45, -90, 0, 105, 0]
look_right_1 = [-45, -45, -90, 0, 135, 0]
look_right_2 = [-45, -45, -90, 0, 105, 0]

back_front = [180, -45, -90, 0, 105, 0]
back_place = [180, -5, -110, 0, 115, 0]
back_look = [180, -30, -130, 0, 155, 0]
back_down = [180, -20, -100, 0, 120, 0]

SAFE_CAMERA_TILT_MIN_DEG = 75
SAFE_CAMERA_TILT_MAX_DEG = 135



T_LOC_STABILIZE = 0.1
BASE_JOINTS = ['base_x', 'base_y', 'base_theta']
ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
GRIPPER_JOINTS = ['gripper']
POSITION_JOINTS = ['ee_x', 'ee_y', 'ee_z', 'ee_roll', 'ee_pitch', 'ee_yaw']
ROBOT_JOINTS = BASE_JOINTS + ARM_JOINTS + GRIPPER_JOINTS

BASE_INDEX = [ROBOT_JOINTS.index(joint) for joint in BASE_JOINTS]
ARM_INDEX = [ROBOT_JOINTS.index(joint) for joint in ARM_JOINTS]
GRIPPER_INDEX = [ROBOT_JOINTS.index(joint) for joint in GRIPPER_JOINTS]


CAMERA_POSET_TOPIC = "/robot/camera_pose"

class DreamIdx:
    BASE_X = 0
    BASE_Y = 1
    BASE_THETA = 2
    JOINT1 = 3
    JOINT2 = 4
    JOINT3 = 5
    JOINT4 = 6
    JOINT5 = 7
    JOINT6 = 8
    GRIPPER = 9

    name_to_idx = {
        "base_x": BASE_X,
        "base_y": BASE_Y,
        "base_theta": BASE_THETA,
        "joint1": JOINT1,
        "joint2": JOINT2,
        "joint3": JOINT3,
        "joint4": JOINT4,
        "joint5": JOINT5,
        "joint6": JOINT6,
        "gripper": GRIPPER,
    }

    @classmethod
    def get_idx(cls, name: str) -> int:
        if name in cls.name_to_idx:
            return cls.name_to_idx[name]
        else:
            raise ValueError(f"Unknown joint name: {name}")
