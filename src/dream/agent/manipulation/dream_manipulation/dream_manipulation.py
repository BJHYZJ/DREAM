# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np
import pinocchio as pin
import time
# from urdf_parser_py.urdf import URDF
from scipy.spatial.transform import Rotation as R

from dream.motion.constants import DreamIdx
from dream.motion import constants

OVERRIDE_STATES: dict[str, float] = {}


def transform_joint_array(joint_array):
    n = len(joint_array)
    new_joint_array = []
    for i in range(n + 3):
        if i < 2:
            new_joint_array.append(joint_array[i])
        elif i < 6:
            new_joint_array.append(joint_array[2] / 4.0)
        else:
            new_joint_array.append(joint_array[i - 3])
    return np.array(new_joint_array)


class DreamManipulationWrapper:
    def __init__(
        self,
        robot,
        gripper_threshold=7.0,
        gripper_max=830,
        gripper_min=0,
        end_link="gripper",
    ):
        self.GRIPPER_MAX = gripper_max
        self.GRIPPER_MIN = gripper_min
        self.joints_pin = {"joint_fake": 0}

        self.GRIPPER_THRESHOLD = gripper_threshold

        print("dream robot starting")
        self.base_joint_list = constants.BASE_JOINTS
        self.arm_joint_list = constants.ARM_JOINTS
        self.gripper_joint_list = constants.GRIPPER_JOINTS
        # self.head_joint_list = ["joint_fake", "joint_head_pan", "joint_head_tilt"]
        # self.init_joint_list = [
        #     "joint_fake",
        #     "joint_lift",
        #     "3",
        #     "2",
        #     "1",
        #     "0",
        #     "joint_wrist_yaw",
        #     "joint_wrist_pitch",
        #     "joint_wrist_roll",
        #     "joint_gripper_finger_left",
        # ]

        # end_link is the frame of reference node
        self.end_link = end_link
        self.joint_list = self.base_joint_list + self.arm_joint_list + self.gripper_joint_list

        # Initialize Client controller
        self.robot = robot
        # self.robot.switch_to_manipulation_mode()
        # time.sleep(2)

        # Constraining the robots movement
        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        # self.pan, self.tilt = self.robot.get_pan_tilt()

    def get_joints(self):

        joint_names = self.joint_list
        self.updateJoints()
        joint_values = list(self.joints.values()) + [0] + list(self.head_joints.values())[1:]

        return joint_names, joint_values

    def look_at_target_tilt(
        self, 
        target_point,
        blocking=True,
    ):
        """
        Look at the target point
        """
        self.robot.look_at_target_tilt(target_point, blocking=blocking)


    def look_at_target_pan(
        self, 
        target_point,
        blocking=True,
    ):
        """
        Look at the target point
        """
        self.robot.look_at_target_pan(target_point, blocking=blocking)


    def move_to_position(
        self,
        base_theta=None,  # in radians
        joint1=None,
        joint2=None,
        joint3=None,
        joint4=None,
        joint5=None,
        joint6=None,
        gripper_pos=None,
        blocking=True,
    ):
        """
        Moves the robots, base, arm, gripper, head to a desired position.
        """
        # Base, arm and gripper state update
        if base_theta is not None:
            self.robot.base_to([0, 0, base_theta], relative=True, blocking=blocking)
            return

        if any([joint1, joint2, joint3, joint4, joint5, joint6]):
            arm_joint_state = self.robot.get_arm_joint_state()
            if joint1 is not None:
                arm_joint_state[0] += joint1
            if joint2 is not None:
                arm_joint_state[1] += joint2
            if joint3 is not None:
                arm_joint_state[2] += joint3
            if joint4 is not None:
                arm_joint_state[3] += joint4
            if joint5 is not None:
                arm_joint_state[4] += joint5
            if joint6 is not None:
                arm_joint_state[5] += joint6

            self.robot.arm_to(angle=arm_joint_state)

        if gripper_pos is not None:
            self.robot.gripper_to(gripper_pos, blocking=blocking)


    # def move_to_position(
    #     self,
    #     base_theta=None,
    #     ee_x=None,
    #     ee_y=None,
    #     ee_z=None,
    #     ee_roll=None,
    #     ee_pitch=None,
    #     ee_yaw=None,
    #     gripper_pos=None,
    #     blocking=True,
    # ):
    #     """
    #     Moves the robots, base, arm, gripper, head to a desired position.
    #     """
    #     # Base, arm and gripper state update
    #     target_state = self.robot.extract_joints_positions()

    #     if base_theta is None:
    #         self.robot.navigate_to([0, 0, base_theta], relative=True, blocking=blocking)
    #         return

    #     if not ee_x is None:
    #         target_state[1] = ee_x
    #     if not ee_y is None:
    #         target_state[2] = ee_y
    #     if not ee_z is None:
    #         target_state[3] = ee_z
    #     if not ee_roll is None:
    #         target_state[4] = ee_roll
    #     if not ee_pitch is None:
    #         target_state[5] = ee_pitch
    #     if not ee_yaw is None:
    #         target_state[6] = ee_yaw
        
    #     if gripper_pos is not None:
    #         target_state[7] = gripper_pos

    #     # if gripper_pos is not None:
    #     #     self.CURRENT_STATE = gripper_pos
    #     #     self.robot.gripper_to(self.CURRENT_STATE, blocking=blocking)

    #     self.robot.arm_to(
    #         joint_angles=target_state[:6], 
    #         gripper=target_state[6], 
    #         blocking=blocking, 
    #         reliable=False
    #     )


    def pickup(self, width):
        """
        Code for grasping the object
        Gripper closes gradually until it encounters resistance
        """
        next_gripper_pos = width
        while True:
            self.robot.gripper_to(
                max(next_gripper_pos, self.GRIPPER_MIN), blocking=True
            )
            curr_gripper_pose = self.robot.get_gripper_position()
            print('Robot means to move gripper to', next_gripper_pos)
            print('Robot actually moves gripper to', curr_gripper_pose, 'curr_gripper_pose - next_gripper_pos =', curr_gripper_pose - next_gripper_pos)
            if next_gripper_pos <= 0:
                return False
            
            if curr_gripper_pose - next_gripper_pos > 10:
                print(f"Gripper stopped closing at position: {curr_gripper_pose}")
                return True  # Stop closing if fully closed or resistance is detected  

            if next_gripper_pos > 0:
                next_gripper_pos -= 100
            else:
                next_gripper_pos = 0 # Make sure the gripper doesn't go below 0

            time.sleep(0.1)

    def place_back(self):
        self.robot.look_front()
        self.robot.arm_to(angle=constants.back_front)
        self.robot.arm_to(angle=constants.back_place)
        self.robot.gripper_to(position=self.GRIPPER_MAX)
        self.robot.arm_to(angle=constants.back_front)
        self.robot.look_front()

    def updateJoints(self):
        """
        update all the current positions of joints
        """
        state = self.robot.extract_joints_positions()
        origin_dist = state[0]

        # Head Joints
        pan, tilt = self.robot.get_pan_tilt()

        self.joints_pin["joint_fake"] = origin_dist
        self.joints_pin["joint_lift"] = state[1]
        armPos = state[2]
        self.joints_pin["joint_arm_l3"] = armPos / 4.0
        self.joints_pin["joint_arm_l2"] = armPos / 4.0
        self.joints_pin["joint_arm_l1"] = armPos / 4.0
        self.joints_pin["joint_arm_l0"] = armPos / 4.0
        self.joints_pin["joint_wrist_yaw"] = state[3]
        self.joints_pin["joint_wrist_roll"] = state[5]
        self.joints_pin["joint_wrist_pitch"] = OVERRIDE_STATES.get("wrist_pitch", state[4])
        self.joints_pin["joint_gripper_finger_left"] = 0

        self.joints_pin["joint_head_pan"] = pan
        self.joints_pin["joint_head_tilt"] = tilt

    # following function is used to move the robot to a desired joint configuration
    def move_to_joints(self, joints, gripper, mode=0):
        """
        Given the desired joints movement this function will the joints accordingly
        """
        state = self.robot.extract_joints_positions()

        # clamp rotational joints between -1.57 to 1.57
        joints["joint_wrist_pitch"] = (joints["joint_wrist_pitch"] + 1.57) % 3.14 - 1.57
        joints["joint_wrist_yaw"] = (joints["joint_wrist_yaw"] + 1.57) % 3.14 - 1.57
        joints["joint_wrist_roll"] = (joints["joint_wrist_roll"] + 1.57) % 3.14 - 1.57
        joints["joint_wrist_pitch"] = self.clamp(joints["joint_wrist_pitch"], -1.57, 0.1)
        target_state = [
            joints["joint_fake"],
            joints["joint_lift"],
            joints["3"] + joints["2"] + joints["1"] + joints["0"],
            joints["joint_wrist_yaw"],
            joints["joint_wrist_pitch"],
            joints["joint_wrist_roll"],
        ]

        # Moving only the lift first
        if mode == 1:
            target1 = state
            target1[0] = target_state[0]
            target1[1] = min(1.1, target_state[1] + 0.2)
            self.robot.arm_to(
                target1, blocking=True, head=np.array([self.pan, self.tilt]), reliable=False
            )

        self.robot.arm_to(target_state, blocking=True)
        # self.robot.arm_to(head_tilt=self.tilt, head_pan=self.pan, blocking=True, reliable=False)

        # self.robot.arm_to(
        #     target_state, blocking=True, head=np.array([self.pan, self.tilt]), reliable=True
        # )

        # NOTE: below code is to fix the pitch drift issue in current hello-robot. Remove it if there is no pitch drift issue
        OVERRIDE_STATES["wrist_pitch"] = joints["joint_wrist_pitch"]

    def get_joint_transform(self, node1, node2):
        """
        This function takes two nodes from a robot URDF file as input and
        outputs the coordinate frame of node2 relative to the coordinate frame of node1.

        Mainly used for transforming coordinates from camera frame to gripper frame.
        """

        # return frame_transform, frame2, frame1
        self.updateJoints()
        frame_pin = self.robot.get_frame_pose(self.joints_pin, node1, node2)

        return frame_pin

    def move_to_pose(self, translation_tensor, rotational_tensor, gripper, move_mode=0):
        """
        Function to move the gripper to a desired translation and rotation
        """
        translation = [translation_tensor[0], translation_tensor[1], translation_tensor[2]]
        rotation = rotational_tensor

        self.updateJoints()

        q = self.robot.get_joint_position()
        q[DreamIdx.GRIPPER] = OVERRIDE_STATES.get(
            "gripper", q[DreamIdx.GRIPPER]
        )
        pin_pose = self.robot.get_ee_pose(matrix=True, link_name=self.end_link, q=q)
        pin_rotation, pin_translation = pin_pose[:3, :3], pin_pose[:3, 3]
        pin_curr_pose = pin.SE3(pin_rotation, pin_translation)

        rot_matrix = R.from_euler("xyz", rotation, degrees=False).as_matrix()

        pin_del_pose = pin.SE3(np.array(rot_matrix), np.array(translation))
        pin_goal_pose_new = pin_curr_pose * pin_del_pose

        final_pos = pin_goal_pose_new.translation.tolist()
        final_quat = pin.Quaternion(pin_goal_pose_new.rotation).coeffs().tolist()
        # print(f"final pos and quat {final_pos}\n {final_quat}")

        full_body_cfg = self.robot.solve_ik(
            final_pos, final_quat, None, False, custom_ee_frame=self.end_link
        )
        if full_body_cfg is None:
            print("Warning: Cannot find an IK solution for desired EE pose!")
            return False

        pin_joint_pos = self.robot._extract_joint_pos(full_body_cfg)
        transform_joint_pos = transform_joint_array(pin_joint_pos)

        self.joint_array1 = transform_joint_pos

        ik_joints = {}
        for joint_index in range(len(self.joint_array1)):
            ik_joints[self.joint_list[joint_index]] = self.joint_array1[joint_index]

        # Actual Movement of joints
        self.move_to_joints(ik_joints, gripper, move_mode)
