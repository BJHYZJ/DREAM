import numpy as np
import time
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

        # end_link is the frame of reference node
        self.end_link = end_link
        self.joint_list = self.base_joint_list + self.arm_joint_list + self.gripper_joint_list

        # Initialize Client controller
        self.robot = robot


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


    def pickup(self, width: int=830, multi_stage: bool=False) -> bool:
        """
        Code for grasping the object
        Gripper closes gradually until it encounters resistance
        """
        next_gripper_pos = width
        if multi_stage:
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
        else:
            self.robot.gripper_to(
                self.GRIPPER_MIN, blocking=True
            )
            return True