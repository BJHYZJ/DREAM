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

from dream.agent.manipulation.dream_manipulation.image_publisher import ImagePublisher
from dream.agent.manipulation.dream_manipulation.place import Placing
from dream.agent.manipulation.dream_manipulation.dream_manipulation import (
    DreamManipulationWrapper as ManipulationWrapper,
)
import dream.motion.constants as constants
from dream.utils.geometry import point_global_to_base

def process_image_for_placing(obj, hello_robot, detection_model, save_dir=None):
    if save_dir is not None:
        save_dir = save_dir + "/" + obj
    placing = Placing(hello_robot.robot, detection_model, save_dir=save_dir)
    retries = [
        [0, 0],
        [0, -0.1],
        [0, 0.1],
        [-0.1, 0],
        [-0.1, -0.1],
        [-0.1, 0.1],
        [0.1, 0],
        [0.1, -0.1],
        [0.1, 0.1],
    ]
    success = False
    head_tilt = hello_robot.tilt
    head_pan = hello_robot.pan
    base_trans = 0

    for i in range(9):
        print("Capturing image: ")
        print(f"retry entries : {retries[i]}")
        delta_base, delta_tilt = retries[i]
        hello_robot.move_to_position(
            base_trans=base_trans + delta_base, head_tilt=head_tilt + delta_tilt, head_pan=head_pan
        )
        actions = placing.process(obj, 1, head_tilt=head_tilt + delta_tilt)
        if actions is not None:
            base_trans, head_tilt = actions
            hello_robot.move_to_position(
                base_trans=base_trans, head_tilt=head_tilt, head_pan=head_pan
            )
            success = True
            break

    if not success:
        print("Did not detect object!")
        return None, None

    base_trans = 0
    head_tilt = hello_robot.tilt
    head_pan = hello_robot.pan

    for i in range(9):
        print("Capturing image: ")
        print(f"retry entries : {retries[i]}")
        delta_base, delta_tilt = retries[i]
        hello_robot.move_to_position(
            base_trans=base_trans + delta_base, head_tilt=head_tilt + delta_tilt, head_pan=head_pan
        )
        translation = placing.process(obj, 2, head_tilt=head_tilt + delta_tilt)
        if translation is not None:
            return [0], np.array([-translation[1], -translation[0], -translation[2]])

    print("Did not detect object!")
    return None, None


def apply_se3_transform(se3_obj, point):
    homogeneous_point = np.append(point.flatten(), 1)
    print(homogeneous_point)
    transformed_homogeneous_point = se3_obj.homogeneous.dot(homogeneous_point)
    transformed_point = transformed_homogeneous_point[:3]

    return transformed_point


def capture_and_process_image(mode, obj, socket, manip_wrapper: ManipulationWrapper):
    """Find an an object in the camera frame and return the translation and rotation of the object.

    Returns:
        rotation: Rotation of the object
        translation: Translation of the object
        depth: Depth of the object (distance from the camera; only for pick mode)
        width: Width of the object (only for pick mode)
    """

    print("Currently in " + mode + " mode and the robot is about to manipulate " + obj + ".")

    image_publisher = ImagePublisher(manip_wrapper.robot, socket)

    # Centering the object
    head_tilt_angles = [0, -5, 5]
    tilt_retries = 1
    side_retries = 0
    retry_flag = True
    theta_cumulative = 0
    # head_tilt = 105
    # head_pan = -1.57

    while retry_flag:

        print("Capturing image: ")
        print(f"retry flag : {retry_flag}")
        print(f"side retries : {side_retries}")
        print(f"tilt retries : {tilt_retries}")

        translation, rotation, depth, width, retry_flag = image_publisher.publish_image(obj, mode)

        if retry_flag == 1:
            # base_trans = translation[0]
            # head_tilt += rotation[0]
            target_in_cam = np.array(translation).astype(np.float32)

            manip_wrapper.robot.look_at_target(target_in_cam, is_in_map=False)
            # manip_wrapper.move_to_position(
            #     base_trans=base_trans, head_pan=head_pan, head_tilt=head_tilt
            # )

        elif retry_flag != 0 and side_retries == 3:
            print("Tried in all angles but couldn't succeed")
            if mode == "place":
                return None, None
            else:
                return None, None, None, None

        elif side_retries == 2 and tilt_retries == 3:
            manip_wrapper.move_to_position(base_theta=np.deg2rad(15))
            manip_wrapper.robot.look_at_target(target_in_cam, is_in_map=False)
            side_retries = 3
            theta_cumulative += 15

        elif retry_flag == 2:
            if tilt_retries == 3:
                if side_retries == 0:
                    manip_wrapper.move_to_position(base_theta=np.deg2rad(15))
                    manip_wrapper.robot.look_at_target(target_in_cam, is_in_map=False)
                    side_retries = 1
                    theta_cumulative += 15
                else:
                    manip_wrapper.move_to_position(base_theta=np.deg2rad(-30))
                    manip_wrapper.robot.look_at_target(target_in_cam, is_in_map=False)
                    side_retries = 2
                    theta_cumulative -= 30
                tilt_retries = 1
            else:
                print(f"retrying with head tilt : {head_tilt_angles[tilt_retries]}")
                manip_wrapper.move_to_position(
                    joint5=head_tilt_angles[tilt_retries]
                )
                tilt_retries += 1

    if mode == "place":
        translation = np.array([-translation[1], -translation[0], -translation[2]])

    if mode == "pick":
        print("Pick: Returning translation, rotation, depth, width")
        return rotation, translation, depth, width, theta_cumulative
    else:
        print("Place: Returning translation, rotation")
        return rotation, translation, theta_cumulative


def move_to_point(robot, point, base_node, gripper_node, move_mode=1, pitch_rotation=0):
    """
    Function for moving the gripper to a specific point
    """
    rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    dest_frame = pin.SE3(rotation, point)
    transform = robot.get_joint_transform(base_node, gripper_node)

    # Rotation from gripper frame frame to gripper frame
    transformed_frame = transform * dest_frame

    transformed_frame.translation[2] -= 0.2

    robot.move_to_pose(
        [
            transformed_frame.translation[0],
            transformed_frame.translation[1],
            transformed_frame.translation[2],
        ],
        [pitch_rotation, 0, 0],
        [1],
        move_mode=move_mode,
    )
    # state = robot.robot.get_six_joints()
    # state[1] += 0.02
    # state[2] += 0.02
    # # state[0] -= 0.012
    # robot.robot.arm_to(state, blocking=True)


def pregrasp_open_loop(self, object_xyz: np.ndarray, distance_from_object: float = 0.35):
    """Move to a pregrasp position in an open loop manner.

    Args:
        object_xyz (np.ndarray): Location to grasp
        distance_from_object (float, optional): Distance from object. Defaults to 0.2.
    """
    self.robot.arm_to(constants.pregrasp, blocking=True)
    xyt = self.robot.get_arm_base_in_map_xyt()
    relative_object_xyz = point_global_to_base(object_xyz, xyt)
    ee_in_arm_base_pose = self.robot.get_ee_in_arm_base()
    ee_rotation = ee_in_arm_base_pose[:3, :3]
    ee_position = ee_in_arm_base_pose[:3, 3]

    vector_to_object = relative_object_xyz - ee_position
    vector_to_object = vector_to_object / np.linalg.norm(vector_to_object)

    print("Absolute object xyz was:", object_xyz)
    print("Relative object xyz was:", relative_object_xyz)
    shifted_object_xyz = relative_object_xyz - (distance_from_object * vector_to_object)
    print("Pregrasp xyz:", shifted_object_xyz)
    pregrasp_pose = np.eye(4)
    pregrasp_pose[:3, :3] = ee_rotation
    pregrasp_pose[:3, 3] = shifted_object_xyz

    success, target_joint_angles, debug_info = self.robot._robot_model.manip_ik(
        target_pose=pregrasp_pose, q_init=constants.pregrasp, is_radians=False)

    print("Pregrasp joint angles: ")
    print(" - joint1: ", target_joint_angles[0])
    print(" - joint2: ", target_joint_angles[1])
    print(" - joint3: ", target_joint_angles[2])
    print(" - joint4: ", target_joint_angles[3])
    print(" - joint5: ", target_joint_angles[4])
    print(" - joint5: ", target_joint_angles[5])

    # get point 10cm from object
    if not success:
        print("Failed to find a valid IK solution.")
        self._success = False
        return

    print(f"{self.name}: Moving to pre-grasp position.")
    self.robot.arm_to(target_joint_angles, blocking=True)
    print("Moving tilt and pan to center object in image, ensure robot can see target object.")
    self.robot.look_at_target(target_point=object_xyz, is_in_map=True, blocking=True)
    print("... done.")


def pickup(
    manip_wrapper,
    rotation,
    translation,
    camera_in_arm_base,
    arm_angles_deg,
    gripper_width=830,
):
    ee_goal_in_camera_pose = np.eye(4)
    ee_goal_in_camera_pose[:3, :3] = rotation
    ee_goal_in_camera_pose[:3, 3] = translation
    ee_goal_in_arm_base = camera_in_arm_base @ ee_goal_in_camera_pose
    # print(f"pin_transformed frame {pin_transformed_frame}")
    manip_wrapper.robot.gripper_to(position=gripper_width)

    success, joints_solution, debug_info = manip_wrapper.robot._robot_model.manip_ik(
        ee_goal_in_arm_base, 
        q_init=arm_angles_deg, 
        is_radians=False, 
        verbose=False
    )
    
    picked = False
    if success:
        print("set gripper to suit position.")
        # move arm to safty place
        manip_wrapper.robot.arm_to(angle=constants.look_front)
        # First, move joint6 to the target position. 
        # This process prevents the gripper from colliding with the object during rotation.
        joint6_change_deg = constants.look_front.copy()
        joint6_change_deg[5] = joints_solution[5]
        manip_wrapper.robot.arm_to(angle=joint6_change_deg)
        manip_wrapper.robot.arm_to(angle=joints_solution)
        picked = manip_wrapper.pickup(gripper_width)
        if picked:
            manip_wrapper.robot.arm_to(angle=constants.look_front)
            # look at back and place to back
            manip_wrapper.robot.arm_to(angle=constants.back_front)
            manip_wrapper.robot.arm_to(angle=constants.back_place)
            manip_wrapper.robot.gripper_to(position=830)
            manip_wrapper.robot.arm_to(angle=constants.back_front)
            manip_wrapper.robot.arm_to(angle=constants.look_front)
    
    if (not success) or (not picked):
        print("AnyGrasp is not work, try to use Two Stage Heuristic grasp founction")

        

    return True