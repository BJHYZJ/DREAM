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
            side_retries = 3

        elif retry_flag == 2:
            if tilt_retries == 3:
                if side_retries == 0:
                    manip_wrapper.move_to_position(base_theta=np.deg2rad(15))
                    side_retries = 1
                else:
                    manip_wrapper.move_to_position(base_theta=np.deg2rad(-30))
                    side_retries = 2
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
        return rotation, translation, depth, width
    else:
        print("Place: Returning translation, rotation")
        return rotation, translation


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


def pickup(
    manip_wrapper,
    rotation,
    translation,
    camera_in_arm_base,
    arm_angles_deg,
    gripper_height=0.03,
    gripper_depth=0.03,
    gripper_width=830,
):

    rotation_top_mat = np.array([
        [ 0.0,  0.0,  1.0],
        [ 0.0, -1.0,  0.0],
        [ 1.0,  0.0,  0.0],
    ], np.float32)

    ee_goal_in_camera_pose = np.eye(4)
    ee_goal_in_camera_pose[:3, :3] = rotation @ rotation_top_mat
    ee_goal_in_camera_pose[:3, 3] = translation
    ee_goal_in_arm_base = camera_in_arm_base @ ee_goal_in_camera_pose
    # print(f"pin_transformed frame {pin_transformed_frame}")


    success, joints_solution, debug_info = manip_wrapper.robot._robot_model.compute_arm_ik(
        ee_goal_in_arm_base, 
        q_init=arm_angles_deg, 
        is_radians=False, 
        verbose=False
    )

    if success:
        print("set gripper to suit position.")
        manip_wrapper.robot.gripper_to(position=gripper_width)
        manip_wrapper.robot.arm_to(angle=joints_solution)
    else:
        assert 1 == 1
