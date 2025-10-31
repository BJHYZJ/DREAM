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
    head_tilt_angles = [0, -10, 10]
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
    gripper_height=0.03,
    gripper_depth=0.03,
    gripper_width=1,
):
    """
    rotation: Relative rotation of gripper pose w.r.t camera
    translation: Relative translation of gripper pose w.r.t camera
    base_node: Camera Node

    Supports home robot top down grasping as well

    Graping trajectory steps
    1. Rotation of gripper
    2. Lift the gripper
    3. Move the base such gripper in line with the grasp
    4. Gradually Move the gripper to the desired position
    """
    # Transforming the final point from Model camera frame to robot camera frame
    ee_goal_in_camera_translation = translation.copy()

    # Rotation from Camera frame to Model frame
    rotation_bottom_mat = np.array(
        [
            [0.0000000, -1.0000000, 0.0000000],
            [-1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, -1.0000000],
        ]
    )

    rotation_top_mat = np.array([
            [ 0.0000000,  0.0000000,  1.0000000],
            [ 0.0000000, -1.0000000,  0.0000000],
            [ 1.0000000,  0.0000000,  0.0000000],
        ])

    # # Rotation from model frame to pose frame
    # rotation1_mat = np.array(
    #     [
    #         [rotation[0][0], rotation[0][1], rotation[0][2]],
    #         [rotation[1][0], rotation[1][1], rotation[1][2]],
    #         [rotation[2][0], rotation[2][1], rotation[2][2]],
    #     ]
    # )

    # Rotation from camera frame to pose frame
    ee_goal_in_camera_rotation = rotation @ rotation_bottom_mat.T
    # print(f"pin rotation{pin_rotation}")

    # Relative rotation and translation of grasping point relative to camera
    # target_in_camera_frame = pin.SE3(np.array(target_rotation_in_camera), np.array(target_point_in_camera))
    ee_goal_in_camera_pose = np.eye(4)
    ee_goal_in_camera_pose[:3, :3] = ee_goal_in_camera_rotation @ rotation_top_mat
    ee_goal_in_camera_pose[:3, 3] = ee_goal_in_camera_translation
    # print(f"pin dest frame {pin_dest_frame}")

    # Camera to gripper frame transformation
    # pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)

    # transformed_frame = del_pose * dest_frame
    # pin_transformed_frame = pin_cam2gripper_transform * target_in_camera_frame
    ee_goal_in_arm_base = camera_in_arm_base @ ee_goal_in_camera_pose
    # print(f"pin_transformed frame {pin_transformed_frame}")

    # === Visualization: arm_base and ee_goal_in_arm_base coordinate frames ===
    import open3d as o3d
    def _make_frame_from_T(T: np.ndarray, size: float = 0.15) -> o3d.geometry.TriangleMesh:
        R = T[:3, :3]
        t = T[:3, 3]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        # Build a 4x4 transform for Open3D
        T_o3d = np.eye(4)
        T_o3d[:3, :3] = R
        T_o3d[:3, 3] = t
        frame.transform(T_o3d)
        return frame

    geoms = []
    # arm_base at origin (identity)
    arm_base_T = np.eye(4)
    geoms.append(_make_frame_from_T(arm_base_T, size=0.18))
    # ee goal frame in arm_base
    geoms.append(_make_frame_from_T(ee_goal_in_arm_base, size=0.15))
    o3d.visualization.draw_geometries(geoms, window_name="arm_base (big) and ee_goal_in_arm_base")


    success, joints_solution, debug_info = manip_wrapper.robot._robot_model.compute_arm_ik(ee_goal_in_arm_base, verbose=False)

    # Lifting the arm to high position as part of pregrasping position
    # print("pan, tilt before", manip_wrapper.robot.get_pan_tilt())
    print("set gripper to suit position.")
    manip_wrapper.move_to_position(gripper_pos=gripper_width)
    # robot.move_to_position(lift_pos=1.05, head_pan=None, head_tilt=None)
    # print("pan, tilt after", robot.robot.get_pan_tilt())

    # Rotation for aligning Robot gripper frame to Model gripper frame
    # rotation2_top_mat = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])

    # final Rotation of gripper to hold the objcet
    # pin_final_rotation = np.dot(pin_transformed_frame.rotation, rotation2_top_mat)
    # print(f"pin final rotation {pin_final_rotation}")

    rpy_angles = pin.rpy.matrixToRpy(pin_final_rotation)
    print("pan, tilt before", robot.robot.get_pan_tilt())
    robot.move_to_pose(
        [0, 0, 0],
        [rpy_angles[0], rpy_angles[1], rpy_angles[2]],
        [1],
    )
    print("pan, tilt after", robot.robot.get_pan_tilt())

    # Final grasping point relative to camera
    pin_cam2gripper_transform = robot.get_joint_transform(base_node, gripper_node)
    pin_transformed_point1 = apply_se3_transform(pin_cam2gripper_transform, pin_point)
    # print(f"pin transformed point1 {pin_transformed_point1}")

    # Final grasping point relative to base
    pin_cam2base_transform = robot.get_joint_transform(base_node, "base_link")
    pin_base_point = apply_se3_transform(pin_cam2base_transform, pin_point)
    # print(f"pin base point {pin_base_point}")

    diff_value = (
        0.225 - gripper_depth - gripper_height
    )  # 0.225 is the distance between link_Straight_gripper node and the gripper tip
    pin_transformed_point1[2] -= diff_value
    ref_diff = diff_value

    # Moving gripper to a point that is 0.2m away from the pose center in the line of gripper
    print("pan, tilt before", robot.robot.get_pan_tilt())
    robot.move_to_pose(
        [pin_transformed_point1[0], pin_transformed_point1[1], pin_transformed_point1[2] - 0.2],
        [0, 0, 0],
        [1],
        move_mode=1,
    )
    print("pan, tilt after", robot.robot.get_pan_tilt())

    # Z-Axis of link_straight_gripper points in line of gripper
    # So, the z co-ordiante of point w.r.t gripper gives the distance of point from gripper
    pin_base2gripper_transform = robot.get_joint_transform("base_link", gripper_node)
    pin_transformed_point2 = apply_se3_transform(pin_base2gripper_transform, pin_base_point)
    curr_diff = pin_transformed_point2[2]

    # The distance between gripper and point is covered gradullay to allow for velocity control when it approaches the object
    # Lower velocity helps is not topping the light objects
    diff = abs(curr_diff - ref_diff)
    if diff > 0.08:
        dist = diff - 0.08
        state = robot.robot.get_six_joints()
        state[1] += 0.02
        state[2] += 0.02
        # state[0] -= 0.012
        robot.robot.arm_to(state, blocking=True)
        robot.move_to_pose([0, 0, dist], [0, 0, 0], [1])
        diff = diff - dist

    while diff > 0.01:
        dist = min(0.03, diff)
        robot.move_to_pose([0, 0, dist], [0, 0, 0], [1])
        diff = diff - dist

    # Now the gripper reached the grasping point and starts picking procedure
    robot.pickup(gripper_width)

    # Lifts the arm
    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1] + 0.2, 1.1))

    # Tucks the gripper so that while moving to place it won't collide with any obstacles
    robot.move_to_position(arm_pos=0.01)
    robot.move_to_position(wrist_pitch=0.0)
    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1], 0.9), wrist_yaw=2.5)
    robot.move_to_position(lift_pos=min(robot.robot.get_six_joints()[1], 0.55))
