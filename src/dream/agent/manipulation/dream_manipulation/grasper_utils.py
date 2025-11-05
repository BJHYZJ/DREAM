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

from dream.agent.manipulation.dream_manipulation.image_publisher import ImagePublisher, DreamCamera
from dream.agent.manipulation.dream_manipulation.place import Placing
from dream.agent.manipulation.dream_manipulation.dream_manipulation import (
    DreamManipulationWrapper as ManipulationWrapper,
)
from dream.agent.zmq_client_dream import DreamRobotZmqClient
import dream.motion.constants as constants

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


def capture_and_process_image(mode, obj, tar_in_map, socket, manip_wrapper: ManipulationWrapper):
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
    rotation_min_theta_abs = 0.15  # ~8 degree
    # head_tilt = 105
    # head_pan = -1.57

    print("*" * 20, f"look at {obj}", "*" * 20)
    manip_wrapper.robot.look_at_target(tar_in_map=tar_in_map)

    while retry_flag:

        print("Capturing image: ")
        print(f"retry flag : {retry_flag}")
        print(f"side retries : {side_retries}")
        print(f"tilt retries : {tilt_retries}")

        translation, rotation, depth, width, c2ab, obj_points, retry_flag = image_publisher.publish_image(obj, mode)

        if retry_flag == 1:
            # center robot by detected object translation
            target_in_cam = np.array(translation).astype(np.float32)
            obj_in_arm_base = c2ab[:3, :3] @ target_in_cam + c2ab[:3, 3]
            arm_base_in_map_pose = manip_wrapper.robot.get_transform(transfrom_name="arm_base_in_map_pose")  # arm base link
            base_in_map_pose = manip_wrapper.robot.get_transform(transfrom_name="base_in_map_pose")  # base_link
            # calculation for base rotation angle
            target_in_map = (arm_base_in_map_pose[:3, :3] @ obj_in_arm_base.T).T + arm_base_in_map_pose[:3, 3]
            base_in_map_xyz = base_in_map_pose[:3, 3]
            base_x_axis =  base_in_map_pose[:3, :3] @ np.array([1, 0, 0])
            vec_to_obj = target_in_map - base_in_map_xyz
            u = base_x_axis[:2] / np.linalg.norm(base_x_axis[:2])
            v = vec_to_obj[:2] / np.linalg.norm(vec_to_obj[:2])
            dot = np.clip(np.dot(u, v), -1.0, 1.0)
            det = u[0] * v[1] - u[1] * v[0]
            theta = np.arctan2(det, dot)
            if abs(theta) > rotation_min_theta_abs:
                print(f"Robot base theta is not suit for manipulation, rotate around {np.rad2deg(theta)}.")
                manip_wrapper.move_to_position(base_theta=theta, blocking=True)
            manip_wrapper.robot.look_at_target(tar_in_map=target_in_map)
            # manip_wrapper.move_to_position(
            #     base_trans=base_trans, head_pan=head_pan, head_tilt=head_tilt
            # )

        elif retry_flag != 0 and side_retries == 3:
            print("Tried in all angles but couldn't succeed")
            if mode == "place":
                return None, None, None, None, None
            else:
                return None, None, None, None, None, None

        elif side_retries == 2 and tilt_retries == 3:
            manip_wrapper.move_to_position(base_theta=np.deg2rad(15))
            side_retries = 3
            theta_cumulative += 15

        elif retry_flag == 2:
            if tilt_retries == 3:
                if side_retries == 0:
                    manip_wrapper.move_to_position(base_theta=np.deg2rad(15))
                    side_retries = 1
                    theta_cumulative += 15
                else:
                    manip_wrapper.move_to_position(base_theta=np.deg2rad(-30))
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
        return rotation, translation, depth, width, c2ab, obj_points, theta_cumulative
    else:
        print("Place: Returning translation, rotation")
        return rotation, translation, c2ab, obj_points, theta_cumulative


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

def pregrasp_position(
    object_xyz: np.ndarray,  # arm base
    ee_xyz: np.ndarray,  # arm base
    distance_from_object: float=0.35
) -> np.ndarray:
    vector_to_object = object_xyz - ee_xyz
    vector_to_object = vector_to_object / np.linalg.norm(vector_to_object)
    print("Relative object xyz was:", object_xyz)
    shifted_object_xyz = object_xyz - (distance_from_object * vector_to_object)
    print("Pregrasp xyz:", shifted_object_xyz)
    return shifted_object_xyz


def pickup(
    manip_wrapper: ManipulationWrapper,
    rotation: np.ndarray,
    translation: np.ndarray,
    c2ab: np.ndarray,
    object_points: np.ndarray,  # obj_points in camera
    gripper_width: int=830,
    distance_from_object: float=0.35
):
    ee_goal_in_camera_pose = np.eye(4)
    ee_goal_in_camera_pose[:3, :3] = rotation
    ee_goal_in_camera_pose[:3, 3] = translation
    ee_goal_in_arm_base_pose = c2ab @ ee_goal_in_camera_pose
    
    assert len(object_points.shape) == 2 and object_points.shape[1] == 3
    object_points_in_arm_base = (c2ab[:3, :3] @ object_points.T).T + c2ab[:3, 3] # object points in arm base

    arm_angles_deg = manip_wrapper.robot.get_arm_joint_state()
    success0, joints_solution0, debug_info0 = manip_wrapper.robot._robot_model.manip_ik(
        ee_goal_in_arm_base_pose, 
        q_init=arm_angles_deg, 
        is_radians=False, 
        verbose=False
    )

    if not success0:
        print("Anygrasp pose can be resolve by IK, try to use heuristic pickup strategy, Good Luck!")
        ee_in_arm_base_pose = manip_wrapper.robot.get_ee_in_arm_base()
        # try use heuristic pickup
        # transfer obj_points to arm base frame
        centered_pts = object_points_in_arm_base - np.median(object_points_in_arm_base, axis=0, keepdims=True)
        cov = np.cov(centered_pts, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal_axis = eig_vecs[:, np.argmax(eig_vals)]
        object_yaw = np.arctan2(principal_axis[1], principal_axis[0])
        object_yaw += np.pi / 2

        ee_rot = ee_in_arm_base_pose[:3, :3]
        current_gripper_yaw = np.arctan2(ee_rot[1, 0], ee_rot[0, 0])
        delta_yaw = np.arctan2(
            np.sin(object_yaw - current_gripper_yaw),
            np.cos(object_yaw - current_gripper_yaw),
        )
        print(f"Heuristic gripper yaw delta (deg): {np.degrees(delta_yaw):.2f}")

        # Build a new target pose that keeps pitch/roll but enforces yaw and a reasonable position
        desired_position = np.median(object_points_in_arm_base, axis=0)
        desired_position[2] = np.median(object_points_in_arm_base[:, 2]) + 0.015

        Rz = np.array(
            [
                [np.cos(delta_yaw), -np.sin(delta_yaw), 0.0],
                [np.sin(delta_yaw),  np.cos(delta_yaw), 0.0],
                [0.0,                0.0,               1.0],
            ]
        )
        R_current = ee_in_arm_base_pose[:3, :3]
        R_desired = R_current @ Rz
        ee_goal_in_arm_base_pose = np.eye(4)  # Overwrite the old ee_goal_in_arm_base_pose
        ee_goal_in_arm_base_pose[:3, :3] = R_desired
        ee_goal_in_arm_base_pose[:3, 3] = desired_position

        # Use yaw-updated configuration as IK seed
        arm_angles_deg_new = arm_angles_deg.copy()
        arm_angles_deg_new[5] += np.degrees(delta_yaw)

        success1, joints_solution1, debug_info1 = manip_wrapper.robot._robot_model.manip_ik(
            ee_goal_in_arm_base_pose,
            q_init=arm_angles_deg_new,
            is_radians=False,
            verbose=False,
        )

    if not success0 and not success1:
        print("ಥ﹏ಥ Both the Anygrasp and Heuristic methods were ineffective and failed.")
        return False
    elif success0:
        joints_solution = joints_solution0
    elif success1:
        joints_solution = joints_solution1
    else:
        raise ValueError


    # ================ process pregrasp while will imporve manipulation success rate ================
    object_xyz = np.median(object_points_in_arm_base, axis=0) 
    ee_xyz = manip_wrapper.robot.get_ee_in_arm_base()[:3, 3]
    pregrasp_xyz = pregrasp_position(
        object_xyz=object_xyz, 
        ee_xyz=ee_xyz, 
        distance_from_object=distance_from_object
    )
    pregrasp_pose = np.eye(4)
    pregrasp_pose[:3, 3] = pregrasp_xyz
    pregrasp_pose[:3, :3] = ee_goal_in_arm_base_pose[:3, :3]

    pre_success, pregrasp_joint_angles, _ = manip_wrapper.robot._robot_model.manip_ik(
        pregrasp_pose, 
        q_init=arm_angles_deg, 
        is_radians=False, 
        verbose=False
    )

    # pause slam to avoid robot body be scan to scene
    manip_wrapper.robot.pause_slam(reliable=True)
    if pre_success:
        print(f"Moving to pre-grasp position.")
        print("Pregrasp joint angles: ")
        print(" - joint1: ", pregrasp_joint_angles[0])
        print(" - joint2: ", pregrasp_joint_angles[1])
        print(" - joint3: ", pregrasp_joint_angles[2])
        print(" - joint4: ", pregrasp_joint_angles[3])
        print(" - joint5: ", pregrasp_joint_angles[4])
        print(" - joint5: ", pregrasp_joint_angles[5])
        manip_wrapper.robot.arm_to(pregrasp_joint_angles, blocking=True)   

    manip_wrapper.robot.arm_to(angle=joints_solution, blocking=True)
    picked = manip_wrapper.pickup(width=gripper_width)
    if not picked:
        print("(ಥ﹏ಥ) It failed because I didn't do it properly...")
        manip_wrapper.robot.resume_slam(reliable=True)
        return False
    # place_black, the camera will look at robot body, pause slam to avoid add robot mesh to scene
    manip_wrapper.place_back()
    manip_wrapper.robot.resume_slam(reliable=True)
    return True