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
import math
import os
import numpy as np
import pinocchio

from typing import List, Optional, Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation
from dream.core.interfaces import ContinuousFullBodyAction
from dream.motion.base import IKSolverBase
# from dream.motion.constants import (
#     MANIP_STRETCH_URDF,
#     PLANNER_STRETCH_URDF,
#     STRETCH_GRASP_FRAME,
#     STRETCH_HOME_Q,
# )
from dream.motion.pinocchio_ik_solver import PinocchioIKSolver, PositionIKOptimizer
from dream.motion.robot import Footprint
from scipy.spatial.transform import Rotation as R

# used for mapping joint states in STRETCH_*_Q to match the sim/real joint action space
# def map_joint_q_state_to_action_space(q):
#     return np.array(
#         [
#             q[4],  # arm_0
#             q[3],  # lift
#             q[8],  # yaw
#             q[7],  # pitch
#             q[6],  # roll
#             q[9],  # head pan
#             q[10],  # head tilt
#         ]
#     )


# class DreamIdx:
#     BASE_X = 0
#     BASE_Y = 1
#     BASE_THETA = 2
#     JOINT1 = 3
#     JOINT2 = 4
#     JOINT3 = 5
#     JOINT4 = 6
#     JOINT5 = 7
#     JOINT6 = 8
#     GRIPPER = 9

#     name_to_idx = {
#         "base_x": BASE_X,
#         "base_y": BASE_Y,
#         "base_theta": BASE_THETA,
#         "joint1": JOINT1,
#         "joint2": JOINT2,
#         "joint3": JOINT3,
#         "joint4": JOINT4,
#         "joint5": JOINT5,
#         "joint6": JOINT6,
#         "gripper": GRIPPER,
#     }

#     @classmethod
#     def get_idx(cls, name: str) -> int:
#         if name in cls.name_to_idx:
#             return cls.name_to_idx[name]
#         else:
#             raise ValueError(f"Unknown joint name: {name}")

# # Stores joint indices for the Stretch configuration space
# class HelloStretchIdx:
#     BASE_X = 0
#     BASE_Y = 1
#     BASE_THETA = 2
#     LIFT = 3
#     ARM = 4
#     GRIPPER = 5
#     WRIST_ROLL = 6
#     WRIST_PITCH = 7
#     WRIST_YAW = 8
#     HEAD_PAN = 9
#     HEAD_TILT = 10

#     name_to_idx = {
#         "base_x": BASE_X,
#         "base_y": BASE_Y,
#         "base_theta": BASE_THETA,
#         "lift": LIFT,
#         "arm": ARM,
#         "gripper_finger_right": GRIPPER,
#         "gripper": GRIPPER,
#         "wrist_roll": WRIST_ROLL,
#         "wrist_pitch": WRIST_PITCH,
#         "wrist_yaw": WRIST_YAW,
#         "head_pan": HEAD_PAN,
#         "head_tilt": HEAD_TILT,
#     }

#     @classmethod
#     def get_idx(cls, name: str) -> int:
#         if name in cls.name_to_idx:
#             return cls.name_to_idx[name]
#         else:
#             raise ValueError(f"Unknown joint name: {name}")


class RangerxARMKinematics:
    """Define motion planning structure for the robot. Exposes kinematics."""
    
    # Constants
    GRIPPER_OPEN = 830
    GRIPPER_CLOSED = 0
    
    # IK solver parameters (similar to PinocchioIKSolver)
    IK_EPS = 1e-4
    IK_MAX_ITER = 200
    IK_STEP_SIZE = 0.1
    IK_DAMPING = 1e-6
    JOINT5_TILI_RANGE = [deg for deg in range(45, 145, 1)]
    # Fixed transform from end-effector frame (link_eef) to camera frame (camera_link)
    # Units: meters. Provided by user calibration.
    CAMERA_IN_EE = np.array([
        [0.500, -0.000,  0.866,  0.100],
        [-0.000, -1.000, -0.000, -0.018],
        [0.866,  0.000, -0.500, -0.147],
        [0.000,  0.000,  0.000,  1.000],
    ], dtype=np.float32)
    
    def __init__(self, urdf_path: Optional[str] = None, verbose: bool = False):
        """Initialize with Pinocchio for local IK computation.
        
        Args:
            urdf_path: Path to URDF file. If None, uses default xarm6_kinematics.urdf
            verbose: Whether to print detailed initialization info
        """
        if urdf_path is None:
            urdf_path = str(
                Path(__file__).resolve().parents[2]  # .../DREAM/src
                / "dream_ros2_bridge" / "urdf" / "xarm6_kinematics.urdf"
            )
        
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_name = "link_eef"
        self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        self.verbose = verbose
        
        if verbose:
            print(f"[RangerxARM] Loaded URDF: {urdf_path}")
            print(f"[RangerxARM] Number of DOF: {self.model.nq}")
            print(f"[RangerxARM] End effector frame: {self.ee_frame_name}")
            print(f"[RangerxARM] Joint limits:")
            for i in range(min(6, self.model.nq)):
                lower = self.model.lowerPositionLimit[i]
                upper = self.model.upperPositionLimit[i]
                print(f"  Joint{i+1}: [{lower:.3f}, {upper:.3f}] rad = [{np.degrees(lower):.1f}°, {np.degrees(upper):.1f}°]")

    def get_footprint(self) -> Footprint:
        """Return footprint for the robot. This is expected to be a mask."""
        return Footprint(width=0.50, length=0.74, width_offset=0.0, length_offset=0.0)
    
    def compute_arm_ik(
        self,
        target_pose: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        return_degrees: bool = True,
        max_iterations: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[bool, Optional[np.ndarray], dict]:
        """Compute IK using Pinocchio iterative solver.
        
        Args:
            target_pose: 4x4 transformation matrix (position in meters)
            q_init: Initial joint angles [6] in radians. If None, uses a known good configuration.
            return_degrees: If True, return joint angles in degrees; otherwise radians
            max_iterations: Maximum IK iterations. If None, uses self.IK_MAX_ITER
            verbose: Print convergence info
        
        Returns:
            success: Whether IK converged
            joint_angles: Solution in degrees (if return_degrees=True) or radians [6]
            debug_info: Dict with 'iterations', 'final_error', 'final_error_norm'
        """
        if max_iterations is None:
            max_iterations = self.IK_MAX_ITER
        
        # Initialize joint configuration
        if q_init is None:
            q = np.deg2rad([0, -45, -90, 0, 110, 0])
        else:
            q = np.asarray(q_init, dtype=float)
        
        # Ensure q has correct size for Pinocchio model
        if len(q) != self.model.nq:
            q = np.concatenate([q, np.zeros(self.model.nq - 6)])
        
        # Convert target pose to SE3
        target = pinocchio.SE3(target_pose[:3, :3], target_pose[:3, 3])
        
        # Iterative IK solver
        for iteration in range(max_iterations):
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacements(self.model, self.data)
            
            # Compute error in SE3 space
            error = pinocchio.log(self.data.oMf[self.ee_frame_id].inverse() * target).vector
            error_norm = np.linalg.norm(error)
            
            if verbose and iteration % 20 == 0:
                print(f"[IK] iter={iteration}, error_norm={error_norm:.6f}")
            
            # Check convergence
            if error_norm < self.IK_EPS:
                joint_angles = q[:6]
                if return_degrees:
                    joint_angles = np.rad2deg(joint_angles)
                
                debug_info = {
                    'iterations': iteration,
                    'final_error': error,
                    'final_error_norm': error_norm
                }
                return True, joint_angles, debug_info
            
            # Compute Jacobian and update step
            J = pinocchio.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pinocchio.ReferenceFrame.LOCAL
            )
            dq = np.linalg.lstsq(J + self.IK_DAMPING * np.eye(6), error, rcond=None)[0]
            
            # Limit step size for stability
            step_norm = np.linalg.norm(dq)
            if step_norm > self.IK_STEP_SIZE:
                dq = dq * self.IK_STEP_SIZE / step_norm
            
            # Update configuration
            q = pinocchio.integrate(self.model, q, dq)
            
            # Enforce joint limits
            for i in range(min(6, self.model.nq)):
                q[i] = np.clip(q[i], self.model.lowerPositionLimit[i], self.model.upperPositionLimit[i])
        
        # Failed to converge
        debug_info = {
            'iterations': max_iterations,
            'final_error': error,
            'final_error_norm': error_norm
        }
        return False, None, debug_info
    
    def compute_arm_fk(
        self,
        joint_angles: np.ndarray,
        return_mm: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics.
        
        Args:
            joint_angles: Joint angles [6] in radians
            return_mm: If True, return position in mm; otherwise meters
        
        Returns:
            position: End effector position [x, y, z]
            rotation: End effector rotation matrix [3x3]
        """
        # Ensure q has correct size for Pinocchio model
        if len(joint_angles) != self.model.nq:
            q = np.concatenate([joint_angles, np.zeros(self.model.nq - 6)])
        else:
            q = joint_angles
        
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacements(self.model, self.data)
        
        pose = self.data.oMf[self.ee_frame_id]
        position = pose.translation.copy()
        
        if return_mm:
            position *= 1000
        
        return position, pose.rotation.copy()
    
    @staticmethod
    def pose_from_xyzrpy(
        x: float, y: float, z: float,
        roll: float, pitch: float, yaw: float,
        degrees: bool = True,
        input_mm: bool = False
    ) -> np.ndarray:
        """Construct 4x4 transformation matrix from position and orientation.
        
        Args:
            x, y, z: Position
            roll, pitch, yaw: Orientation (ZYX Euler angles)
            degrees: If True, angles are in degrees; otherwise radians
            input_mm: If True, position is in mm; otherwise meters
        
        Returns:
            4x4 transformation matrix
        """
        pose = np.eye(4)
        
        # Position (convert mm to m if needed)
        if input_mm:
            pose[:3, 3] = [x / 1000, y / 1000, z / 1000]
        else:
            pose[:3, 3] = [x, y, z]
        
        # Rotation (ZYX convention: yaw-pitch-roll)
        pose[:3, :3] = R.from_euler('ZYX', [yaw, pitch, roll], degrees=degrees).as_matrix()
        
        return pose



    def compute_look_at_target(
        self,
        arm_angles_deg: np.ndarray,
        target_in_map_point: np.ndarray,
        base_in_map_pose: np.ndarray,
        camera_in_base_pose: np.ndarray,
        ee_in_base_pose: np.ndarray,
        joint_index: int = 4,          # Joint-5 (0-based index)
        deadband_deg: float = 0.3      # No movement if error smaller than this (degrees)
    ) -> np.ndarray:
        """Move a single joint (default: joint-5) to center target on camera Y-axis.

        Simple and robust approach:
        - Build candidates around the current joint angle within physical limits
        (default: ±30 degrees, step 1 degree)
        - For each candidate, run FK, get camera pose as base_T_ee @ CAMERA_IN_EE
        - Score = abs(atan2(y_cam, z_cam)) using unit direction vector
        - Return angles with the best score
        """

        # 1) Prepare target in base frame
        map_T_base = np.linalg.inv(base_in_map_pose)
        tgt_h = np.array([target_in_map_point[0], target_in_map_point[1], target_in_map_point[2], 1.0], dtype=float)
        target_in_base = (map_T_base @ tgt_h)[:3]

        # 2) Quick deadband check using the current camera pose
        R_cam_cur = camera_in_base_pose[:3, :3]
        p_cam_cur = camera_in_base_pose[:3, 3]
        d_base_cur = target_in_base - p_cam_cur
        dist_cur = float(np.linalg.norm(d_base_cur))
        if dist_cur > 1e-9:
            dir_cam_cur = R_cam_cur.T @ (d_base_cur / dist_cur)
            cur_err_deg = abs(np.degrees(np.arctan2(dir_cam_cur[1], dir_cam_cur[2])))
            if cur_err_deg < deadband_deg:
                return arm_angles_deg

        # 3) Build candidate angles within limits
        lo_deg = float(np.degrees(self.model.lowerPositionLimit[joint_index]))
        hi_deg = float(np.degrees(self.model.upperPositionLimit[joint_index]))

        candidates_deg = [deg for deg in self.JOINT5_TILI_RANGE if deg >= lo_deg and deg <= hi_deg]

        # 4) Brute-force search
        best_angle_deg = arm_angles_deg[joint_index]
        best_err_deg = float('inf')

        base_T_ee_cam = np.linalg.inv(ee_in_base_pose) @ camera_in_base_pose
        err_deg_dict = {}
        q0_deg = np.asarray(arm_angles_deg, dtype=float)
        for a_deg in candidates_deg:
            q_try_deg = q0_deg.copy()
            q_try_deg[joint_index] = a_deg

            q_try = np.radians(q_try_deg)
            if len(q_try) != self.model.nq:
                q_try = np.concatenate([q_try, np.zeros(self.model.nq - len(q_try))])

            # FK: base_T_ee for the test pose
            pinocchio.forwardKinematics(self.model, self.data, q_try)
            pinocchio.updateFramePlacements(self.model, self.data)
            ee_pose = self.data.oMf[self.ee_frame_id]
            base_T_ee = ee_pose.homogeneous

            # Camera pose for the test pose
            base_T_cam = base_T_ee @ base_T_ee_cam
            R_cam = base_T_cam[:3, :3]
            p_cam = base_T_cam[:3, 3]

            # Direction from camera to target in camera frame
            d_base = target_in_base - p_cam
            dist = float(np.linalg.norm(d_base))
            if dist <= 1e-9:
                err_deg = 0.0
            else:
                dir_cam = R_cam.T @ (d_base / dist)
                if dir_cam[2] <= 0:
                    # Target behind the camera; skip this candidate
                    continue
                err_deg = abs(np.degrees(np.arctan2(dir_cam[1], dir_cam[2])))
            err_deg_dict[a_deg] = err_deg
            if err_deg < best_err_deg:
                best_err_deg = err_deg
                best_angle_deg = a_deg

        # If nothing improved, keep as-is
        if not np.isfinite(best_err_deg):
            return arm_angles_deg

        result = arm_angles_deg.copy()
        result[joint_index] = best_angle_deg
        return result




if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    
    print("=" * 60)
    print("RangerxARM Kinematics Test")
    print("=" * 60)
    
    kinematics = RangerxARMKinematics(verbose=True)
    
    # Test 1: Forward Kinematics
    print("\n[Test 1] Forward Kinematics")
    print("-" * 60)
    joints_deg = [0, -45, -90, 0, 110, 0]
    joints_rad = np.deg2rad(joints_deg)
    
    position_m, rotation = kinematics.compute_arm_fk(joints_rad, return_mm=False)
    position_mm, _ = kinematics.compute_arm_fk(joints_rad, return_mm=True)
    roll, pitch, yaw = np.rad2deg(R.from_matrix(rotation).as_euler('xyz'))
    
    print(f"Input joints (deg): {joints_deg}")
    print(f"Position (m): [{position_m[0]:.4f}, {position_m[1]:.4f}, {position_m[2]:.4f}]")
    print(f"Position (mm): [{position_mm[0]:.1f}, {position_mm[1]:.1f}, {position_mm[2]:.1f}]")
    print(f"Orientation (deg): roll={roll:.1f}, pitch={pitch:.1f}, yaw={yaw:.1f}")
    
    # Test 2: Inverse Kinematics (using pose_from_xyzrpy helper)
    print("\n[Test 2] Inverse Kinematics")
    print("-" * 60)
    target_x, target_y, target_z = 204.3, -8.9, 590.2  # mm
    target_roll, target_pitch, target_yaw = 177.8, -25, 0  # deg
    
    target_pose = RangerxARMKinematics.pose_from_xyzrpy(
        target_x, target_y, target_z,
        target_roll, target_pitch, target_yaw,
        degrees=True,
        input_mm=True
    )
    
    print(f"Target position (mm): [{target_x}, {target_y}, {target_z}]")
    print(f"Target orientation (deg): roll={target_roll}, pitch={target_pitch}, yaw={target_yaw}")
    
    success, joints_solution, debug_info = kinematics.compute_arm_ik(target_pose, verbose=False)
    
    if success:
        print(f"✓ IK converged in {debug_info['iterations']} iterations")
        print(f"  Final error norm: {debug_info['final_error_norm']:.6f}")
        print(f"  Solution (deg): {np.round(joints_solution, 2).tolist()}")
        
        # Verify with FK
        pos_verify, rot_verify = kinematics.compute_arm_fk(np.deg2rad(joints_solution), return_mm=True)
        print(f"  FK verification (mm): [{pos_verify[0]:.1f}, {pos_verify[1]:.1f}, {pos_verify[2]:.1f}]")
    else:
        print(f"✗ IK failed after {debug_info['iterations']} iterations")
        print(f"  Final error norm: {debug_info['final_error_norm']:.6f}")
    
    # # Test 3: Look at target (简化版 - 只调整joint5)
    # print("\n[Test 3] Look at Target - Joint5 Only (简化版)")
    # print("-" * 60)
    # current_joints = [0.4, -9.2, -112.6, -0.5, 96.8, 0.3]  # degrees
    # target_point = np.array([0.5, 0.0, 0.2])  # meters in arm base frame
    
    # print(f"当前关节角度 (度): {current_joints}")
    # print(f"目标点 (米): {target_point.tolist()}")
    
    # new_joint5 = kinematics.compute_joint5_look_at(
    #     np.array(current_joints),
    #     target_point
    # )
    
    # print(f"✓ 新的joint5角度: {new_joint5:.1f}° (原来是 {current_joints[4]:.1f}°)")
    # print(f"  变化: {new_joint5 - current_joints[4]:+.1f}°")
    
    # # 完整的新关节角度
    # new_joints = current_joints.copy()
    # new_joints[4] = new_joint5
    # print(f"  完整关节角度: {[round(j, 1) for j in new_joints]}")
    
    # print("\n" + "=" * 60)
