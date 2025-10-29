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

from transforms3d.quaternions import mat2quat
from transforms3d.euler import quat2euler

def quat2rpy(quat):
    # Convert to numpy array
    if type(quat) is list:
        quat = np.array(quat)
    # Convert to rpy
    rpy = np.array(quat2euler(quat, axes="sxyz")) / np.pi * 180
    return rpy


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
    # Units: meters. Provided by user calibration.  # ros2 run tf2_ros tf2_echo link_eef camera_color_optical_frame
    # CAMERA_IN_EE = np.array([
    #     [-0.002, -0.868,  0.496,  0.100],
    #     [1.000, -0.005, -0.004, -0.032],
    #     [0.006,  0.496,  0.868, -0.147],
    #     [0.000,  0.000,  0.000,  1.000],
    # ], dtype=np.float32)

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

    @staticmethod
    def xyzrpy_from_pose(
        pose: np.ndarray,
        degrees: bool = True,
        output_mm: bool = True
    ) -> Tuple[float, float, float, float, float, float]:
        """Extract position and ZYX-orientation (roll, pitch, yaw) from 4x4 pose.

        Uses the same convention as pose_from_xyzrpy: rotation = R.from_euler('ZYX', [yaw, pitch, roll]).
        """
        if pose.shape != (4, 4):
            raise ValueError(f"pose must be 4x4, got {pose.shape}")

        x, y, z = pose[:3, 3].astype(float)
        if output_mm:
            x, y, z = x * 1000.0, y * 1000.0, z * 1000.0

        # as_euler('ZYX') returns [yaw, pitch, roll]
        yaw, pitch, roll = R.from_matrix(pose[:3, :3]).as_euler('ZYX', degrees=degrees)
        return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)


    def compute_look_at_target_tilt(
        self,
        arm_angles_deg: np.ndarray,
        target_in_map_point: np.ndarray,
        base_in_map_pose: np.ndarray,
        camera_in_base_pose: np.ndarray,
        ee_in_base_pose: np.ndarray,
    ) -> np.ndarray:

        target_in_map_homo = np.array([target_in_map_point[0], target_in_map_point[1], target_in_map_point[2], 1.0], dtype=float)
        target_in_base = (np.linalg.inv(base_in_map_pose) @ target_in_map_homo)[:3]
        camera_in_ee = np.linalg.inv(ee_in_base_pose) @ camera_in_base_pose

        R_cam_in_base_cur = camera_in_base_pose[:3, :3]
        p_cam_in_base_cur = camera_in_base_pose[:3, 3]

        all_attempts = 10
        for attempt in range(all_attempts):
            if attempt > 0:
                p_cam_in_base_cur[2] -= 20
            distance_base_cur = target_in_base - p_cam_in_base_cur
            dist_cur = float(np.linalg.norm(distance_base_cur))

            dir_cam_z = distance_base_cur / dist_cur
            dir_cam_cur = R_cam_in_base_cur.T @ (dir_cam_z)

            tilt_err = np.arctan2(dir_cam_cur[1], dir_cam_cur[2])

            c, s = np.cos(-tilt_err), np.sin(-tilt_err)
            Rx = np.array([[1.0, 0.0, 0.0],
                        [0.0,   c,  -s],
                        [0.0,   s,   c]], dtype=float)
            R_cam_in_base_new = R_cam_in_base_cur @ Rx

            camera_in_base_pose_new = np.eye(4, dtype=float)
            camera_in_base_pose_new[:3, :3] = R_cam_in_base_new
            camera_in_base_pose_new[:3, 3] = p_cam_in_base_cur

            ee_in_base_pose_new = camera_in_base_pose_new @ np.linalg.inv(camera_in_ee)

            q_init = np.deg2rad(arm_angles_deg)
            success, arm_angles_deg_new, _ = self.compute_arm_ik(
                ee_in_base_pose_new,
                q_init=q_init,
                return_degrees=True,
                verbose=False
            )
            if success:
                break



        # quat = mat2quat(ee_in_base_pose_new[:3, :3])
        # tar_xyz = (ee_in_base_pose_new[:3, 3] * 1000)
        # tar_rpy = quat2rpy(quat)
        # joint_positions_goal = list(tar_xyz) + list(tar_rpy)

        # joint_positions_goal = self.xyzrpy_from_pose(ee_in_base_pose_new)


        # self.visualize_camera_poses(
        #     camera_in_base_pose=camera_in_base_pose,
        #     camera_in_base_pose_new=camera_in_base_pose_new,
        #     target_point=target_in_base,
        #     axis_len=0.15,
        #     title="camera_in_base",
        #     save=None,
        # )

        # self.visualize_camera_poses(
        #     camera_in_base_pose=ee_in_base_pose,
        #     camera_in_base_pose_new=ee_in_base_pose_new,
        #     target_point=target_in_base,
        #     axis_len=0.15,
        #     title="ee_in_base",
        #     save=None,
        # )

        if success:
            return arm_angles_deg_new
        return arm_angles_deg

    def visualize_frames(
        self,
        poses: List[Tuple[np.ndarray, str]],
        axis_len: float = 0.1,
        title: Optional[str] = None,
        point: Optional[np.ndarray] = None,
        save: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        I = np.eye(4)
        self._draw_frame(ax, I, "base", axis_len=axis_len, alpha=0.25, linewidth=1.0)

        for T, label in poses:
            self._draw_frame(ax, T, label, axis_len=axis_len, alpha=1.0, linewidth=2.0)

        if point is not None:
            p = np.asarray(point, dtype=float).reshape(3)
            ax.scatter([p[0]], [p[1]], [p[2]], c="k", s=40, label="target")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        if title:
            ax.set_title(title)

        # bounds
        all_pts = [T[:3, 3] for T, _ in poses]
        if point is not None:
            all_pts.append(np.asarray(point, dtype=float).reshape(3))
        if all_pts:
            all_pts = np.array(all_pts)
            mins = all_pts.min(axis=0)
            maxs = all_pts.max(axis=0)
            pad = max(1e-3, 0.1 * float(np.linalg.norm(maxs - mins)))
            ax.set_xlim(mins[0] - pad, maxs[0] + pad)
            ax.set_ylim(mins[1] - pad, maxs[1] + pad)
            ax.set_zlim(mins[2] - pad, maxs[2] + pad)
        self._set_axes_equal(ax)

        if point is not None:
            ax.legend(loc="upper left")

        if save:
            import os
            os.makedirs(os.path.dirname(save), exist_ok=True)
            plt.savefig(save, bbox_inches="tight", dpi=150)
        plt.show()

    def visualize_camera_poses(
        self,
        camera_in_base_pose: np.ndarray,
        camera_in_base_pose_new: np.ndarray,
        target_point: Optional[np.ndarray] = None,
        axis_len: float = 0.1,
        title: Optional[str] = "Camera frames in base",
        save: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        I = np.eye(4)
        self._draw_frame(ax, I, "base", axis_len=axis_len, alpha=0.25, linewidth=1.0)

        T_cur = np.asarray(camera_in_base_pose, dtype=float)
        T_new = np.asarray(camera_in_base_pose_new, dtype=float)

        # 当前相机细线，新相机加粗
        self._draw_frame(ax, T_cur, "cur", axis_len=axis_len, alpha=0.9, linewidth=2.0)
        self._draw_frame(ax, T_new, "new", axis_len=axis_len, alpha=1.0, linewidth=4.0)

        if target_point is not None:
            p = np.asarray(target_point, dtype=float).reshape(3)
            ax.scatter([p[0]], [p[1]], [p[2]], c="k", s=40, label="target")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        if title:
            ax.set_title(title)

        # bounds
        all_pts = [T_cur[:3, 3], T_new[:3, 3]]
        if target_point is not None:
            all_pts.append(np.asarray(target_point, dtype=float).reshape(3))
        all_pts = np.array(all_pts)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        pad = max(1e-3, 0.1 * float(np.linalg.norm(maxs - mins)))
        ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        ax.set_zlim(mins[2] - pad, maxs[2] + pad)
        self._set_axes_equal(ax)

        if target_point is not None:
            ax.legend(loc="upper left")

        if save:
            import os
            os.makedirs(os.path.dirname(save), exist_ok=True)
            plt.savefig(save, bbox_inches="tight", dpi=150)
        plt.show()

    @staticmethod
    def _draw_frame(ax, T: np.ndarray, label: str, axis_len: float, alpha: float = 1.0, linewidth: float = 2.0) -> None:
        Rm = T[:3, :3]
        p = T[:3, 3]
        x_axis = Rm[:, 0] * axis_len
        y_axis = Rm[:, 1] * axis_len
        z_axis = Rm[:, 2] * axis_len
        ax.plot([p[0], p[0] + x_axis[0]], [p[1], p[1] + x_axis[1]], [p[2], p[2] + x_axis[2]], color="r", alpha=alpha, linewidth=linewidth)
        ax.plot([p[0], p[0] + y_axis[0]], [p[1], p[1] + y_axis[1]], [p[2], p[2] + y_axis[2]], color="g", alpha=alpha, linewidth=linewidth)
        ax.plot([p[0], p[0] + z_axis[0]], [p[1], p[1] + z_axis[1]], [p[2], p[2] + z_axis[2]], color="b", alpha=alpha, linewidth=linewidth)
        ax.text(p[0], p[1], p[2], f" {label}", color="k")

    @staticmethod
    def _set_axes_equal(ax) -> None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max([x_range, y_range, z_range])
        x_middle = float(np.mean(x_limits))
        y_middle = float(np.mean(y_limits))
        z_middle = float(np.mean(z_limits))
        ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
        ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
        ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

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
