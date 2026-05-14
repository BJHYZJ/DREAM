"""
 * Codes cited from AnyGrasp: Robust and Efficient Grasp Perception in Spatial and Temporal Domains
 * Author: Fang, Hao-Shu and Wang, Chenxi and Fang, Hongjie and Gou, Minghao and Liu, Jirong and Yan, Hengxu and Liu, Wenhai and Xie, Yichen and Lu, Cewu
 * GitHub: https://github.com/graspnet/anygrasp_sdk
 * All rights reserved by Fang, Hao-Shu.
 *
 * Modifications were made for integration purposes.
"""

import copy
import math
import time
from pathlib import Path
import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup
from gsnet import AnyGrasp
# from image_processors import LangSAMProcessor
from PIL import Image
from utils.camera import CameraParameters
from utils.types import Bbox
from utils.utils import draw_rectangle, get_3d_points, sample_points, visualize_cloud_geometries, create_arrow
from utils.zmq_socket import ZmqSocket

from image_processors import OWLSAMProcessor

from datetime import datetime


class ObjectHandler:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.grasping_model = AnyGrasp(self.cfgs)
        self.grasping_model.load_net()

        if self.cfgs.open_communication:
            self.socket = ZmqSocket(self.cfgs)

        self.lang_sam = OWLSAMProcessor()
        # self.lang_sam = LangSAMProcessor()
        self.rerun_frame = 1

        self.robot_min_height = -0.3,
        self.robot_max_height = 1.0

    def receive_input(self, tries):
        if self.cfgs.open_communication:
            print("\n\nWaiting for data from Robot")
            # Reading color array
            colors = self.socket.recv_rgb_img()
            self.socket.send_data("RGB received")

            # Depth data
            depths = self.socket.recv_depth_img()
            # print(np.max(depths), np.min(depths))
            self.socket.send_data("depth received")

            # Camera Intrinsics
            fx, fy, cx, cy = self.socket.recv_array()
            self.socket.send_data("intrinsics received")

            # camera in arm transform
            c2ab = self.socket.recv_array()
            self.socket.send_data("camera in arm base pose received")

            # Object query
            self.query = self.socket.recv_string()
            self.socket.send_data("text query received")
            print(f"Text - {self.query}")

            # action -> ["pick", "place", "pick_back"]
            self.action = self.socket.recv_string()
            self.socket.send_data("Mode received")
            print(f"Manipualtion Mode - {self.action}")
            print(self.socket.recv_string())

            image = Image.fromarray(colors)
        else:
            data_dir = Path("example_data")
            image_path = data_dir / "test_image.png"
            depth_path = data_dir / "test_depth.png"
            colors = np.array(Image.open(image_path))
            image = Image.open(image_path)
            depths = np.array(Image.open(depth_path)) * 0.001
            fx, fy, cx, cy = 454.06378173828125, 453.5252380371094, 319.597900390625, 180.85589599609375

            c2ab = np.eye(4)
            if tries == 1:
                self.action = "pick"  # you can type "pick", "place"
                self.query = "remote control"  # you can type "bowl" "pliers" "screwdriver" "remote control"

        # Camera Parameters
        colors = colors / 255.0
        self.cam = CameraParameters(fx, fy, cx, cy, image, colors, depths, c2ab)

    def manipulate(self):
        """
        Wrapper for grasping and placing

        11 is the maximum number of retries in case of object or grasp detection failure
        Try - 1 -> captures image and centers the robot
        Try - 2 -> captures image and tries to perform action
        If failed:
            Try - 3,4 -> tries in a different camera orientation
        Even then if it fails:
            Try - 5,6,7 -> moves base to left tries again three different camera orientations
        Even then if it fails:
            Try - 8,9,10 -> moves base to the right and again tries three different camera orientations
        Finally if it fails to detect any pose in above all attempts:
            Try 11 -> If object is detected but anygrasp couldn't find a pose as a last resort
                      the cropped object image is sent to the model.
        In any of the above attempts it is able to succeed it won't perform any further tries
        """

        cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tries = 1
        retry = True
        while retry and tries <= 11:
            self.receive_input(tries)

            # Directory for saving visualisations
            self.save_dir = Path(self.cfgs.environment) / self.query / cur_time
            self.save_dir.mkdir(parents=True, exist_ok=True)
            show_debug_images = self.cfgs.debug and not self.cfgs.headless

            image_file_name = self.save_dir / f"image_{tries}.png"
            depth_file_name = self.save_dir / f"depth_{tries}.png"
            depth = np.nan_to_num(self.cam.depths, nan=0.0, posinf=0.0, neginf=0.0)
            depth = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            self.cam.image.save(image_file_name)
            Image.fromarray(depth).save(depth_file_name)
            print(f"Saving the camera image at {image_file_name}")
            print(f"Saving the depth image at {depth_file_name}")
            if show_debug_images:
                self.cam.image.show()
                Image.fromarray(depth).show()

            box_filename = self.save_dir / f"bbox_{tries}.png"
            mask_filename = self.save_dir / f"mask_{tries}.png"

            # Object Segmentation Mask
            seg_mask, bbox = self.lang_sam.detect_obj(
                self.cam.image,
                self.query,
                box_filename=box_filename,
                mask_filename=mask_filename,
                visualize=show_debug_images,
            )

            if bbox is None:
                if self.cfgs.open_communication:
                    print("Didn't detect the object, Trying Again")
                    tries = tries + 1
                    print(f"Try no: {tries}")
                    data_msg = "No Objects detected, Have to try again"
                    self.socket.send_data([[0], [0], [0, 0, 2], [], data_msg])
                    if tries == 11:
                        return
                    continue
                else:
                    print(
                        "Didn't find the Object. Try with another object or tune grasper height and width parameters in demo.py"
                    )
                    retry = False
                    continue

            print(f"{self.query} detected !!!")

            # Center the robot
            if tries == 1 and self.cfgs.open_communication and self.action != "pick_back":
                self.center_robot(bbox)
                tries += 1
                time.sleep(0.5)
                continue

            points = get_3d_points(self.cam)

            if self.action == "place":
                retry = not self.place(points, seg_mask, tries)
            elif self.action == "pick":
                retry = not self.pickup(points, seg_mask, bbox, tries, (tries == 11))
            elif self.action == "pick_back":
                retry = not self.pick_back(points, seg_mask, tries)
            else:
                raise ValueError

            if retry:
                if self.cfgs.open_communication:
                    print("Trying Again")
                    tries = tries + 1
                    print(f"Try no: {tries}")
                    data_msg = "No poses, Have to try again"
                    self.socket.send_data([[0], [0], [0, 0, 2], [], data_msg])
                else:
                    print(
                        "Try with another object or tune grasper height and width parameters in demo.py"
                    )
                    retry = False


    def center_robot(self, bbox: Bbox):
        """
        Center the robots base and camera to face the center of the Object Bounding box
        """
        c2ab = self.cam.c2ab  # camera in arm base pose
        R_c2ab = c2ab[:3, :3]
        t_c2ab = c2ab[:3,  3]
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
        u = int((bbox_x_min + bbox_x_max) / 2)
        v = int((bbox_y_min + bbox_y_max) / 2)
        Z = self.cam.depths[v, u].item()
        X = (u - self.cam.cx) / self.cam.fx * Z
        Y = (v - self.cam.cy) / self.cam.fy * Z
        p_cam = np.array([X, Y, Z])
        p_arm_base = R_c2ab @ p_cam + t_c2ab
        print(f"Object center in arm base frame: {p_arm_base.tolist()}")

        if self.cfgs.open_communication:
            data_msg = f"Object center in arm base frame: {p_arm_base.tolist()} is received."
            self.socket.send_data([p_arm_base, [], [0, 0, 1], [], data_msg])

    def place(
        self, 
        points: np.ndarray, 
        seg_mask: np.ndarray,
        tries: int,
    ) -> bool:
        colors = self.cam.colors
        c2ab = self.cam.c2ab  # camera in arm base pose
        R_c2ab = c2ab[:3, :3]
        t_c2ab = c2ab[:3,  3]

        points_x, points_y, points_z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
        flat_x, flat_y, flat_z = (
            points_x.reshape(-1),
            points_y.reshape(-1),
            points_z.reshape(-1),
        )
        # Removing all points whose depth is zero(undetermined)
        zero_depth_seg_mask = (
            (flat_x != 0)
            * (flat_y != 0)
            * (flat_z != 0)
            * (~np.isnan(flat_z))
            * seg_mask.reshape(-1)
        )
        flat_x = flat_x[zero_depth_seg_mask]
        flat_y = flat_y[zero_depth_seg_mask]
        flat_z = flat_z[zero_depth_seg_mask]

        filtered_colors = colors.reshape(-1, 3)[zero_depth_seg_mask]
        filtered_points = (R_c2ab @ np.stack([flat_x, flat_y, flat_z], axis=-1).T).T + t_c2ab  # object points in arm base frame

        floor_mask = filtered_points[:, 2] < self.robot_min_height  # base footprint is -0.325
        filtered_points = filtered_points[~floor_mask]
        filtered_colors = filtered_colors[~floor_mask]

        place_x, place_y = np.median(np.unique(filtered_points[:, :2], axis=0), axis=0)
        x_margin, y_margin = 0.1, 0.1
        x_mask = np.logical_and(filtered_points[:, 0] > (place_x - x_margin), filtered_points[:, 0] < (place_x + x_margin))
        y_mask = np.logical_and(filtered_points[:, 1] > (place_y - y_margin), filtered_points[:, 1] < (place_y + y_margin))
        z_mask = np.logical_and(filtered_points[:, 2] > self.robot_min_height, filtered_points[:, 2] < self.robot_max_height)
        place_mask = np.logical_and(x_mask, y_mask, z_mask)
        place_z = np.quantile(filtered_points[place_mask][:, 2], 0.95) + 0.05
        
        place_point = np.array(
            [place_x, place_y, place_z],
            dtype=np.float32
        )
        
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        object_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        geometries = []
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.02, height=0.02)
        cylinder.translate(place_point)
        cylinder.paint_uniform_color([0, 1, 0])
        geometries.append(cylinder)

        visualize_cloud_geometries(
            object_pcd,
            geometries,
            save_file=self.save_dir / f"placing_{tries}.png",
            visualize=self.cfgs.debug and not self.cfgs.headless,
            show_coordinate_frame=False,
        )

        if self.cfgs.open_communication:
            data_msg = "Now you received the place pose, good luck."
            self.socket.send_data(
                [
                    place_point,
                    [0],
                    [0, 0, 0],
                    [],
                    data_msg,
                ]
            )
        return True


    def pickup(
        self,
        points: np.ndarray,
        seg_mask: np.ndarray,
        bbox: Bbox,
        tries: int,
        last_attempt: bool=False,
    ):
        colors = self.cam.colors
        c2ab = self.cam.c2ab  # camera in arm base pose
        R_c2ab = c2ab[:3, :3]
        t_c2ab = c2ab[:3,  3]
        points_z = points[:, :, 2]

        rotation_top_mat = np.array([
            [ 0.0,  0.0,  1.0],
            [ 0.0, -1.0,  0.0],
            [ 1.0,  0.0,  0.0],
        ], np.float32)

        # Filtering points based on the distance from camera
        mask = (
            (points_z > self.cfgs.min_depth)
            & (points_z < self.cfgs.max_depth)
            & ~np.isnan(points_z)
        )

        filtered_points = points[mask].astype(np.float32)
        filtered_colors = colors[mask].astype(np.float32)

        if self.cfgs.sampling_rate < 1:
            filtered_points, indices = sample_points(filtered_points, self.cfgs.sampling_rate)
            filtered_colors = filtered_colors[indices]


        # scene_points_file_name = self.save_dir / f"pointcloud_{tries}.png"
        grasp_project_file_name = self.save_dir / f"grasp_project_{tries}.png"
        grasp_all_file_name = self.save_dir / f"grasp_all_{tries}.png"
        grasp_best_file_name = self.save_dir / f"grasp_best_{tries}.png"

        # Get 3D bounding box from seg_mask (target object region)
        # Combine depth mask with segmentation mask
        object_mask = mask & seg_mask
        object_points = points[object_mask]
        # object_colors = colors[object_mask]

        render_scale = 1.5
        render_width, render_height = points.shape[1] * render_scale, points.shape[0] * render_scale
        render_zoom = 0.35
        render_point_size = 2.0 * render_scale

        # scene_pcd = o3d.geometry.PointCloud()
        # scene_pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3).astype(np.float64))
        # scene_pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3).astype(np.float64))
        # scene_pcd.transform(np.diag([1.0, -1.0, -1.0, 1.0]))

        # visualize_cloud_geometries(
        #     scene_pcd,
        #     [],
        #     visualize=self.cfgs.debug and not self.cfgs.headless,
        #     save_file=scene_points_file_name,
        #     width=render_width,
        #     height=render_height,
        #     zoom=render_zoom,
        #     point_size=render_point_size,
        #     background_color=[1.0, 1.0, 1.0],
        #     show_coordinate_frame=False,
        # )

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(object_points.astype(np.float64))
        # pcd.colors = o3d.utility.Vector3dVector(object_colors.astype(np.float64))
        # o3d.visualization.draw_geometries([pcd], window_name="Object Points")

        lims = [
            object_points[:, 0].min(), object_points[:, 0].max(),
            object_points[:, 1].min(), object_points[:, 1].max(),
            object_points[:, 2].min(), object_points[:, 2].max()
        ]

        gg, cloud = self.grasping_model.get_grasp(  # cloud is valid when debug is True
            filtered_points,
            filtered_colors,
            lims=lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True
        )

        object_points_ab = (R_c2ab @ object_points.T).T + t_c2ab  # a b: arm_base_link

        if gg is None or len(gg) == 0:
            if last_attempt:
                self.socket.send_data(
                    [
                        None,  # translation in arm base pose
                        None,  # rotation in arm base pose
                        [0, 0, 3],
                        object_points_ab,  # object in arm base pose
                        "Last try, just include object points",
                    ]
                )
            print("No Grasp detected after collision detection!")
            return False

        gg = gg.nms().sort_by_score()
        filter_gg = GraspGroup()
        
        W, H = self.cam.image.size
        min_score, max_score = 1, -10

        grasp_project_image = self.cam.image.copy().convert("RGB")
        grasp_project_draw = draw_rectangle(grasp_project_image, bbox)

        for g in gg:
            R_cam = g.rotation_matrix @ rotation_top_mat
            p_cam = g.translation  # grasp center
            ix, iy = (
                int(((p_cam[0] * self.cam.fx) / p_cam[2]) + self.cam.cx),
                int(((p_cam[1] * self.cam.fy) / p_cam[2]) + self.cam.cy),
            )
            ix = max(0, min(ix, W - 1))
            iy = max(0, min(iy, H - 1))
            
            R_base = R_c2ab @ R_cam
            p_base = (R_c2ab @ p_cam) + t_c2ab

            approach_dir = R_base[:, 2]

            base_to_target_xy = np.array([p_base[0], p_base[1], 0.0], dtype=np.float32)
            dir_xy = base_to_target_xy / np.linalg.norm(base_to_target_xy)  # arm base -> target

            # Project the gripper approach direction onto the plane spanned by dir_xy and the vertical axis.
            approach_in_dir_xy = np.dot(approach_dir, dir_xy) * dir_xy  # Component of approach_dir along dir_xy.
            approach_in_z = approach_dir[2] * np.array([0.0, 0.0, 1.0])  # Component of approach_dir along the z axis.
            approach = approach_in_dir_xy + approach_in_z
            approach /= float(np.linalg.norm(approach))

            penalty = 0.0
            cos_th = float(np.clip(np.dot(dir_xy, approach), -1.0, 1.0))
            if cos_th < 0.0:
                angle_th = np.arccos(cos_th)
                assert angle_th >= 0
                penalty = 0.2 * (angle_th - np.pi / 2) ** 2
                # print(angle_th, penalty, g.score, g.score - penalty)

            if not last_attempt:
                score = g.score - penalty
            else:
                score = g.score

            if not last_attempt:
                if seg_mask[iy, ix]:
                    grasp_project_draw.ellipse([(ix - 2, iy - 2), (ix + 2, iy + 2)], fill="green")
                    # filter_gg_obj.add(g)
                    if g.score >= 0.095:
                        g.score = score
                    min_score = min(min_score, g.score)
                    max_score = max(max_score, g.score)
                    filter_gg.add(g)
                else:
                    grasp_project_draw.ellipse([(ix - 1, iy - 1), (ix + 1, iy + 1)], fill="red")
            else:
                # filter_gg_obj.add(g)
                g.score = score
                filter_gg.add(g)

        if len(filter_gg) == 0:
            print(
                "No grasp poses detected for this object try to move the object a little and try again"
            )
            if last_attempt:
                self.socket.send_data(
                    [
                        None,  # translation in arm base pose
                        None,  # rotation in arm base pose
                        [0, 0, 3],
                        object_points_ab,  # object in arm base pose
                        "Last try, just include object points",
                    ]
                )
            return False
        
        filter_gg = filter_gg.nms().sort_by_score()
        grasp_project_image.save(grasp_project_file_name)
        print(f"Saved grasp projections at {grasp_project_file_name}") 
        if self.cfgs.debug and not self.cfgs.headless:
            grasp_project_image.show()

        if cloud is None:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(filtered_points.astype(np.float64))
            cloud.colors = o3d.utility.Vector3dVector(filtered_colors.astype(np.float64))

        trans_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        cloud_vis = copy.deepcopy(cloud).transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        filter_grippers = filter_gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        for gripper in filter_grippers:
            gripper.transform(trans_mat)

        visualize_cloud_geometries(
            cloud_vis,
            grippers,
            visualize=self.cfgs.debug and not self.cfgs.headless,
            save_file=grasp_all_file_name,
            width=render_width,
            height=render_height,
            zoom=render_zoom,
            point_size=render_point_size,
            background_color=[1.0, 1.0, 1.0],
            show_coordinate_frame=False,
        )
        visualize_cloud_geometries(
            copy.deepcopy(cloud_vis),
            [filter_grippers[0].paint_uniform_color([1.0, 0.0, 0.0])],
            visualize=self.cfgs.debug and not self.cfgs.headless,
            save_file=grasp_best_file_name,
            width=render_width,
            height=render_height,
            zoom=render_zoom,
            point_size=render_point_size,
            background_color=[1.0, 1.0, 1.0],
            show_coordinate_frame=False,
        )
        
        translation = (R_c2ab @ filter_gg[0].translation) + t_c2ab
        rotation = R_c2ab @ (filter_gg[0].rotation_matrix @ rotation_top_mat)
        if self.cfgs.open_communication:
            data_msg = "Now you received the gripper pose, good luck."
            self.socket.send_data(
                [
                    translation,  # translation in arm base pose
                    rotation,  # rotation in arm base pose
                    [filter_gg[0].depth, filter_gg[0].width, 0],
                    object_points_ab,  # object in arm base pose
                    data_msg,
                ]
            )
        return True


    def pick_back(
        self,
        points: np.ndarray,
        seg_mask: np.ndarray,
        tries: int,
    ):
        colors = self.cam.colors
        c2ab = self.cam.c2ab  # camera in arm base pose
        R_c2ab = c2ab[:3, :3]
        t_c2ab = c2ab[:3,  3]
        points_z = points[:, :, 2]

        # Filtering points based on the distance from camera
        mask = (
            (points_z > self.cfgs.min_depth)
            & (points_z < self.cfgs.max_depth)
            & ~np.isnan(points_z)
        )

        # Get 3D bounding box from seg_mask (target object region)
        # Combine depth mask with segmentation mask
        object_mask = mask & seg_mask
        object_points = points[object_mask]
        object_colors = colors[object_mask]
        object_points = (R_c2ab @ object_points.T).T + t_c2ab


        pick_x, pick_y = np.median(np.unique(object_points[:, :2], axis=0), axis=0)
        x_margin, y_margin = 0.05, 0.05
        x_mask = np.logical_and(object_points[:, 0] > (pick_x - x_margin), object_points[:, 0] < (pick_x + x_margin))
        y_mask = np.logical_and(object_points[:, 1] > (pick_y - y_margin), object_points[:, 1] < (pick_y + y_margin))
        z_mask = np.logical_and(object_points[:, 2] > self.robot_min_height, object_points[:, 2] < self.robot_max_height)
        pick_mask = np.logical_and(x_mask, y_mask, z_mask)
        pick_zs = object_points[pick_mask][:, 2]
        pick_z = np.quantile(pick_zs, 0.95)
        object_height = pick_zs.max() - pick_zs.min()
        pick_z -=  min(0.05, object_height * 2 / 3)  # 0.40 is the back backet min height, lower than this will collision with computer
        pick_z = max(pick_z, 0.40)
        pick_point = np.array(
            [pick_x, pick_y, pick_z],
            dtype=np.float32
        )
        
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(object_points)
        object_pcd.colors = o3d.utility.Vector3dVector(object_colors)
        geometries = []
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=0.01)
        cylinder.translate(pick_point)
        cylinder.paint_uniform_color([0, 1, 0])
        geometries.append(cylinder)

        visualize_cloud_geometries(
            object_pcd,
            geometries,
            save_file=self.save_dir / f"pick_back_{tries}.png",
            visualize=self.cfgs.debug and not self.cfgs.headless,
            show_coordinate_frame=False,
        )

        if self.cfgs.open_communication:
            data_msg = "Now you received the gripper (pick back) pose, good luck."
            self.socket.send_data(
                [
                    pick_point,
                    [0],
                    [0, 0, 0],
                    object_points,
                    data_msg,
                ]
            )
        return True
