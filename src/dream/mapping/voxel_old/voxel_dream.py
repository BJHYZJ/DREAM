# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import base64
import logging
import os
import pickle
import re
import skimage
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from PIL import Image
from scipy.ndimage import maximum_filter, median_filter
from torch import Tensor

from dream.core.interfaces import Observations
from dream.llms import OpenaiClient
from dream.llms.prompts import DYNAMEM_VISUAL_GROUNDING_PROMPT
from dream.utils.image import Camera, camera_xyz_to_global_xyz
from dream.utils.morphology import binary_dilation, binary_erosion, get_edges
from dream.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from dream.utils.voxel import VoxelizedPointcloud, scatter3d
from dream.utils.data_tools.dict import update
from dream.visualization.urdf_visualizer import URDFVisualizer
from dream.utils.visualization import create_disk
from dream.mapping.grid import GridParams
from dream.mapping.instance import Instance, InstanceMemory

from .voxel import VALID_FRAMES, Frame
# from .voxel import SparseVoxelMap


logger = logging.getLogger(__name__)


# class SparseVoxelMapDream(SparseVoxelMap):
class SparseVoxelMapDream:

    DEFAULT_INSTANCE_MAP_KWARGS = dict(
        du_scale=1,
        instance_association="bbox_iou",
        log_dir_overwrite_ok=True,
        mask_cropped_instances="False",
    )
    debug_valid_depth: bool = False
    debug_instance_memory_processing_time: bool = False

    def __init__(
        self,
        resolution: float = 0.01,
        semantic_memory_resolution: float = 0.05,
        feature_dim: int = 3,
        grid_size: Tuple[int, int] = None,
        grid_resolution: float = 0.05,
        obs_min_height: float = 0.1,
        obs_max_height: float = 1.8,
        obs_min_density: float = 10,
        smooth_kernel_size: int = 2,
        neg_obs_height: float = 0.0,
        add_local_radius_points: bool = True,
        remove_visited_from_obstacles: bool = False,
        local_radius: float = 0.8,
        min_depth: float = 0.25,
        max_depth: float = 2.5,
        pad_obstacles: int = 0,
        background_instance_label: int = -1,
        instance_memory_kwargs: Dict[str, Any] = {},
        voxel_kwargs: Dict[str, Any] = {},
        encoder=None,
        map_2d_device: str = "cpu",
        device: Optional[str] = None,
        use_instance_memory: bool = False,
        use_median_filter: bool = False,
        median_filter_size: int = 5,
        median_filter_max_error: float = 0.01,
        use_derivative_filter: bool = False,
        derivative_filter_threshold: float = 0.5,
        prune_detected_objects: bool = False,
        add_local_radius_every_step: bool = False,
        min_points_per_voxel: int = 10,
        use_negative_obstacles: bool = False,
        point_update_threshold: float = 0.9,
        detection=None,
        image_shape=(480, 360),
        log="test",
        mllm=False,
    ):
        # super().__init__(
        #     resolution=resolution,
        #     feature_dim=feature_dim,
        #     grid_size=grid_size,
        #     grid_resolution=grid_resolution,
        #     obs_min_height=obs_min_height,
        #     obs_max_height=obs_max_height,
        #     obs_min_density=obs_min_density,
        #     smooth_kernel_size=smooth_kernel_size,
        #     neg_obs_height=neg_obs_height,
        #     add_local_radius_points=add_local_radius_points,
        #     remove_visited_from_obstacles=remove_visited_from_obstacles,
        #     local_radius=local_radius,
        #     min_depth=min_depth,
        #     max_depth=max_depth,
        #     pad_obstacles=pad_obstacles,
        #     background_instance_label=background_instance_label,
        #     instance_memory_kwargs=instance_memory_kwargs,
        #     voxel_kwargs=voxel_kwargs,
        #     encoder=encoder,
        #     map_2d_device=map_2d_device,
        #     device=device,
        #     use_instance_memory=use_instance_memory,
        #     use_median_filter=use_median_filter,
        #     median_filter_size=median_filter_size,
        #     median_filter_max_error=median_filter_max_error,
        #     use_derivative_filter=use_derivative_filter,
        #     derivative_filter_threshold=derivative_filter_threshold,
        #     prune_detected_objects=prune_detected_objects,
        #     add_local_radius_every_step=add_local_radius_every_step,
        #     min_points_per_voxel=min_points_per_voxel,
        #     use_negative_obstacles=use_negative_obstacles,
        # )

        # TODO: We an use fastai.store_attr() to get rid of this boilerplate code
        self.feature_dim = feature_dim
        self.obs_min_height = obs_min_height
        self.obs_max_height = obs_max_height
        self.neg_obs_height = neg_obs_height
        self.obs_min_density = obs_min_density
        self.prune_detected_objects = prune_detected_objects

        # Smoothing kernel params
        self.smooth_kernel_size = smooth_kernel_size
        if self.smooth_kernel_size > 0:
            self.smooth_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(smooth_kernel_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.smooth_kernel = None

        # Median filter params
        self.median_filter_size = median_filter_size
        self.use_median_filter = use_median_filter
        self.median_filter_max_error = median_filter_max_error
        self.use_negative_obstacles = use_negative_obstacles

        # If we have an allowed radius to move in, we can store a mask with extra obstacles
        self.allowed_map: torch.Tensor = None

        # Derivative filter params
        self.use_derivative_filter = use_derivative_filter
        self.derivative_filter_threshold = derivative_filter_threshold

        self.grid_resolution = grid_resolution
        self.voxel_resolution = resolution
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pad_obstacles = int(pad_obstacles)
        self.background_instance_label = background_instance_label

        self.instance_memory_kwargs = update(
            copy.deepcopy(self.DEFAULT_INSTANCE_MAP_KWARGS), instance_memory_kwargs
        )
        self.use_instance_memory = use_instance_memory
        self.voxel_kwargs = voxel_kwargs
        self.encoder = encoder
        self.map_2d_device = map_2d_device
        self._min_points_per_voxel = min_points_per_voxel
        self.urdf_visualizer = URDFVisualizer()

        # Is the 2d map stale?
        self._stale_2d = True

        # Set the device we use for things here
        if device is not None:
            self.device = device
        else:
            self.device = self.map_2d_device

        # Create kernel(s) for obstacle dilation over 2d/3d maps
        if self.pad_obstacles > 0:
            self.dilate_obstacles_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(self.pad_obstacles))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_obstacles_kernel = None

        # Add points with local_radius to the voxel map at (0,0,0) unless we receive lidar points
        self._add_local_radius_points = add_local_radius_points
        self._add_local_radius_every_step = add_local_radius_every_step
        self._remove_visited_from_obstacles = remove_visited_from_obstacles
        self.local_radius = local_radius

        # Create disk for mapping explored areas near the robot - since camera can't always see it
        self._disk_size = np.ceil(self.local_radius / self.grid_resolution)

        self._visited_disk = torch.from_numpy(
            create_disk(self._disk_size, (2 * self._disk_size) + 1)
        ).to(map_2d_device)

        self.grid = GridParams(grid_size=grid_size, resolution=resolution, device=map_2d_device)
        self.grid_size = self.grid.grid_size
        self.grid_origin = self.grid.grid_origin
        self.resolution = self.grid.resolution

        self.point_update_threshold = point_update_threshold
        self._history_soft: Optional[Tensor] = None
        self.semantic_memory = VoxelizedPointcloud(voxel_size=semantic_memory_resolution).to(
            self.device
        )

        self.image_shape = image_shape
        self.obs_count = 0
        self.detection_model = detection
        self.log = log
        self.mllm = mllm
        if self.mllm:
            # Used to do visual grounding task
            self.gpt_client = OpenaiClient(
                DYNAMEM_VISUAL_GROUNDING_PROMPT, model="gpt-4o-2024-05-13"
            )

        # Init variables
        self.reset()


    def reset(self) -> None:
        """Clear out the entire voxel map."""
        self.observations: List[Frame] = []
        # Create an instance memory to associate bounding boxes in space
        if self.use_instance_memory:
            self.instances = InstanceMemory(
                num_envs=1,
                encoder=self.encoder,
                **self.instance_memory_kwargs,
            )
        else:
            self.instances = None

        # Create voxelized pointcloud
        self.voxel_pcd = VoxelizedPointcloud(
            voxel_size=self.voxel_resolution,
            dim_mins=None,
            dim_maxs=None,
            feature_pool_method="mean",
            **self.voxel_kwargs,
        )

        self._seq = 0
        self._2d_last_updated = -1
        # Create map here - just reset *some* variables
        self.reset_cache()

    def reset_cache(self):
        """Clear some tracked things"""
        # Stores points in 2d coords where robot has been
        self._visited = torch.zeros(self.grid_size, device=self.map_2d_device)

        # Store instances detected (all of them for now)
        if self.use_instance_memory:
            self.instances.reset()

        self.voxel_pcd.reset()

        # Store 2d map information
        # This is computed from our various point clouds
        self._map2d = None



    def find_alignment_over_model(self, queries: str):
        clip_text_tokens = self.encoder.encode_text(queries).cpu()
        points, features, weights, _ = self.semantic_memory.get_pointcloud()
        if points is None:
            return None
        features = F.normalize(features, p=2, dim=-1).cpu()
        point_alignments = clip_text_tokens.float() @ features.float().T

        # print(point_alignments.shape)
        return point_alignments

    def find_alignment_for_text(self, text: str):
        points, features, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text).cpu()
        return points[alignments.argmax(dim=-1)].detach().cpu()

    def find_obs_id_for_text(self, text: str):
        obs_counts = self.semantic_memory._obs_counts
        alignments = self.find_alignment_over_model(text).cpu()
        return obs_counts[alignments.argmax(dim=-1)].detach().cpu()

    def verify_point(
        self,
        text: str,
        point: Union[torch.Tensor, np.ndarray],
        distance_threshold: float = 0.1,
        similarity_threshold: float = 0.21,
    ):
        """
        Running visual grounding is quite time consuming.
        Thus, sometimes if the point has very high cosine similarity with text query, we might opt not to run visual grounding again.
        This function evaluates the cosine similarity.
        """
        if isinstance(point, np.ndarray):
            point = torch.from_numpy(point)
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        distances = torch.linalg.norm(point - points.detach().cpu(), dim=-1)
        if torch.min(distances) > distance_threshold:
            print("Points are so far from other points!")
            return False
        alignments = self.find_alignment_over_model(text).detach().cpu()[0]
        if torch.max(alignments[distances <= distance_threshold]) < similarity_threshold:
            print("Points close the the point are not similar to the text!")
        return torch.max(alignments[distances < distance_threshold]) >= similarity_threshold

    def get_2d_map(
        self, debug: bool = False, return_history_id: bool = False, kernel: int = 7
    ) -> Tuple[Tensor, ...]:
        """
        Get 2d map with explored area and frontiers.
        return_history_id: if True, return when each voxel was recently updated
        """

        # Is this already cached? If so we don't need to go to all this work
        if (
            self._map2d is not None
            and self._history_soft is not None
            and self._seq == self._2d_last_updated
        ):
            return self._map2d if not return_history_id else (*self._map2d, self._history_soft)

        # Convert metric measurements to discrete
        # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        xyz, _, counts, _ = self.voxel_pcd.get_pointcloud()
        # print(counts)
        # if xyz is not None:
        #     counts = torch.ones(xyz.shape[0])
        obs_ids = self.voxel_pcd._obs_counts
        if xyz is None:
            xyz = torch.zeros((0, 3))
            counts = torch.zeros((0))
            obs_ids = torch.zeros((0))

        device = xyz.device
        xyz = ((xyz / self.grid_resolution) + self.grid_origin + 0.5).long()
        # xyz[xyz[:, -1] < 0, -1] = 0

        # Crop to robot height
        # Use floor/ceil to ensure correct rounding for negative heights
        # int() truncates towards zero: int(-3.86)=-3, but we need floor(-3.86)=-4
        min_height = int(np.floor(self.obs_min_height / self.grid_resolution))
        max_height = int(np.ceil(self.obs_max_height / self.grid_resolution))
        # print(f'[DEBUG] obs_min_height={self.obs_min_height}, obs_max_height={self.obs_max_height}')
        # print(f'[DEBUG] min_height={min_height}, max_height={max_height}, height_bins={max_height - min_height}')

        # grid_size = self.grid_size + [max_height]
        # voxels = torch.zeros(grid_size, device=device)

        # # Mask out obstacles only above a certain height
        # obs_mask = xyz[:, -1] < max_height

        # Adjust grid size to accommodate negative heights
        # Height range is [min_height, max_height), so total height bins = max_height - min_height
        height_bins = max_height - min_height
        grid_size = self.grid_size + [height_bins]
        
        # Shift z coordinates so that min_height maps to 0
        if xyz.shape[0] > 0:
            xyz[:, -1] = xyz[:, -1] - min_height
        
        # Mask out obstacles: keep points in range [0, height_bins) after shifting
        obs_mask = (xyz[:, -1] >= 0) & (xyz[:, -1] < height_bins)
        # print(f'[DEBUG] Total points before filtering: {xyz.shape[0]}')
        # print(f'[DEBUG] Points after height filtering: {obs_mask.sum().item()}')
        xyz = xyz[obs_mask, :]
        counts = counts[obs_mask][:, None]
        # print(counts)
        obs_ids = obs_ids[obs_mask][:, None]
        
        # Handle negative obstacles if enabled
        # Points below neg_obs_height are treated as obstacles at a fixed height
        if self.use_negative_obstacles and xyz.shape[0] > 0:
            neg_height = int(np.floor(self.neg_obs_height / self.grid_resolution))
            neg_height_shifted = neg_height - min_height
            # Only process if neg_height is actually below min_height
            if neg_height < min_height:
                negative_obstacles = xyz[:, -1] < neg_height_shifted
                # Set negative obstacles to a specific height within valid range
                xyz[negative_obstacles, -1] = 0  # Set to first valid bin

        # voxels[x_coords, y_coords, z_coords] = 1
        voxels = scatter3d(xyz, counts, grid_size)
        history_ids = scatter3d(xyz, obs_ids, grid_size, "max")

        # Compute the obstacle voxel grid based on what we've seen
        # Since we shifted coordinates, the full z-dimension is already [0:height_bins]
        obstacles_soft = torch.sum(voxels, dim=-1)
        obstacles = obstacles_soft > self.obs_min_density
        # print(f'[DEBUG] obs_min_density={self.obs_min_density}')
        # print(f'[DEBUG] obstacles_soft max={obstacles_soft.max().item():.1f}, mean={obstacles_soft.mean().item():.1f}')
        # print(f'[DEBUG] Number of obstacle cells: {obstacles.sum().item()} / {obstacles.numel()}')

        history_soft = torch.max(history_ids, dim=-1).values
        history_soft = torch.from_numpy(maximum_filter(history_soft.float().numpy(), size=kernel))

        if self._remove_visited_from_obstacles:
            # Remove "visited" points containing observations of the robot
            obstacles *= (1 - self._visited).bool()

        if self.dilate_obstacles_kernel is not None:
            obstacles = binary_dilation(
                obstacles.float().unsqueeze(0).unsqueeze(0),
                self.dilate_obstacles_kernel,
            )[0, 0].bool()

        # Explored area = only floor mass
        # floor_voxels = voxels[:, :, :min_height]
        explored_soft = torch.sum(voxels, dim=-1)

        # Add explored radius around the robot, up to min depth
        explored = explored_soft > 0
        explored = (torch.zeros_like(explored) + self._visited).to(torch.bool) | explored

        if self.smooth_kernel_size > 0:
            # Opening and closing operations here on explore
            explored = binary_erosion(
                binary_dilation(explored.float().unsqueeze(0).unsqueeze(0), self.smooth_kernel),
                self.smooth_kernel,
            )
            explored = binary_dilation(
                binary_erosion(explored, self.smooth_kernel),
                self.smooth_kernel,
            )[0, 0].bool()
        if debug:
            import matplotlib.pyplot as plt

            plt.subplot(2, 2, 1)
            plt.imshow(obstacles_soft.detach().cpu().numpy())
            plt.title("obstacles soft")
            plt.axis("off")
            plt.subplot(2, 2, 2)
            plt.imshow(explored_soft.detach().cpu().numpy())
            plt.title("explored soft")
            plt.axis("off")
            plt.subplot(2, 2, 3)
            plt.imshow(obstacles.detach().cpu().numpy())
            plt.title("obstacles")
            plt.axis("off")
            plt.subplot(2, 2, 4)
            plt.imshow(explored.detach().cpu().numpy())
            plt.axis("off")
            plt.title("explored")
            plt.show()

        # Set the boundary in case the robot runs out from the environment
        obstacles[0:30, :] = True
        obstacles[-30:, :] = True
        obstacles[:, 0:30] = True
        obstacles[:, -30:] = True
        # Generate exploration heuristic to prevent robot from staying around the boundary
        if history_soft is not None:
            history_soft[0:35, :] = history_soft.max().item()
            history_soft[-35:, :] = history_soft.max().item()
            history_soft[:, 0:35] = history_soft.max().item()
            history_soft[:, -35:] = history_soft.max().item()

        # Update cache
        self._map2d = (obstacles, explored)
        self._2d_last_updated = self._seq
        self._history_soft = history_soft
        if not return_history_id:
            return obstacles, explored
        else:
            return obstacles, explored, history_soft

    def get_2d_alignment_heuristics(self, text: str, debug: bool = False):
        """
        Transform the similarity with text into a 2D value map that can be used to evaluate
        how much exploring to one point can benefit open vocabulary navigation
        """
        if self.semantic_memory._points is None:
            return None
        # Convert metric measurements to discrete
        # Gets the xyz correctly - for now everything is assumed to be within the correct distance of origin
        xyz, _, _, _ = self.semantic_memory.get_pointcloud()
        xyz = xyz.detach().cpu()
        if xyz is None:
            xyz = torch.zeros((0, 3))

        device = xyz.device
        xyz = ((xyz / self.grid_resolution) + self.grid_origin).long()
        xyz[xyz[:, -1] < 0, -1] = 0

        # Crop to robot height
        min_height = int(self.obs_min_height / self.grid_resolution)
        max_height = int(self.obs_max_height / self.grid_resolution)
        grid_size = self.grid_size + [max_height]

        # Mask out obstacles only above a certain height
        obs_mask = xyz[:, -1] < max_height
        xyz = xyz[obs_mask, :]
        alignments = self.find_alignment_over_model(text)[0].detach().cpu()
        alignments = alignments[obs_mask][:, None]

        alignment_heuristics = scatter3d(xyz, alignments, grid_size, "max")
        alignment_heuristics = torch.max(alignment_heuristics, dim=-1).values
        alignment_heuristics = torch.from_numpy(
            maximum_filter(alignment_heuristics.numpy(), size=5)
        )
        return alignment_heuristics

    def process_rgbd_images(
        self, rgb: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray, node_id: int
    ):
        """
        Process rgbd images for Dynamem
        """
        # Log input data
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        self.obs_count += 1

        cv2.imwrite(self.log + "/rgb" + str(self.obs_count) + ".jpg", rgb[:, :, [2, 1, 0]])
        np.save(self.log + "/rgb" + str(self.obs_count) + ".npy", rgb)
        np.save(self.log + "/depth" + str(self.obs_count) + ".npy", depth)
        np.save(self.log + "/intrinsics" + str(self.obs_count) + ".npy", intrinsics)
        np.save(self.log + "/pose" + str(self.obs_count) + ".npy", pose)

        # Update obstacle map
        self.voxel_pcd.clear_points(
            torch.from_numpy(depth), torch.from_numpy(intrinsics), torch.from_numpy(pose)
        )
        self.add(
            camera_pose=torch.Tensor(pose),
            rgb=torch.Tensor(rgb),
            depth=torch.Tensor(depth),
            camera_K=torch.Tensor(intrinsics),
            node_id=node_id
        )

        rgb, depth = torch.Tensor(rgb), torch.Tensor(depth)
        rgb = rgb.permute(2, 0, 1).to(torch.uint8)

        if self.image_shape is not None:
            h, w = self.image_shape
            h_image, w_image = depth.shape
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            intrinsics = np.copy(intrinsics)
            intrinsics[0, 0] *= w / w_image
            intrinsics[1, 1] *= h / h_image
            intrinsics[0, 2] *= w / w_image
            intrinsics[1, 2] *= h / h_image

        height, width = depth.squeeze().shape
        camera = Camera.from_K(np.array(intrinsics), width=width, height=height)
        camera_xyz = camera.depth_to_xyz(np.array(depth))
        world_xyz = torch.Tensor(camera_xyz_to_global_xyz(camera_xyz, np.array(pose)))

        median_depth = torch.from_numpy(median_filter(depth, size=5))
        median_filter_error = (depth - median_depth).abs()
        valid_depth = torch.logical_and(depth < self.max_depth, depth > self.min_depth)
        valid_depth = valid_depth & (median_filter_error < 0.01).bool()
        mask = ~valid_depth

        # Update semantic memory
        self.semantic_memory.clear_points(
            depth, torch.from_numpy(intrinsics), torch.from_numpy(pose), min_samples_clear=10
        )

        with torch.no_grad():
            rgb, features = self.encoder.run_mask_siglip(rgb, self.image_shape)  # type:ignore
            rgb, features = rgb.squeeze(), features.squeeze()

        valid_xyz = world_xyz[~mask]
        features = features[~mask]
        valid_rgb = rgb.permute(1, 2, 0)[~mask]
        if len(valid_xyz) != 0:
            self.add_to_semantic_memory(valid_xyz, features, valid_rgb)

    def add_to_semantic_memory(
        self,
        valid_xyz: Optional[torch.Tensor],
        feature: Optional[torch.Tensor],
        valid_rgb: Optional[torch.Tensor],
        weights: Optional[torch.Tensor] = None,
        threshold: float = 0.95,
    ):
        """
        Add pixel points into the semantic memory
        """
        # Adding all points to voxelizedPointCloud is useless and expensive, we should exclude threshold of all points
        selected_indices = torch.randperm(len(valid_xyz))[: int((1 - threshold) * len(valid_xyz))]
        if len(selected_indices) == 0:
            return
        if valid_xyz is not None:
            valid_xyz = valid_xyz[selected_indices]
        if feature is not None:
            feature = feature[selected_indices]
        if valid_rgb is not None:
            valid_rgb = valid_rgb[selected_indices]
        if weights is not None:
            weights = weights[selected_indices]

        valid_xyz = valid_xyz.to(self.device)
        if feature is not None:
            feature = feature.to(self.device)
        if valid_rgb is not None:
            valid_rgb = valid_rgb.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        self.semantic_memory.add(
            points=valid_xyz,
            features=feature,
            rgb=valid_rgb,
            weights=weights,
            obs_count=self.obs_count,
        )

    def localize_text(self, text, debug=True, return_debug=False):
        if self.mllm:
            return self.localize_with_mllm(text, debug=debug, return_debug=return_debug)
        else:
            return self.localize_with_feature_similarity(
                text, debug=debug, return_debug=return_debug
            )

    def find_all_images(
        self,
        text: str,
        min_similarity_threshold: Optional[float] = None,
        min_point_num: int = 100,
        max_img_num: Optional[int] = 3,
    ):
        """
        Select all images with high pixel similarity with text (by identifying whether points in this image are relevant objects)

        Args:
            min_similarity_threshold: Make sure every point with similarity greater than this value would be considered as the relevant objects
            min_point_num: Make sure we select at least these many points as relevant images.
            max_img_num: The maximum number of images we want to identify as relevant objects.
        """
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        points = points.cpu()
        alignments = self.find_alignment_over_model(text).cpu().squeeze()
        obs_counts = self.semantic_memory._obs_counts.cpu()

        turning_point = (
            min(min_similarity_threshold, alignments[torch.argsort(alignments)[-min_point_num]])
            if min_similarity_threshold is not None
            else alignments[torch.argsort(alignments)[-min_point_num]]
        )
        mask = alignments >= turning_point
        obs_counts = obs_counts[mask]
        alignments = alignments[mask]
        points = points[mask]

        unique_obs_counts, inverse_indices = torch.unique(obs_counts, return_inverse=True)

        points_with_max_alignment = torch.zeros((len(unique_obs_counts), points.size(1)))
        max_alignments = torch.zeros(len(unique_obs_counts))

        for i in range(len(unique_obs_counts)):
            # Get indices of elements belonging to the current cluster
            indices_in_cluster = (inverse_indices == i).nonzero(as_tuple=True)[0]
            if len(indices_in_cluster) <= 2:
                continue

            # Extract the alignments and points for the current cluster
            cluster_alignments = alignments[indices_in_cluster].squeeze()
            cluster_points = points[indices_in_cluster]

            # Find the point with the highest alignment in the cluster
            max_alignment_idx_in_cluster = cluster_alignments.argmax()
            point_with_max_alignment = cluster_points[max_alignment_idx_in_cluster]

            # Store the result
            points_with_max_alignment[i] = point_with_max_alignment
            max_alignments[i] = cluster_alignments.max()

        if max_img_num is not None:
            top_k = min(max_img_num, len(max_alignments))
        else:
            top_k = len(max_alignments)
        top_alignments, top_indices = torch.topk(
            max_alignments, k=top_k, dim=0, largest=True, sorted=True
        )
        top_points = points_with_max_alignment[top_indices]
        top_obs_counts = unique_obs_counts[top_indices]

        sorted_obs_counts, sorted_indices = torch.sort(top_obs_counts, descending=False)
        sorted_points = top_points[sorted_indices]
        top_alignments = top_alignments[sorted_indices]

        return sorted_obs_counts, sorted_points, top_alignments

    def llm_locator(self, image_ids: Union[torch.Tensor, np.ndarray, list], text: str):
        """
        Prompting the mLLM to select the images containing objects of interest.

        Input:
            image_ids: a series of images you want to send to mLLM
            text: text query

        Return
        """
        user_messages = []
        for obs_id in image_ids:
            obs_id = int(obs_id) - 1
            rgb = np.copy(self.observations[obs_id].rgb.numpy())
            depth = self.observations[obs_id].depth
            rgb[depth > 2.5] = [0, 0, 0]
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            user_messages.append(image)
        user_messages.append("The object you need to find is " + text)

        response = self.gpt_client(user_messages)
        return self.parse_localization_response(response)

    def parse_localization_response(self, response: str):
        """
        Parse the output of GPT4o to extract the selected image's id
        """
        try:
            # Use regex to locate the 'Images:' section, allowing for varying whitespace and line breaks
            images_section_match = re.search(r"Images:\s*([\s\S]+)", response, re.IGNORECASE)
            if not images_section_match:
                raise ValueError("The 'Images:' section is missing.")

            # Extract the content after 'Images:'
            images_content = images_section_match.group(1).strip()

            # Check if the content is 'None' (case-insensitive)
            if images_content.lower() == "none":
                return None

            # Use regex to find all numbers, regardless of separators like commas, periods, or spaces
            numbers = re.findall(r"\d+", images_content)

            if not numbers:
                raise ValueError("No numbers found in the 'Images:' section.")

            # Convert all found numbers to integers
            numbers = [int(num) for num in numbers]

            # Return all numbers as a list if multiple numbers are found
            if len(numbers) > 0:
                return numbers[-1]
            else:
                return None

        except Exception as e:
            # Handle any exceptions and optionally log the error message
            print(f"Error: {e}")
            return None

    def localize_with_mllm(self, text: str, debug=True, return_debug=False):
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text).cpu()
        point = points[alignments.argmax(dim=-1)].detach().cpu().squeeze()
        obs_counts = self.semantic_memory._obs_counts
        image_id = obs_counts[alignments.argmax(dim=-1)].detach().cpu()
        debug_text = ""
        target_point = None

        image_ids, points, alignments = self.find_all_images(
            # text, min_similarity_threshold=0.12, max_img_num=3
            text,
            max_img_num=3,
        )
        target_id = self.llm_locator(image_ids, text)

        if target_id is None:
            debug_text += "#### - Cannot verify whether this instance is the target. **😞** \n"
            image_id = None
            point = None
        else:
            target_id -= 1
            target_point = points[target_id]
            image_id = image_ids[target_id]
            point = points[target_id]
            debug_text += "#### - An image is identified \n"

        if image_id is not None:
            rgb = self.observations[image_id - 1].rgb
            pose = self.observations[image_id - 1].camera_pose
            depth = self.observations[image_id - 1].depth
            K = self.observations[image_id - 1].camera_K

            res = self.detection_model.compute_obj_coord(text, rgb, depth, K, pose)
            if res is not None:
                target_point = res
            else:
                target_point = point

        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, image_id, point

    def localize_with_feature_similarity(
        self, text, similarity_threshold: float = 0.14, debug=True, return_debug=False
    ):
        # points, _, _, _ = self.semantic_memory.get_pointcloud()
        # alignments = self.find_alignment_over_model(text).cpu()
        # point = points[alignments.argmax(dim=-1)].detach().cpu().squeeze()
        # obs_counts = self.semantic_memory._obs_counts
        # obs_id = obs_counts[alignments.argmax(dim=-1)].detach().cpu()
        obs_id, point, score, _res = self._get_best_observation_by_bbox_ratio(
            text=text, 
            similarity_threshold=similarity_threshold, 
            max_point_num=50, 
            max_img_num=3
        )

        # Lightweight selection: based on point cloud similarity and recency
        # obs_id, point = self._select_best_frame_lightweight(text, threshold=0.05)
        
        debug_text = ""
        target_point = None
        # TODO center points number 
        if _res is not None:
        # if obs_id is not None and obs_id <= len(self.observations):
            rgb = self.observations[obs_id - 1].rgb
            pose = self.observations[obs_id - 1].camera_pose
            depth = self.observations[obs_id - 1].depth
            K = self.observations[obs_id - 1].camera_K
            rgb = cv2.cvtColor(rgb.numpy(), cv2.COLOR_RGB2BGR)
            # double check if the detection is correct, use the point with highest confidence_threshold
            res = self.detection_model.compute_obj_coord(text, rgb, depth, K, pose)
            cv2.imwrite(self.log + "/rgb_" + text + "_" + str(obs_id.item() - 1) + ".png", rgb)
            # rgb_bgr = cv2.cvtColor(rgb.numpy(), cv2.COLOR_RGB2BGR)
            # res = self.detection_model.compute_obj_coord(text, rgb_bgr, depth, K, pose)
            # cv2.imwrite(self.log + "/rgb_" + text + "_" + str(obs_id - 1) + ".png", rgb_bgr)
        else:
            res = None
            
        if res is not None:
            target_point = res
            debug_text += (
                "#### - Object is detected in observations . **😃** Directly navigate to it.\n"
            )
        else:
            alignments = self.find_alignment_over_model(text).cpu()
            cosine_similarity_check = alignments.max().item() > 0.21
            if cosine_similarity_check:
                target_point = point

                debug_text += (
                    "#### - The point has high cosine similarity. **😃** Directly navigate to it.\n"
                )
            else:
                debug_text += "#### - Cannot verify whether this instance is the target. **😞** \n"
        print("--------------------------------")
        print(debug_text)
        if not debug:
            return target_point
        elif not return_debug:
            return target_point, debug_text
        else:
            return target_point, debug_text, obs_id, point

    def _select_best_frame_lightweight(self, text, threshold=0.05, 
                                        point_weight=0.6, recency_weight=0.4):
        """
        Lightweight frame selection: use find_all_images to get candidates,
        then rank by point count and recency
        
        Args:
            text: Query text
            similarity_threshold: Minimum similarity threshold
            point_weight: Weight for point count (default 0.6)
            recency_weight: Weight for recency (default 0.4)
            
        Returns:
            (obs_id, point) Selected frame ID and representative point
        """
        # Step 1: Get candidate frames from find_all_images
        candidate_obs_ids, candidate_points, candidate_similarities = self.find_all_images(
            text, 
            min_similarity_threshold=threshold,
            min_point_num=50,
            max_img_num=10  # Get more candidates for better selection
        )
        
        # Fallback: if no candidates found
        if len(candidate_obs_ids) == 0:
            points, _, _, _ = self.semantic_memory.get_pointcloud()
            alignments = self.find_alignment_over_model(text).cpu()
            obs_counts = self.semantic_memory._obs_counts
            max_idx = alignments.argmax(dim=-1)
            return obs_counts[max_idx].detach().cpu().item(), points[max_idx].detach().cpu().squeeze()
        
        # Step 2: Count points for each candidate frame
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text).cpu()
        obs_counts = self.semantic_memory._obs_counts.cpu()
        
        # Filter high similarity points
        mask = alignments.squeeze() >= threshold
        filtered_obs_counts = obs_counts[mask]
        filtered_points = points[mask]
        
        # Step 3: For each candidate frame, calculate score
        frame_scores = []
        max_obs_id = max(1, len(self.observations))
        
        for i, obs_id in enumerate(candidate_obs_ids):
            obs_id_val = obs_id.item() if hasattr(obs_id, 'item') else obs_id
            
            # Count how many high-similarity points are in this frame
            frame_point_mask = filtered_obs_counts == obs_id_val
            num_points = frame_point_mask.sum().item()
            
            # Normalize point count to [0, 1]
            point_score = min(1.0, num_points / 100.0)
            
            # Recency score: [0, 1], newer frames get higher scores
            recency_score = (obs_id_val - 1) / max(1, max_obs_id - 1)
            
            # Combined score: weighted by point count and recency
            combined_score = point_weight * point_score + recency_weight * recency_score
            
            frame_scores.append({
                'obs_id': obs_id_val,
                'point': candidate_points[i],
                'num_points': num_points,
                'recency': recency_score,
                'score': combined_score
            })
        
        # Step 4: Sort by combined score and select the best
        frame_scores.sort(key=lambda x: x['score'], reverse=True)
        best_frame = frame_scores[0]
        
        return best_frame['obs_id'], best_frame['point']

    def _get_best_observation_by_bbox_ratio(self, text, similarity_threshold, max_point_num=50, max_img_num=3):
        """
        Select best observation frame by bounding box ratio (lightweight approach)
        
        Args:
            text: Query text
            similarity_threshold: Similarity threshold
            
        Returns:
            (obs_id, point, bbox_ratio) Best observation frame information
        """
        # Get candidate frames
        candidate_obs_ids, candidate_points, candidate_similarities = self.find_all_images(
            text, 
            min_similarity_threshold=similarity_threshold,
            min_point_num=max_point_num,
            max_img_num=max_img_num
        )
        
        best_obs_id = None
        best_point = None
        best_score = 0.0
        best_res = None
        
        for i, obs_id in enumerate(candidate_obs_ids):
            if obs_id <= 0 or obs_id > len(self.observations):
                continue

            # Calculate bounding box ratio
            bbox_ratio, res = self._get_bbox_ratio(obs_id, text, confidence_threshold=0.05)
            if bbox_ratio is None:
                continue
            
            # Combined score: bbox ratio + recency bonus
            # Newer observations get a small bonus (0.01 per 10 frames)
            recency_bonus = obs_id * 0.01  # Small bonus for newer frames
            combined_score = bbox_ratio + recency_bonus
            
            if combined_score > best_score:
                best_obs_id = obs_id
                best_point = candidate_points[i]
                best_score = combined_score
                best_res = res

        return best_obs_id, best_point, best_score, best_res
    
    def _get_bbox_ratio(self, obs_idx, text, confidence_threshold=0.1, visualize=False):
        """
        Calculate the ratio of target object's bounding box to the whole image (lightweight)
        
        Args:
            obs_idx: Observation frame index
            text: Query text
            
        Returns:
            Ratio of bbox area to image area (0.0-1.0)
        """
        try:
            rgb = self.observations[obs_idx - 1].rgb
            depth = self.observations[obs_idx - 1].depth
            K = self.observations[obs_idx - 1].camera_K
            pose = self.observations[obs_idx - 1].camera_pose
            
            # Try to detect target object
            rgb_bgr = cv2.cvtColor(rgb.numpy(), cv2.COLOR_RGB2BGR)
            res, bbox = self.detection_model.compute_obj_coord(
                text=text, 
                rgb=rgb_bgr, 
                depth=depth, 
                camera_K=K, 
                camera_pose=pose, 
                confidence_threshold=confidence_threshold, 
                with_bbox=True
            )
            
            if res is not None:
                # Calculate bounding box area from tl_x, tl_y, br_x, br_y format
                img_area = rgb.shape[0] * rgb.shape[1]  # H * W
                bbox_width = bbox[2] - bbox[0]  # br_x - tl_x
                bbox_height = bbox[3] - bbox[1]  # br_y - tl_y
                bbox_area = bbox_width * bbox_height
                bbox_ratio = bbox_area / img_area
                
                # Visualization if requested
                if visualize:
                    self._save_bbox_visualization(obs_idx, text, bbox, res, bbox_ratio, rgb_bgr)
                
                return bbox_ratio, res
            
            return None, None
            
        except Exception as e:
            print(f"Bbox ratio calculation failed for obs {obs_idx}: {e}")
            return None

    def _save_bbox_visualization(self, obs_idx, text, bbox, res, bbox_ratio, rgb_bgr):
        """
        Save bbox visualization for debugging
        """
        try:
            vis_img = rgb_bgr.copy()
            
            if bbox is not None and res is not None:
                # Draw bounding box
                tl_x, tl_y, br_x, br_y = bbox
                tl_x, tl_y, br_x, br_y = (
                    int(max(0, tl_x.item())),
                    int(max(0, tl_y.item())),
                    int(min(vis_img.shape[1], br_x.item())),
                    int(min(vis_img.shape[0], br_y.item())),
                )
                
                # Draw rectangle
                cv2.rectangle(vis_img, (tl_x, tl_y), (br_x, br_y), (0, 255, 0), 2)
                
                # Add text information
                info_text = f"Bbox Ratio: {bbox_ratio:.3f}"
                cv2.putText(vis_img, info_text, (tl_x, tl_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add object coordinates
                coord_text = f"3D: ({res[0]:.2f}, {res[1]:.2f}, {res[2]:.2f})"
                cv2.putText(vis_img, coord_text, (tl_x, br_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                print(f"  Obs {obs_idx}: Detection found - Bbox ratio: {bbox_ratio:.3f}")
                
            else:
                # No detection found
                cv2.putText(vis_img, f"No detection for: {text}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"  Obs {obs_idx}: No detection found")
            
            # Save visualization
            save_path = f"{self.log}/bbox_debug_obs{obs_idx}_{text.replace(' ', '_')}.png"
            cv2.imwrite(save_path, vis_img)
            
        except Exception as e:
            print(f"Failed to save visualization for obs {obs_idx}: {e}")


    def add(
        self,
        camera_pose: Tensor,
        rgb: Tensor,
        node_id: int,
        xyz: Optional[Tensor] = None,
        camera_K: Optional[Tensor] = None,
        feats: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
        base_pose: Optional[Tensor] = None,
        xyz_frame: str = "camera",
        **info,
    ):
        """Add this to our history of observations. Also update the current running map.

        Parameters:
            camera_pose(Tensor): [4,4] cam_to_world matrix
            rgb(Tensor): N x 3 color points
            camera_K(Tensor): [3,3] camera instrinsics matrix -- usually pinhole model
            xyz(Tensor): N x 3 point cloud points in camera coordinates
            feats(Tensor): N x D point cloud features; D == 3 for RGB is most common
            base_pose(Tensor): optional location of robot base
        """
        # TODO: we should remove the xyz/feats maybe? just use observations as input?
        # TODO: switch to using just Obs struct?
        # Shape checking
        assert rgb.ndim == 3 or rgb.ndim == 2, f"{rgb.ndim=}: must be 2 or 3"
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb)
        if isinstance(camera_pose, np.ndarray):
            camera_pose = torch.from_numpy(camera_pose)
        if depth is not None:
            assert (
                rgb.shape[:-1] == depth.shape
            ), f"depth and rgb image sizes must match; got {rgb.shape=} {depth.shape=}"
        assert xyz is not None or (camera_K is not None and depth is not None)
        if xyz is not None:
            assert (
                xyz.shape[-1] == 3
            ), "xyz must have last dimension = 3 for x, y, z position of points"
            assert rgb.shape == xyz.shape, "rgb shape must match xyz"
            # Make sure shape is correct here for xyz and any passed-in features
            if feats is not None:
                assert (
                    feats.shape[-1] == self.feature_dim
                ), f"features must match voxel feature dimenstionality of {self.feature_dim}"
                assert xyz.shape[0] == feats.shape[0], "features must be available for each point"
            else:
                pass
            if isinstance(xyz, np.ndarray):
                xyz = torch.from_numpy(xyz)
        if depth is not None:
            assert depth.ndim == 2 or xyz_frame == "world"
        if camera_K is not None:
            assert camera_K.ndim == 2, "camera intrinsics K must be a 3x3 matrix"
        assert (
            camera_pose.ndim == 2 and camera_pose.shape[0] == 4 and camera_pose.shape[1] == 4
        ), "Camera pose must be a 4x4 matrix representing a pose in SE(3)"
        assert (
            xyz_frame in VALID_FRAMES
        ), f"frame {xyz_frame} was not valid; should one one of {VALID_FRAMES}"

        # Apply a median filter to remove bad depth values when mapping and exploring
        # This is not strictly necessary but the idea is to clean up bad pixels
        if depth is not None and self.use_median_filter:
            median_depth = torch.from_numpy(median_filter(depth, size=self.median_filter_size))
            median_filter_error = (depth - median_depth).abs()

        # Get full_world_xyz
        if xyz is not None:
            if xyz_frame == "camera":
                full_world_xyz = (
                    torch.cat([xyz, torch.ones_like(xyz[..., [0]])], dim=-1) @ camera_pose.T
                )[..., :3]
            elif xyz_frame == "world":
                full_world_xyz = xyz
            else:
                raise NotImplementedError(f"Unknown xyz_frame {xyz_frame}")
        else:
            full_world_xyz = unproject_masked_depth_to_xyz_coordinates(  # Batchable!
                depth=depth.unsqueeze(0).unsqueeze(1),
                pose=camera_pose.unsqueeze(0),
                inv_intrinsics=torch.linalg.inv(camera_K[:3, :3]).unsqueeze(0),
            )
        
        if False:
            color = rgb.reshape(-1, 3) / 255.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(full_world_xyz.detach().cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(color.detach().cpu().numpy())
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, coordinate_frame])

        # add observations before we start changing things
        self.observations.append(
            Frame(
                camera_pose,
                camera_K,
                xyz,
                rgb,
                feats,
                depth,
                node_id=node_id,
                instance=None,
                instance_classes=None,
                instance_scores=None,
                base_pose=base_pose,
                info=info,
                obs=None,
                full_world_xyz=full_world_xyz,
                xyz_frame=xyz_frame,
            )
        )

        valid_depth = torch.full_like(rgb[:, 0], fill_value=True, dtype=torch.bool)
        if depth is not None:
            valid_depth = (depth > self.min_depth) & (depth < self.max_depth)

            if self.use_derivative_filter:
                edges = get_edges(depth, threshold=self.derivative_filter_threshold)
                valid_depth = valid_depth & ~edges

            if self.use_median_filter:
                valid_depth = (
                    valid_depth & (median_filter_error < self.median_filter_max_error).bool()
                )

        # Add to voxel grid
        if feats is not None:
            feats = feats[valid_depth].reshape(-1, feats.shape[-1])
        rgb = rgb[valid_depth].reshape(-1, 3)
        world_xyz = full_world_xyz.view(-1, 3)[valid_depth.flatten()]

        # TODO: weights could also be confidence, inv distance from camera, etc
        if world_xyz.nelement() > 0:
            selected_indices = torch.randperm(len(world_xyz))[
                : int((1 - self.point_update_threshold) * len(world_xyz))
            ]
            if len(selected_indices) == 0:
                return
            if world_xyz is not None:
                world_xyz = world_xyz[selected_indices]
            if feats is not None:
                feats = feats[selected_indices]
            if rgb is not None:
                rgb = rgb[selected_indices]
            self.voxel_pcd.add(world_xyz, features=feats, rgb=rgb, weights=None)

        if self._add_local_radius_points:
            # TODO: just get this from camera_pose?
            self._update_visited(camera_pose[:3, 3].to(self.map_2d_device))
        if base_pose is not None:
            self._update_visited(base_pose.to(self.map_2d_device))

        # Increment sequence counter
        self._seq += 1

    def _update_visited(self, base_pose: Tensor):
        """Update 2d map of where robot has visited"""
        # Add exploration here
        # Base pose can be whatever, going to assume xyt for now
        map_xy = ((base_pose[:2] / self.grid_resolution) + self.grid_origin[:2]).int()
        x0 = int(map_xy[0] - self._disk_size)
        x1 = int(map_xy[0] + self._disk_size + 1)
        y0 = int(map_xy[1] - self._disk_size)
        y1 = int(map_xy[1] + self._disk_size + 1)
        assert x0 >= 0
        assert y0 >= 0
        self._visited[x0:x1, y0:y1] += self._visited_disk

    def xy_to_grid_coords(self, xy: np.ndarray) -> Optional[np.ndarray]:
        if not isinstance(xy, np.ndarray):
            xy = np.array(xy)
        return self.grid.xy_to_grid_coords(torch.Tensor(xy))

    def grid_coords_to_xy(self, grid_coords: np.ndarray) -> np.ndarray:
        if not isinstance(grid_coords, np.ndarray):
            grid_coords = np.array(grid_coords)
        return self.grid.grid_coords_to_xy(torch.Tensor(grid_coords))

    def grid_coords_to_xyt(self, grid_coords: np.ndarray) -> np.ndarray:
        if not isinstance(grid_coords, np.ndarray):
            grid_coords = np.array(grid_coords)
        return self.grid.grid_coords_to_xyt(torch.Tensor(grid_coords))

    def read_from_pickle(self, pickle_file_name, num_frames: int = -1):
        print("Reading from ", pickle_file_name)
        if isinstance(pickle_file_name, str):
            pickle_file_name = Path(pickle_file_name)
        assert pickle_file_name.exists(), f"No file found at {pickle_file_name}"
        with pickle_file_name.open("rb") as f:
            data = pickle.load(f)
        for i, (camera_pose, xyz, rgb, feats, depth, base_pose, K, world_xyz, node_id) in enumerate(
            zip(
                data["camera_poses"],
                data["xyz"],
                data["rgb"],
                data["feats"],
                data["depth"],
                data["base_poses"],
                data["camera_K"],
                data["world_xyz"],
                data['node_id']
            )
        ):
            # Handle the case where we dont actually want to load everything
            if num_frames > 0 and i >= num_frames:
                break

            camera_pose = self.fix_data_type(camera_pose)
            xyz = self.fix_data_type(xyz)
            rgb = self.fix_data_type(rgb)
            depth = self.fix_data_type(depth)
            intrinsics = self.fix_data_type(K)
            if feats is not None:
                feats = self.fix_data_type(feats)
            base_pose = self.fix_data_type(base_pose)
            self.voxel_pcd.clear_points(depth, intrinsics, camera_pose)
            self.add(
                camera_pose=camera_pose,
                xyz=xyz,
                rgb=rgb,
                feats=feats,
                depth=depth,
                base_pose=base_pose,
                camera_K=K,
                node_id=node_id,
            )

            self.obs_count += 1
        self.semantic_memory._points = data["combined_xyz"]
        self.semantic_memory._features = data["combined_feats"]
        self.semantic_memory._weights = data["combined_weights"]
        self.semantic_memory._rgb = data["combined_rgb"]
        self.semantic_memory._obs_counts = data["obs_id"]
        self.semantic_memory._mins = self.semantic_memory._points.min(dim=0).values
        self.semantic_memory._maxs = self.semantic_memory._points.max(dim=0).values
        self.semantic_memory.obs_count = max(self.semantic_memory._obs_counts).item()
        self.semantic_memory.obs_count = max(self.semantic_memory._obs_counts).item()

    def write_to_pickle(self, filename: Optional[str] = None) -> None:
        """Write out to a pickle file. This is a rough, quick-and-easy output for debugging, not intended to replace the scalable data writer in data_tools for bigger efforts.

        Args:
            filename (Optional[str], optional): Filename to write to. Defaults to None.
        """
        if not os.path.exists("debug"):
            os.mkdir("debug")
        if filename is None:
            filename = self.log + ".pkl"
        data: Dict[str, Any] = {}
        data["camera_poses"] = []
        data["camera_K"] = []
        data["base_poses"] = []
        data["xyz"] = []
        data["world_xyz"] = []
        data["rgb"] = []
        data["depth"] = []
        data["feats"] = []
        data['node_id'] = []
        for frame in self.observations:
            # add it to pickle
            # TODO: switch to using just Obs struct?
            data["camera_poses"].append(frame.camera_pose)
            data["base_poses"].append(frame.base_pose)
            data["camera_K"].append(frame.camera_K)
            data["xyz"].append(frame.xyz)
            data["world_xyz"].append(frame.full_world_xyz)
            data["rgb"].append(frame.rgb)
            data["depth"].append(frame.depth)
            data["feats"].append(frame.feats)
            data['node_id'].append(frame.node_id)
            for k, v in frame.info.items():
                if k not in data:
                    data[k] = []
                data[k].append(v)
        (
            data["combined_xyz"],
            data["combined_feats"],
            data["combined_weights"],
            data["combined_rgb"],
        ) = self.semantic_memory.get_pointcloud()
        data["obs_id"] = self.semantic_memory._obs_counts
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print("write all data to", filename)


    def get_outside_frontier(self, xyt, planner):
        """
        This function selects the edges of currently reachable space.
        """
        obstacles, _ = self.get_2d_map()
        if len(xyt) == 3:
            xyt = xyt[:2]
        reachable_points = planner.get_reachable_points(planner.to_pt(xyt))
        reachable_xs, reachable_ys = zip(*reachable_points)
        reachable_xs = torch.tensor(reachable_xs)
        reachable_ys = torch.tensor(reachable_ys)

        reachable_map = torch.zeros_like(obstacles)
        reachable_map[reachable_xs, reachable_ys] = 1
        reachable_map = reachable_map.to(torch.bool)
        edges = get_edges(reachable_map)
        return edges & ~reachable_map