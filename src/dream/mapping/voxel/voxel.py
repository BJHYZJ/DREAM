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
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from threading import Lock

import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import zstandard as zstd
from PIL import Image
from scipy.ndimage import maximum_filter, median_filter
from sklearn.decomposition import PCA
from torch import Tensor
from dataclasses import dataclass

from dream.core.interfaces import Observations
from dream.llms import OpenaiClient
from dream.llms.prompts import DYNAMEM_VISUAL_GROUNDING_PROMPT, DREAM_VISUAL_VERIFY_PROMPT
from dream.utils.image import Camera, camera_xyz_to_global_xyz
from dream.utils.morphology import binary_dilation, binary_erosion, get_edges
from dream.utils.point_cloud_torch import unproject_masked_depth_to_xyz_coordinates
from dream.mapping.voxel.voxel_util import VoxelizedPointcloud, scatter3d
from dream.utils.data_tools.dict import update
from dream.visualization.urdf_visualizer import URDFVisualizer
from dream.utils.visualization import create_disk
from dream.mapping.grid import GridParams
from dream.motion import Footprint, DreamIdx, PlanResult, RobotModel
from dream.perception.detection.owl import OwlPerception

@dataclass
class Frame:
    camera_pose: Any  # camera_in_map_pose
    base_pose: Any
    camera_K: Any
    rgb: Any
    depth: Any
    valid_depth: Any
    obs_id: Any
    feats: Any
    is_pose_graph_node: bool=False
    info: Any = None

logger = logging.getLogger(__name__)

class SparseVoxelMap:
    def __init__(
        self,
        voxel_resolution: float = 0.01,  # semantic memory resolution, for object localization
        feature_dim: int = 3,
        grid_size: Tuple[int, int] = None,
        grid_resolution: float = 0.05,  # occupancy grid map resolution, for path planning
        ground_max_height: float = -0.3,
        obs_min_height: float = -0.1,
        obs_max_height: float = 1.5,
        obs_min_density: float = 10,
        smooth_kernel_size: int = 2,
        neg_obs_height: float = 0.0,
        add_local_radius_points: bool = True,
        remove_visited_from_obstacles: bool = False,
        local_radius: float = 0.8,
        min_depth: float = 0.25,
        max_depth: float = 2.5,
        pad_obstacles: int = 0.2,  # meter to obstacle
        voxel_kwargs: Dict[str, Any] = {},
        encoder=None,
        map_2d_device: str = "cpu",
        device: Optional[str] = None,
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
        detection: Optional[OwlPerception]=None,
        image_shape=(360, 720),
        compression_features: bool=False,  # save memroy, but slowly
        log="test",
        mllm=False,
        with_mllm_verify=True,
        verify_point_similarity: float=0.21
    ):

        # TODO: We an use fastai.store_attr() to get rid of this boilerplate code
        self.feature_dim = feature_dim
        self.ground_max_height = ground_max_height
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
        self.voxel_resolution = voxel_resolution
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pad_obstacles = int(pad_obstacles / self.grid_resolution)

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

        self.grid = GridParams(grid_size=grid_size, resolution=self.grid_resolution, device=map_2d_device)
        self.grid_size = self.grid.grid_size
        self.grid_origin = self.grid.grid_origin

        self.point_update_threshold = point_update_threshold
        self._history_soft: Optional[Tensor] = None

        # voxelized pointcloud with semantic for localize target object and path planning
        self.semantic_memory = VoxelizedPointcloud(
            voxel_size=self.voxel_resolution,
            dim_mins=None,
            dim_maxs=None,
            feature_pool_method="mean",
            **self.voxel_kwargs,
        ).to(self.device)

        # Clear out the entire voxel map.
        self.observations: dict[int, Frame] = dict()

        self.image_shape = image_shape
        self.compression_features = compression_features
        if self.compression_features:
            self._zstd_level = 3
            self._zstd_compressor = zstd.ZstdCompressor(level=self._zstd_level)
            self._zstd_decompressor = zstd.ZstdDecompressor()

        self.detection_model = detection
        self.log = log
        # Log input data
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        self._seq = 0
        self._2d_last_updated = -1
        self.mllm = mllm
        self.with_mllm_verify = with_mllm_verify
        if self.mllm:
            # Used to do visual grounding task
            self.gpt_client = OpenaiClient(
                DYNAMEM_VISUAL_GROUNDING_PROMPT, model="gpt-4o-2024-05-13"
            )
        if not self.mllm and self.with_mllm_verify:
            # Used to verify
            self.gpt_verify_client = OpenaiClient(
                DREAM_VISUAL_VERIFY_PROMPT, model="gpt-5.1-2025-11-13"
            )
        self.verify_point_similarity = verify_point_similarity  # verify_point similarity threshold
        # Init variables
        # Create map here - just reset *some* variables
        self.reset()


    def reset(self):
        """Clear some tracked things"""
        # Stores points in 2d coords where robot has been
        self._visited = torch.zeros(self.grid_size, device=self.map_2d_device)

        self.semantic_memory.reset()

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
        return obs_counts[alignments.argmax(dim=-1)].detach().cpu().item()

    def verify_point(
        self,
        text: str,
        point: Union[torch.Tensor, np.ndarray],
        distance_threshold: float = 0.1,
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
        if torch.max(alignments[distances <= distance_threshold]) < self.verify_point_similarity:
            print("Points close the the point are not similar to the text!")
        return torch.max(alignments[distances < distance_threshold]) >= self.verify_point_similarity

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
        xyz, _, counts, _ = self.semantic_memory.get_pointcloud()
        # print(counts)
        # if xyz is not None:
        #     counts = torch.ones(xyz.shape[0])
        obs_ids = self.semantic_memory._obs_counts
        if xyz is None:
            xyz = torch.zeros((0, 3))
            counts = torch.zeros((0))
            obs_ids = torch.zeros((0))
        else:
            xyz = xyz.clone()  # normalized ground height
            xyz[:, 2] -= self.ground_max_height

        device = xyz.device
        xyz = ((xyz / self.grid_resolution) + self.grid_origin + 0.5).long()
        xyz[xyz[:, -1] < 0, -1] = 0

        # Crop to robot height
        min_height = int(self.obs_min_height / self.grid_resolution)
        max_height = int(self.obs_max_height / self.grid_resolution)
        # print('min_height', min_height, 'max_height', max_height)
        grid_size = self.grid_size + [max_height]
        voxels = torch.zeros(grid_size, device=device)

        # Mask out obstacles only above a certain height
        obs_mask = xyz[:, -1] < max_height
        xyz = xyz[obs_mask, :]
        counts = counts[obs_mask][:, None]
        # print(counts)
        obs_ids = obs_ids[obs_mask][:, None]

        # voxels[x_coords, y_coords, z_coords] = 1
        voxels = scatter3d(xyz, counts, grid_size)
        history_ids = scatter3d(xyz, obs_ids, grid_size, "max")

        # Compute the obstacle voxel grid based on what we've seen
        obstacle_voxels = voxels[:, :, min_height:max_height]
        obstacles_soft = torch.sum(obstacle_voxels, dim=-1)
        obstacles = obstacles_soft > self.obs_min_density

        history_ids = history_ids[:, :, min_height:max_height]
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
            import os
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

            os.makedirs(self.log, exist_ok=True)
            plt.savefig(os.path.join(self.log, f"map_debug_{int(self._seq)}.png"), dpi=150, bbox_inches="tight")
            plt.close("all")

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


    def feature_compression(
        self,
        feature: Optional[torch.Tensor],
    ) -> Optional[Dict[str, Any]]:
        """
        Compress an HWC feature map already on CPU using symmetric int8 quantization + zstd.
        """
        if feature is None:
            return None
        if not torch.is_tensor(feature):
            feature = torch.as_tensor(feature)
        x = feature.detach().to(torch.float32)
        if x.ndim != 3:
            raise ValueError("feature_compression expects an HWC tensor")

        chw = x.permute(2, 0, 1).contiguous()
        chw = torch.clamp(chw, -1.0, 1.0)
        quant_scale = 1.0 / 127.0
        q = torch.clamp((chw / quant_scale).round(), -127, 127).to(torch.int8)
        if q.device.type != "cpu":
            q = q.cpu()
        payload = self._zstd_compressor.compress(q.numpy().tobytes())

        return {
            "data": payload,
            "shape": tuple(q.shape),
            "layout": "chw",
            "scale": quant_scale,
            "codec": "zstd",
        }

    def feature_decompression(
        self,
        package: Any,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Decompress feature package produced by feature_compression.
        """
        if package is None:
            return None

        if torch.is_tensor(package):
            out = package
        elif isinstance(package, dict) and package.get("layout") == "chw":
            codec = package.get("codec", "zstd")
            shape = tuple(package["shape"])
            scale = float(package.get("scale", 1.0 / 127.0))

            if codec == "zstd":
                buffer = self._zstd_decompressor.decompress(package["data"])
            elif codec == "raw":
                buffer = package["data"]
            else:
                raise ValueError(f"Unknown codec '{codec}'")

            arr = np.frombuffer(buffer, dtype=np.int8).reshape(shape)
            chw = torch.from_numpy(arr.astype(np.int8)).to(torch.float32) * scale
            out = chw.permute(1, 2, 0).contiguous()
        else:
            raise ValueError("Unsupported feature package for decompression")

        if dtype is not None:
            out = out.to(dtype)
        if device is not None:
            out = out.to(device)
        return out
        

    def process_rgbd_images(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray, 
        intrinsics: np.ndarray, 
        camera_pose: np.ndarray, 
        base_pose: np.ndarray,
        obs_id: int,
        save_all_obs: bool=True,
        **info,
    ):
        """
        Process rgbd images for DREAM
        """
        if save_all_obs:
            cv2.imwrite(self.log + "/rgb" + str(obs_id) + ".jpg", rgb[:, :, [2, 1, 0]])
            np.save(self.log + "/rgb" + str(obs_id) + ".npy", rgb)
            np.save(self.log + "/depth" + str(obs_id) + ".npy", depth)
            np.save(self.log + "/intrinsics" + str(obs_id) + ".npy", intrinsics)
            np.save(self.log + "/camera_pose" + str(obs_id) + ".npy", camera_pose)

        rgb = torch.Tensor(rgb)
        depth = torch.Tensor(depth)
        intrinsics = torch.Tensor(intrinsics)
        camera_pose = torch.Tensor(camera_pose)
        base_pose = torch.Tensor(base_pose)

        # Resize depth/rgb and scale intrinsics once here
        if self.image_shape is not None:
            Ht, Wt = self.image_shape
            H0, W0 = depth.shape

            # depth: nearest
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0), size=(Ht, Wt), mode="nearest"
            ).squeeze(0).squeeze(0)

            # rgb: bilinear
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb_chw = rgb.permute(2, 0, 1).unsqueeze(0)
                rgb_chw = F.interpolate(rgb_chw, size=(Ht, Wt), mode="bilinear", align_corners=False)
                rgb = rgb_chw.squeeze(0).permute(1, 2, 0)

            # scale intrinsics
            scale_x = float(Wt) / float(W0)
            scale_y = float(Ht) / float(H0)
            intrinsics[0, 0] = intrinsics[0, 0] * scale_x
            intrinsics[1, 1] = intrinsics[1, 1] * scale_y
            intrinsics[0, 2] = intrinsics[0, 2] * scale_x
            intrinsics[1, 2] = intrinsics[1, 2] * scale_y

        # Precompute depth filters and masks once
        if self.use_median_filter:
            median_depth = torch.from_numpy(median_filter(depth, size=self.median_filter_size))
            median_filter_error = (depth - median_depth).abs()

        # Unified valid depth mask for both pipelines (complete filtering)
        valid_depth = (depth > self.min_depth) & (depth < self.max_depth)
        if self.use_derivative_filter:
            edges = get_edges(depth, threshold=self.derivative_filter_threshold)
            valid_depth = valid_depth & (~edges)
        if self.use_median_filter:
            valid_depth = valid_depth & (median_filter_error < self.median_filter_max_error).bool()


        feats = self.add_to_semantic_memory(
            camera_pose=camera_pose,
            base_pose=base_pose,
            rgb=rgb,
            depth=depth,
            obs_id=obs_id,
            valid_depth=valid_depth,
            camera_K=intrinsics,
            return_feats=True,
        )

        # Compressed features to reduce memory consumption
        if self.compression_features:
            compressed_feats = self.feature_compression(feature=feats)

        # add observations before we start changing things
        assert obs_id not in self.observations.keys(), "obs_id must be incremented."
        self.observations[obs_id] = Frame(
            camera_pose=camera_pose,
            base_pose=base_pose,
            camera_K=intrinsics,
            rgb=rgb,
            depth=depth,
            valid_depth=valid_depth,
            obs_id=obs_id,
            feats=compressed_feats if self.compression_features else feats,
            info=info,
        )

    def add_to_semantic_memory(
        self,
        camera_pose: Tensor,
        base_pose: Tensor,
        rgb: Tensor,
        obs_id: int,
        camera_K: Tensor,
        depth: Tensor,
        valid_depth: Optional[torch.Tensor]=None,
        weights: Optional[torch.Tensor]=None,
        feats: Optional[Tensor]=None,
        return_feats: bool=False,
        threshold: float = 0.95,
    ):
        """
        Add pixel points into the semantic memory
        """
        # Adding all points to voxelizedPointCloud is useless and expensive, we should exclude threshold of all points
        # Assume depth/rgb/camera_K are already resized and scaled in process_rgbd_images
       
        self.semantic_memory.clear_points(
            depth=depth, 
            intrinsics=camera_K, 
            camera_pose=camera_pose, 
            min_samples_clear=10,
            depth_in_view_max_distance=self.max_depth
        )
       
        rgb = rgb.permute(2, 0, 1).to(torch.uint8)
        height, width = depth.squeeze().shape
        camera = Camera.from_K(np.array(camera_K), width=width, height=height)
        camera_xyz = camera.depth_to_xyz(np.array(depth))
        world_xyz = torch.Tensor(camera_xyz_to_global_xyz(camera_xyz, np.array(camera_pose)))
        
        # Then add the new environment points to semantic memory voxelized pointcloud
        # Apply a median filter to remove bad depth values when mapping and exploring
        # This is not strictly necessary but the idea is to clean up bad pixels
        # Pre-filtering should be done in process_rgbd_images; fallback if not provided
        if valid_depth is None:
            if depth is not None and self.use_median_filter:
                median_depth = torch.from_numpy(median_filter(depth, size=self.median_filter_size))
                median_filter_error = (depth - median_depth).abs()
            valid_depth = torch.logical_and(depth < self.max_depth, depth > self.min_depth)
            if self.use_derivative_filter:
                edges = get_edges(depth, threshold=self.derivative_filter_threshold)
                valid_depth = valid_depth & (~edges)
            if self.use_median_filter:
                valid_depth = valid_depth & (median_filter_error < self.median_filter_max_error).bool()

        if feats is None:
            with torch.no_grad():
                _, feats = self.encoder.run_mask_siglip(rgb)  # type:ignore
                _, feats = rgb.squeeze(), feats.squeeze()

        valid_xyz = world_xyz[valid_depth]
        features = feats[valid_depth]
        valid_rgb = rgb.permute(1, 2, 0)[valid_depth]
        
        if world_xyz.nelement() != 0:
            selected_indices = torch.randperm(len(valid_xyz))[: int((1 - threshold) * len(valid_xyz))]
            if len(selected_indices) == 0:
                return
            if valid_xyz is not None:
                valid_xyz = valid_xyz[selected_indices]
            if features is not None:
                features = features[selected_indices]
            if valid_rgb is not None:
                valid_rgb = valid_rgb[selected_indices]
            if weights is not None:
                weights = weights[selected_indices]

            if valid_xyz is not None and valid_xyz.nelement() != 0:
                valid_xyz = valid_xyz.to(self.device)
            if features is not None and features.nelement() != 0:
                features = features.to(self.device)
            if valid_rgb is not None and valid_rgb.nelement() != 0:
                valid_rgb = valid_rgb.to(self.device)
            if weights is not None and weights.nelement() != 0:
                weights = weights.to(self.device)

            self.semantic_memory.add(
                points=valid_xyz,
                obs_id=obs_id,
                features=features,
                rgb=valid_rgb,
                weights=weights,
            )

        # Update visited map (prefer robot base; fallback to camera if desired)
        if self._add_local_radius_points:
            if base_pose is not None:
                self._update_visited(base_pose[:3, 3].to(self.map_2d_device))
            else:
                # Fallback: use camera position when base not available
                self._update_visited(camera_pose[:3, 3].to(self.map_2d_device))

        # Increment sequence counter
        self._seq += 1

        if return_feats:
            return feats


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

    def mllm_locator(self, image_ids: Union[torch.Tensor, np.ndarray, list], text: str):
        """
        Prompting the mLLM to select the images containing objects of interest.

        Input:
            image_ids: a series of images you want to send to mLLM
            text: text query

        Return
        """
        user_messages = []
        for obs_id in image_ids:
            obs_id = int(obs_id)
            rgb = np.copy(self.observations[obs_id].rgb.numpy())
            depth = self.observations[obs_id].depth
            rgb[depth > 2.5] = [0, 0, 0]
            image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            user_messages.append(image)
        user_messages.append("The object you need to find is " + text)

        response = self.gpt_client(user_messages)
        return self.parse_localization_response(response)

    def mllm_verify(self, obs_id: int, text: str):
        rgb = np.copy(self.observations[obs_id].rgb.numpy())
        depth = self.observations[obs_id].depth
        rgb[depth > self.max_depth] = [0, 0, 0]
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        user_messages = []
        user_messages.append(image)
        user_messages.append("The object you need to verify is " + text)
        response = self.gpt_verify_client(user_messages)
        return self.parse_verfy_response(response)

    def parse_verfy_response(self, response: str) -> Optional[bool]:
        """
        Parse the output of the verify prompt to a boolean.
        """
        try:
            match = re.search(r"(?im)^\s*Answer\s*:\s*(True|False)\s*$", response)
            if not match:
                match = re.search(r"\b(True|False)\b", response, re.IGNORECASE)

            if match:
                return match.group(1).lower() == "true"

            logger.warning("Verify response missing True/False: %s", response)
            return None
        except Exception as exc:
            logger.error("Failed to parse verify response '%s': %s", response, exc)
            return None

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
        target_id = self.mllm_locator(image_ids, text)

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
        self, text, similarity_threshold: float=0.05, debug=True, return_debug=False
    ):
        points, _, _, _ = self.semantic_memory.get_pointcloud()
        alignments = self.find_alignment_over_model(text).cpu()
        point = points[alignments.argmax(dim=-1)].squeeze()
        obs_counts = self.semantic_memory._obs_counts
        obs_id = obs_counts[alignments.argmax(dim=-1)].item()
        debug_text = ""
        target_point = None

        rgb = self.observations[obs_id].rgb
        pose = self.observations[obs_id].camera_pose
        depth = self.observations[obs_id].depth
        K = self.observations[obs_id].camera_K

        rgb_np = rgb.numpy()
        rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.log + "/rgb_" + text + "_" + str(obs_id) + ".png", rgb_bgr)

        res = self.detection_model.compute_obj_coord(
            text=text, 
            rgb=rgb_np, 
            depth=depth, 
            camera_K=K, 
            camera_pose=pose, 
            confidence_threshold=similarity_threshold,
            depth_threshold=self.max_depth
        )
            
        if res is not None:
            target_point = res
            debug_text += (
                "#### - Object is detected in observations . **😃** Directly navigate to it.\n"
            )
        else:
            # Fallback: use mLLM to verify the current frame if available
            if self.with_mllm_verify:
                mllm_verify = self.mllm_verify(obs_id, text)
                print(f"LLM Response: {mllm_verify}")
                if mllm_verify:
                    target_point = point
                    debug_text += (
                        "#### - mLLM verified target in this frame. **😃** Using nearest point.\n"
                    )

            if target_point is None:
                alignments = self.find_alignment_over_model(text).cpu()
                cosine_similarity_check = alignments.max().item() > self.verify_point_similarity
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


    def detect_text(self, text, obs_id, similarity_threshold=0.05):

        rgb = self.observations[obs_id].rgb
        pose = self.observations[obs_id].camera_pose
        depth = self.observations[obs_id].depth
        K = self.observations[obs_id].camera_K

        rgb_np = rgb.numpy()
        # rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(self.log + "/rgb_" + text + "_" + str(obs_id) + ".png", rgb_bgr)
        text_exist = False
        res = self.detection_model.compute_obj_coord(
            text=text, 
            rgb=rgb_np, 
            depth=depth, 
            camera_K=K, 
            camera_pose=pose, 
            confidence_threshold=similarity_threshold,
            depth_threshold=self.max_depth
        )

        if res is not None:
            text_exist = True
        else:
            # Fallback: use mLLM to verify the current frame if available
            if self.with_mllm_verify:
                mllm_verify = self.mllm_verify(obs_id, text)
                print(f"LLM Response: {mllm_verify}")
                if mllm_verify:
                    text_exist = True

            if text_exist is False:
                alignments = self.find_alignment_over_model(text).cpu()
                cosine_similarity_check = alignments.max().item() > self.verify_point_similarity
                if cosine_similarity_check:
                    text_exist = True

        return text_exist

    # def _select_best_frame_lightweight(self, text, threshold=0.05, 
    #                                     point_weight=0.6, recency_weight=0.4):
    #     """
    #     Lightweight frame selection: use find_all_images to get candidates,
    #     then rank by point count and recency
        
    #     Args:
    #         text: Query text
    #         similarity_threshold: Minimum similarity threshold
    #         point_weight: Weight for point count (default 0.6)
    #         recency_weight: Weight for recency (default 0.4)
            
    #     Returns:
    #         (obs_id, point) Selected frame ID and representative point
    #     """
    #     # Step 1: Get candidate frames from find_all_images
    #     candidate_obs_ids, candidate_points, candidate_similarities = self.find_all_images(
    #         text, 
    #         min_similarity_threshold=threshold,
    #         min_point_num=50,
    #         max_img_num=10  # Get more candidates for better selection
    #     )
        
    #     # Fallback: if no candidates found
    #     if len(candidate_obs_ids) == 0:
    #         points, _, _, _ = self.semantic_memory.get_pointcloud()
    #         alignments = self.find_alignment_over_model(text).cpu()
    #         obs_counts = self.semantic_memory._obs_counts
    #         max_idx = alignments.argmax(dim=-1)
    #         return obs_counts[max_idx].detach().cpu().item(), points[max_idx].detach().cpu().squeeze()
        
    #     # Step 2: Count points for each candidate frame
    #     points, _, _, _ = self.semantic_memory.get_pointcloud()
    #     alignments = self.find_alignment_over_model(text).cpu()
    #     obs_counts = self.semantic_memory._obs_counts.cpu()
        
    #     # Filter high similarity points
    #     mask = alignments.squeeze() >= threshold
    #     filtered_obs_counts = obs_counts[mask]
    #     filtered_points = points[mask]
        
    #     # Step 3: For each candidate frame, calculate score
    #     frame_scores = []
    #     max_obs_id = max(1, len(self.observations))
        
    #     for i, obs_id in enumerate(candidate_obs_ids):
    #         obs_id_val = obs_id.item() if hasattr(obs_id, 'item') else obs_id
            
    #         # Count how many high-similarity points are in this frame
    #         frame_point_mask = filtered_obs_counts == obs_id_val
    #         num_points = frame_point_mask.sum().item()
            
    #         # Normalize point count to [0, 1]
    #         point_score = min(1.0, num_points / 100.0)
            
    #         # Recency score: [0, 1], newer frames get higher scores
    #         recency_score = (obs_id_val - 1) / max(1, max_obs_id - 1)
            
    #         # Combined score: weighted by point count and recency
    #         combined_score = point_weight * point_score + recency_weight * recency_score
            
    #         frame_scores.append({
    #             'obs_id': obs_id_val,
    #             'point': candidate_points[i],
    #             'num_points': num_points,
    #             'recency': recency_score,
    #             'score': combined_score
    #         })
        
    #     # Step 4: Sort by combined score and select the best
    #     frame_scores.sort(key=lambda x: x['score'], reverse=True)
    #     best_frame = frame_scores[0]
        
    #     return best_frame['obs_id'], best_frame['point']

    # def _get_best_observation_by_bbox_ratio(self, text, similarity_threshold, max_point_num=50, max_img_num=3):
    #     """
    #     Select best observation frame by bounding box ratio (lightweight approach)
        
    #     Args:
    #         text: Query text
    #         similarity_threshold: Similarity threshold
            
    #     Returns:
    #         (obs_id, point, bbox_ratio) Best observation frame information
    #     """
    #     # Get candidate frames
    #     candidate_obs_ids, candidate_points, candidate_similarities = self.find_all_images(
    #         text, 
    #         min_similarity_threshold=similarity_threshold,
    #         min_point_num=max_point_num,
    #         max_img_num=max_img_num
    #     )
        
    #     best_obs_id = None
    #     best_point = None
    #     best_score = 0.0
    #     best_res = None
        
    #     for i, obs_id in enumerate(candidate_obs_ids):
    #         if obs_id <= 0 or obs_id > len(self.observations):
    #             continue

    #         # Calculate bounding box ratio
    #         bbox_ratio, res = self._get_bbox_ratio(obs_id, text, confidence_threshold=0.05)
    #         if bbox_ratio is None:
    #             continue
            
    #         # Combined score: bbox ratio + recency bonus
    #         # Newer observations get a small bonus (0.01 per 10 frames)
    #         recency_bonus = obs_id * 0.01  # Small bonus for newer frames
    #         combined_score = bbox_ratio + recency_bonus
            
    #         if combined_score > best_score:
    #             best_obs_id = obs_id
    #             best_point = candidate_points[i]
    #             best_score = combined_score
    #             best_res = res

    #     return best_obs_id, best_point, best_score, best_res
    
    # def _get_bbox_ratio(self, obs_idx, text, confidence_threshold=0.1, visualize=False):
    #     """
    #     Calculate the ratio of target object's bounding box to the whole image (lightweight)
        
    #     Args:
    #         obs_idx: Observation frame index
    #         text: Query text
            
    #     Returns:
    #         Ratio of bbox area to image area (0.0-1.0)
    #     """
    #     try:
    #         rgb = self.observations[obs_idx - 1].rgb
    #         depth = self.observations[obs_idx - 1].depth
    #         K = self.observations[obs_idx - 1].camera_K
    #         pose = self.observations[obs_idx - 1].camera_pose
            
    #         # Try to detect target object
    #         rgb_bgr = cv2.cvtColor(rgb.numpy(), cv2.COLOR_RGB2BGR)
    #         res, bbox = self.detection_model.compute_obj_coord(
    #             text=text, 
    #             rgb=rgb_bgr, 
    #             depth=depth, 
    #             camera_K=K, 
    #             camera_pose=pose, 
    #             confidence_threshold=confidence_threshold, 
    #             with_bbox=True
    #         )
            
    #         if res is not None:
    #             # Calculate bounding box area from tl_x, tl_y, br_x, br_y format
    #             img_area = rgb.shape[0] * rgb.shape[1]  # H * W
    #             bbox_width = bbox[2] - bbox[0]  # br_x - tl_x
    #             bbox_height = bbox[3] - bbox[1]  # br_y - tl_y
    #             bbox_area = bbox_width * bbox_height
    #             bbox_ratio = bbox_area / img_area
                
    #             # Visualization if requested
    #             if visualize:
    #                 self._save_bbox_visualization(obs_idx, text, bbox, res, bbox_ratio, rgb_bgr)
                
    #             return bbox_ratio, res
            
    #         return None, None
            
    #     except Exception as e:
    #         print(f"Bbox ratio calculation failed for obs {obs_idx}: {e}")
    #         return None

    # def _save_bbox_visualization(self, obs_idx, text, bbox, res, bbox_ratio, rgb_bgr):
    #     """
    #     Save bbox visualization for debugging
    #     """
    #     try:
    #         vis_img = rgb_bgr.copy()
            
    #         if bbox is not None and res is not None:
    #             # Draw bounding box
    #             tl_x, tl_y, br_x, br_y = bbox
    #             tl_x, tl_y, br_x, br_y = (
    #                 int(max(0, tl_x.item())),
    #                 int(max(0, tl_y.item())),
    #                 int(min(vis_img.shape[1], br_x.item())),
    #                 int(min(vis_img.shape[0], br_y.item())),
    #             )
                
    #             # Draw rectangle
    #             cv2.rectangle(vis_img, (tl_x, tl_y), (br_x, br_y), (0, 255, 0), 2)
                
    #             # Add text information
    #             info_text = f"Bbox Ratio: {bbox_ratio:.3f}"
    #             cv2.putText(vis_img, info_text, (tl_x, tl_y - 10), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    #             # Add object coordinates
    #             coord_text = f"3D: ({res[0]:.2f}, {res[1]:.2f}, {res[2]:.2f})"
    #             cv2.putText(vis_img, coord_text, (tl_x, br_y + 20), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
    #             print(f"  Obs {obs_idx}: Detection found - Bbox ratio: {bbox_ratio:.3f}")
                
    #         else:
    #             # No detection found
    #             cv2.putText(vis_img, f"No detection for: {text}", (10, 30), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #             print(f"  Obs {obs_idx}: No detection found")
            
    #         # Save visualization
    #         save_path = f"{self.log}/bbox_debug_obs{obs_idx}_{text.replace(' ', '_')}.png"
    #         cv2.imwrite(save_path, vis_img)
            
    #     except Exception as e:
    #         print(f"Failed to save visualization for obs {obs_idx}: {e}")

    def xyt_is_safe(self, xyt: np.ndarray, robot: Optional[RobotModel] = None) -> bool:
        """Check to see if a given xyt position is known to be safe."""
        if robot is not None:
            raise NotImplementedError("not currently checking against robot base geometry")
        obstacles, explored = self.get_2d_map()
        # Convert xy to grid coords
        grid_xy = self.grid.xy_to_grid_coords(xyt[:2])
        # Check to see if grid coords are explored and obstacle free
        if grid_xy is None:
            # Conversion failed - probably out of bounds
            return False
        obstacles, explored = self.get_2d_map()
        # Convert xy to grid coords
        grid_xy = self.grid.xy_to_grid_coords(xyt[:2])
        # Check to see if grid coords are explored and obstacle free
        if grid_xy is None:
            # Conversion failed - probably out of bounds
            return False
        if robot is not None:
            # TODO: check against robot geometry
            raise NotImplementedError("not currently checking against robot base geometry")
        return True


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
        for i, (camera_pose, rgb, feats, depth, K, obs_id) in enumerate(
            zip(
                data["camera_poses"],
                data["rgb"],
                data["depth"],
                data["camera_K"],
                data['obs_id']
            )
        ):
            # Handle the case where we dont actually want to load everything
            if num_frames > 0 and i >= num_frames:
                break

            # camera_pose = self.fix_data_type(camera_pose)
            # rgb = self.fix_data_type(rgb)
            # depth = self.fix_data_type(depth)
            # intrinsics = self.fix_data_type(K)
            # if feats is not None:
            #     feats = self.fix_data_type(feats)
            # base_pose = self.fix_data_type(base_pose)
            # TODO

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
        data["rgb"] = []
        data["depth"] = []
        data['obs_id'] = []
        for obs_id, obs in self.observations.items():
            # add it to pickle
            # TODO: switch to using just Obs struct?
            data["camera_poses"].append(obs.camera_pose)
            data["base_poses"].append(obs.base_pose)
            data["camera_K"].append(obs.camera_K)
            data["rgb"].append(obs.rgb)
            data["depth"].append(obs.depth)
            data['obs_id'].append(obs.obs_id)
            for k, v in obs.info.items():
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


class SparseVoxelMapProxy:
    def __init__(self, voxel_map: SparseVoxelMap, lock: Lock):
        self._voxel_map = voxel_map
        self._lock = lock

    def __getattr__(self, name):
        def locked_method(*args, **kwargs):
            with self._lock:  # Acquire read lock for external access
                method = getattr(self._voxel_map, name)
                return method(*args, **kwargs)

        if callable(getattr(self._voxel_map, name)):
            return locked_method
        else:
            with self._lock:
                return getattr(self._voxel_map, name)
