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

import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import skimage
import skimage.morphology
from typing import Dict, List, Optional, Tuple, Union
from collections import deque

from dream.motion import Footprint
from dream.utils.morphology import binary_dilation, get_edges
from dream.core.robot import RobotModel
from dream.mapping.grid import GridParams
from dream.mapping.voxel import SparseVoxelMap, SparseVoxelMapProxy
from dream.motion.algo.a_star import AStar


class SparseVoxelMapNavigationSpace:

    # Used for making sure we do not divide by zero anywhere
    tolerance: float = 1e-8

    def __init__(
        self,
        robot: RobotModel,
        voxel_map: Union[SparseVoxelMap, SparseVoxelMapProxy],
        grid: Optional[GridParams] = None,
        step_size: float = 0.1,
        rotation_step_size: float = 0.5,
        use_orientation: bool = False,
        orientation_resolution: int = 64,
        dilate_frontier_size: int = 12,
        dilate_obstacle_size: int = 2,
        extend_mode: str = "separate",
        min_frontier_distance: Optional[float] = None,
    ):

        self.robot = robot
        self.step_size = step_size
        self.rotation_step_size = rotation_step_size
        self.voxel_map = voxel_map
        self.create_collision_masks(orientation_resolution)
        self.extend_mode = extend_mode
        if grid is None:
            grid = self.voxel_map.grid
        self.grid = grid

        # Create a stack for storing states to sample
        self._stack: deque[np.ndarray] = deque()

        # Always use 3d states
        self.use_orientation = use_orientation
        if self.use_orientation:
            self.dof = 3
        else:
            self.dof = 2

        self._kernels: Dict[int, torch.nn.Parameter] = {}

        if dilate_frontier_size > 0:
            self.dilate_explored_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(dilate_frontier_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_explored_kernel = None
        if dilate_obstacle_size > 0:
            self.dilate_obstacles_kernel = torch.nn.Parameter(
                torch.from_numpy(skimage.morphology.disk(dilate_obstacle_size))
                .unsqueeze(0)
                .unsqueeze(0)
                .float(),
                requires_grad=False,
            )
        else:
            self.dilate_obstacles_kernel = None

        self.traj = None
        self._min_frontier_distance = (
            min_frontier_distance if min_frontier_distance is not None else 0.85
        )  # metres

    def create_collision_masks(self, orientation_resolution: int):
        """Create a set of orientation masks

        Args:
            orientation_resolution: number of bins to break it into
        """
        self._footprint: Footprint = self.robot.get_robot_model().get_footprint()
        self._orientation_resolution = 64
        self._oriented_masks = []

        for i in range(orientation_resolution):
            theta = i * 2 * np.pi / orientation_resolution
            mask = self._footprint.get_rotated_mask(
                self.voxel_map.grid_resolution, angle_radians=theta
            )
            self._oriented_masks.append(mask)

    def compute_theta(self, cur_x, cur_y, end_x, end_y):
        theta = 0
        if end_x == cur_x and end_y >= cur_y:
            theta = np.pi / 2
        elif end_x == cur_x and end_y < cur_y:
            theta = -np.pi / 2
        else:
            theta = np.arctan((end_y - cur_y) / (end_x - cur_x))
            if end_x < cur_x:
                theta = theta + np.pi
            if theta > np.pi:
                theta = theta - 2 * np.pi
            if theta < -np.pi:
                theta = theta + 2 * np.pi
        return theta

    def debug_robot_position(self, start: torch.Tensor, planner: AStar, debug: bool=False):
        """debug robot current position"""
        obstacles, explored = self.voxel_map.get_2d_map()

        start_pt = planner.to_pt(start)
        
        crop_size = 50
        x0 = max(0, start_pt[0] - crop_size//2)
        x1 = min(obstacles.shape[0], start_pt[0] + crop_size//2)
        y0 = max(0, start_pt[1] - crop_size//2)
        y1 = min(obstacles.shape[1], start_pt[1] + crop_size//2)
        
        crop_obs = obstacles[x0:x1, y0:y1]
        crop_exp = explored[x0:x1, y0:y1]
        
        robot_x_rel = start_pt[0] - x0
        robot_y_rel = start_pt[1] - y0
        
        mask = self.get_oriented_mask(start[2])
        dim = mask.shape[0]
        half_dim = dim // 2
        
        footprint_contour = np.zeros_like(crop_obs.cpu().numpy())
        
        for i in range(crop_size):
            for j in range(crop_size):
                global_x = x0 + i
                global_y = y0 + j
                rel_x = global_x - start_pt[0]
                rel_y = global_y - start_pt[1]
                if abs(rel_x) <= half_dim and abs(rel_y) <= half_dim:
                    mask_x = int(rel_x + half_dim)
                    mask_y = int(rel_y + half_dim)
                    if 0 <= mask_x < dim and 0 <= mask_y < dim:
                        if mask.cpu().numpy()[mask_x, mask_y] > 0:
                            footprint_contour[i, j] = 1
        

        plt.figure(figsize=(15, 10))
        # Sub-image 1: Obstacle map + robot footprint
        plt.subplot(221)
        plt.imshow(crop_obs.cpu().numpy(), cmap='Reds', alpha=0.7)
        plt.contour(footprint_contour, levels=[0.5], colors='green', linewidths=3, alpha=0.8)
        plt.scatter(robot_y_rel, robot_x_rel, c='blue', s=150, marker='o', label='Robot Center', edgecolors='black', linewidth=2)
        plt.title(f"Obstacles + Robot Footprint\nRed=obstacles, Green=robot outline, Blue=center")
        plt.legend()
        
        # Sub-image 2: Exploring the map + robot footprint
        plt.subplot(222)
        plt.imshow(crop_exp.cpu().numpy(), cmap='Greens', alpha=0.7)
        plt.contour(footprint_contour, levels=[0.5], colors='blue', linewidths=3, alpha=0.8)
        plt.scatter(robot_y_rel, robot_x_rel, c='red', s=150, marker='o', label='Robot Center', edgecolors='black', linewidth=2)
        plt.title(f"Explored + Robot Footprint\nGreen=explored, Blue=robot outline, Red=center")
        plt.legend()
        
        # Sub-map 3: Obstacle map (pure obstacles)
        plt.subplot(223)
        plt.imshow(crop_obs.cpu().numpy(), cmap='Reds')
        plt.scatter(robot_y_rel, robot_x_rel, c='blue', s=150, marker='o', label='Robot Center', edgecolors='black', linewidth=2)
        plt.title(f"Obstacles Only\nRed=obstacles, Blue=robot center")
        plt.legend()
        
        # Sub-map 4: Exploration Map (Pure Exploration)
        plt.subplot(224)
        plt.imshow(crop_exp.cpu().numpy(), cmap='Greens')
        plt.scatter(robot_y_rel, robot_x_rel, c='red', s=150, marker='o', label='Robot Center', edgecolors='black', linewidth=2)
        plt.title(f"Explored Only\nGreen=explored, Red=robot center")
        plt.legend()
        
        plt.suptitle(f"Robot Position Analysis (cropped {crop_size}x{crop_size})\nRobot at grid ({start_pt[0]}, {start_pt[1]})", fontsize=14)
        plt.tight_layout()
        plt.show()


    def _get_theta_index(self, theta: float) -> int:
        """gets the index associated with theta here"""
        if theta < 0:
            theta += 2 * np.pi
        if theta >= 2 * np.pi:
            theta -= 2 * np.pi
        assert theta >= 0 and theta <= 2 * np.pi, "only angles between 0 and 2*PI allowed"
        theta_idx = np.round((theta / (2 * np.pi) * self._orientation_resolution) - 0.5)
        if theta_idx == self._orientation_resolution:
            theta_idx = 0
        return int(theta_idx)

    def get_oriented_mask(self, theta: float) -> torch.Tensor:
        theta_idx = self._get_theta_index(theta)
        return self._oriented_masks[theta_idx]

    def is_valid(
        self,
        state: torch.Tensor,
        is_safe_threshold=1.0,
        debug: bool = False,
        verbose: bool = False,
        obstacles: Optional[torch.Tensor] = None,
        explored: Optional[torch.Tensor] = None,
    ) -> bool:
        """Check to see if state is valid; i.e. if there's any collisions if mask is at right place"""
        assert len(state) == 3
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        ok = self.voxel_map.xyt_is_safe(state[:2])
        if not ok:
            # This was
            print("XYT state is not safe")
            return False

        # Now sample mask at this location
        mask = self.get_oriented_mask(state[-1])
        assert mask.shape[0] == mask.shape[1], "square masks only for now"
        dim = mask.shape[0]
        half_dim = dim // 2
        grid_xy = self.voxel_map.grid.xy_to_grid_coords(state[:2])
        x0 = int(grid_xy[0]) - half_dim
        x1 = x0 + dim
        y0 = int(grid_xy[1]) - half_dim
        y1 = y0 + dim

        if obstacles is None:
            attempt = 0
            max_attempts = 10
            while True:
                try:
                    obstacles, explored = self.voxel_map.get_2d_map()
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > max_attempts:
                        raise e
                    print(f"Error getting 2d map: {e}. Retrying...")
                    time.sleep(0.2)

        crop_obs = obstacles[x0:x1, y0:y1]
        crop_exp = explored[x0:x1, y0:y1]
        assert mask.shape == crop_obs.shape
        assert mask.shape == crop_exp.shape

        collision = torch.any(crop_obs & mask)

        p_is_safe = (torch.sum((crop_exp & mask) | ~mask) / (mask.shape[0] * mask.shape[1])).item()
        is_safe = p_is_safe >= is_safe_threshold
        if verbose:
            print(f"{collision=}, {is_safe=}, {p_is_safe=}, {is_safe_threshold=}")

        valid = bool((not collision) and is_safe)
        if debug:
            if collision:
                print("- state in collision")
            if not is_safe:
                print("- not safe")

            print(f"{valid=}")
            obs = obstacles.cpu().numpy().copy()
            exp = explored.cpu().numpy().copy()
            obs[x0:x1, y0:y1] = 1
            plt.subplot(321)
            plt.imshow(obs)
            plt.subplot(322)
            plt.imshow(exp)
            plt.subplot(323)
            plt.imshow(crop_obs.cpu().numpy())
            plt.title("obstacles")
            plt.subplot(324)
            plt.imshow(crop_exp.cpu().numpy())
            plt.title("explored")
            plt.subplot(325)
            plt.imshow(mask.cpu().numpy())
            plt.show()

        return valid


    def sample_target_point(
        self, 
        start: torch.Tensor, 
        point: torch.Tensor, 
        planner: AStar,
        debug: bool=False
    ) -> Optional[np.ndarray]:
        """Sample a position near the mask and return.

        Args:
            look_at_any_point(bool): robot should look at the closest point on target mask instead of average pt
        """

        obstacles, explored = self.voxel_map.get_2d_map()

        # Extract edges from our explored mask
        start_pt = planner.to_pt(start)
        reachable_points = planner.get_reachable_points(start_pt)
        if len(reachable_points) == 0:
            print("No target point find, maybe no point is reachable")
            return None
        reachable_xs, reachable_ys = zip(*reachable_points)
        # # type: ignore comments used to bypass mypy check
        reachable_xs = torch.tensor(reachable_xs)  # type: ignore
        reachable_ys = torch.tensor(reachable_ys)  # type: ignore
        reachable = torch.empty(obstacles.shape, dtype=torch.bool).fill_(False)
        reachable[reachable_xs, reachable_ys] = True

        obstacles, explored = self.voxel_map.get_2d_map()
        reachable = reachable & ~obstacles

        target_x, target_y = planner.to_pt(point)

        xs, ys = torch.where(reachable)
        if len(xs) < 1:
            print("No target point find, maybe no point is reachable")
            return None
        selected_targets = torch.stack([xs, ys], dim=-1)[  # Calculate the distance from all reachable points to the target, and sort them from closest to furthest.
            torch.linalg.norm(
                (torch.stack([xs, ys], dim=-1) - torch.tensor([target_x, target_y])).float(), dim=-1
            )
            .topk(k=len(xs), largest=False)
            .indices
        ]

        for selected_target in selected_targets:
            selected_x, selected_y = planner.to_xy([selected_target[0], selected_target[1]])
            theta = self.compute_theta(selected_x, selected_y, point[0], point[1])

            target_is_valid = self.is_valid(np.array([selected_x, selected_y, theta]))  # Collision detection
            if not target_is_valid:
                continue
            if np.linalg.norm([selected_x - point[0], selected_y - point[1]]) <= self._min_frontier_distance:
                i = (point[0] - selected_target[0]) // abs(point[0] - selected_target[0])
                j = (point[1] - selected_target[1]) // abs(point[1] - selected_target[1])
                index_i = int(selected_target[0].int() + i)
                index_j = int(selected_target[1].int() + j)
                if obstacles[index_i][index_j]:
                    target_is_valid = False

            if not target_is_valid:
                continue
            if debug:
                self.debug_robot_position(np.array([selected_x, selected_y, theta]), planner)
            return np.array([selected_x, selected_y, theta])

        return None

    def sample_exploration(self, xyt, planner, text=None, semantic_rate:float=0.3, debug=False):
        """
        Sample an exploration target
        """
        obstacles, explored, history_soft = self.voxel_map.get_2d_map(
            return_history_id=True, kernel=5
        )
        outside_frontier = self.voxel_map.get_outside_frontier(xyt, planner)

        time_heuristics = self._time_heuristic(history_soft, outside_frontier)
        semantic_heuristics = self._semantic_heuristic(text) if text else None

        if semantic_heuristics is not None:
            total_heuristics = time_heuristics + semantic_rate * semantic_heuristics
        else:
            total_heuristics = time_heuristics

        rounded_heuristics = np.ceil(total_heuristics * 200) / 200
        max_heuristic = rounded_heuristics.max()
        indices = np.column_stack(np.where(rounded_heuristics == max_heuristic))
        robot_pt = np.asarray(planner.to_pt(xyt))
        min_dist_cells = self._min_frontier_distance / max(self.grid.resolution, self.tolerance)
        if len(indices) > 0 and min_dist_cells > 0:
            dists = np.linalg.norm(indices - robot_pt, axis=-1)
            valid_mask = dists >= min_dist_cells
            if np.any(valid_mask):
                indices = indices[valid_mask]
        farthest_index = np.argmax(np.linalg.norm(indices - robot_pt, axis=-1))
        index = indices[farthest_index]
        if debug:
            obstacles_np = obstacles.detach().cpu().numpy()
            explored_np = explored.detach().cpu().numpy()
            frontier_np = outside_frontier.detach().cpu().numpy()
            history_np = history_soft.detach().cpu().numpy()
            heuristics_np = np.asarray(total_heuristics)
            semantic_np = (
                np.asarray(semantic_heuristics)
                if semantic_heuristics is not None
                else np.zeros_like(heuristics_np)
            )

            if np.any(frontier_np):
                active_mask = frontier_np
            elif np.any(explored_np):
                active_mask = explored_np
            else:
                active_mask = obstacles_np
            row_slice, col_slice = self._compute_active_crop(active_mask, padding=25)

            cropped_obstacles = obstacles_np[row_slice, col_slice]
            cropped_explored = explored_np[row_slice, col_slice]
            cropped_frontier = frontier_np[row_slice, col_slice]
            cropped_history = history_np[row_slice, col_slice]
            cropped_heuristics = heuristics_np[row_slice, col_slice]
            cropped_semantic = semantic_np[row_slice, col_slice]
            local_index = (
                index[0] - row_slice.start,
                index[1] - col_slice.start,
            )

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))

            ax = axes[0, 0]
            ax.imshow(cropped_explored, cmap="Greens", alpha=0.45)
            ax.imshow(cropped_obstacles, cmap="Reds", alpha=0.5)
            if np.any(cropped_frontier):
                ax.contour(
                    cropped_frontier,
                    levels=[0.5],
                    colors="cyan",
                    linewidths=1.2,
                )
            # ax.scatter(
            #     local_index[1],
            #     local_index[0],
            #     s=50,
            #     c="yellow",
            #     edgecolors="black",
            #     linewidths=0.6,
            # )
            ax.set_title("Explored / Obstacles / Frontier")
            ax.set_xlabel("Grid Y")
            ax.set_ylabel("Grid X")
            ax.set_aspect("equal")

            ax = axes[0, 1]
            h_img = ax.imshow(cropped_heuristics, cmap="viridis")
            # ax.scatter(
            #     local_index[1],
            #     local_index[0],
            #     s=50,
            #     c="yellow",
            #     edgecolors="black",
            #     linewidths=0.6,
            # )
            ax.set_title("Total Heuristic Score")
            ax.set_xlabel("Grid Y")
            ax.set_ylabel("Grid X")
            ax.set_aspect("equal")
            fig.colorbar(h_img, ax=ax, fraction=0.046, pad=0.04, label="Score")

            ax = axes[1, 0]
            hist_img = ax.imshow(cropped_history, cmap="magma")
            ax.scatter(
                local_index[1],
                local_index[0],
                s=50,
                c="yellow",
                edgecolors="black",
                linewidths=0.6,
            )
            ax.set_title("History (recent observations)")
            ax.set_xlabel("Grid Y")
            ax.set_ylabel("Grid X")
            ax.set_aspect("equal")
            fig.colorbar(hist_img, ax=ax, fraction=0.046, pad=0.04, label="Obs id")

            ax = axes[1, 1]
            if semantic_heuristics is not None:
                sem_img = ax.imshow(cropped_semantic, cmap="plasma")
                ax.scatter(
                    local_index[1],
                    local_index[0],
                    s=50,
                    c="yellow",
                    edgecolors="black",
                    linewidths=0.6,
                )
                ax.set_title("Semantic Alignment")
                fig.colorbar(sem_img, ax=ax, fraction=0.046, pad=0.04, label="Similarity")
            else:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "Semantic alignment disabled",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

            for row in axes:
                for axis in row:
                    axis.set_xlim(-0.5, cropped_heuristics.shape[1] - 0.5)
                    axis.set_ylim(cropped_heuristics.shape[0] - 0.5, -0.5)

            plt.tight_layout()
            plt.savefig("debug_sample_exploration.svg", format="svg", dpi=200)
            plt.close(fig)
        return index, time_heuristics, total_heuristics

    def _time_heuristic(
        self, history_soft, outside_frontier, time_smooth=10, time_threshold=0.01
    ):
        history_soft = np.ma.masked_array(history_soft, ~outside_frontier)
        time_heuristics = history_soft.max() - history_soft
        time_heuristics[history_soft <= self.tolerance] = float("inf")
        time_heuristics = 1 / (1 + np.exp(-time_smooth * (time_heuristics - time_threshold)))
        # index = np.unravel_index(np.argmax(time_heuristics), history_soft.shape)
        return time_heuristics

    def _semantic_heuristic(self, text, semantic_smooth=10, semantic_threshold=0.01):
        semantic_heuristics = self.voxel_map.get_2d_alignment_heuristics(text=text)
        if isinstance(semantic_heuristics, torch.Tensor):
            semantic_heuristics = semantic_heuristics.detach().cpu().numpy()
        else:
            semantic_heuristics = np.asarray(semantic_heuristics)
        return 1 / (1 + np.exp(-semantic_smooth * (semantic_heuristics - semantic_threshold)))


    def _compute_active_crop(
        self, mask: Union[np.ndarray, torch.Tensor], padding: int = 25
    ) -> Tuple[slice, slice]:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        coords = np.argwhere(mask)
        if coords.size == 0:
            height, width = mask.shape
            return slice(0, height), slice(0, width)

        min_r = max(int(coords[:, 0].min()) - padding, 0)
        max_r = min(int(coords[:, 0].max()) + padding + 1, mask.shape[0])
        min_c = max(int(coords[:, 1].min()) - padding, 0)
        max_c = min(int(coords[:, 1].max()) + padding + 1, mask.shape[1])
        return slice(min_r, max_r), slice(min_c, max_c)

    def to_pt(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Converts a point from continuous, world xy coordinates to grid coordinates.

        Args:
            xy: The point in continuous xy coordinates.

        Returns:
            The point in discrete grid coordinates.
        """
        # # type: ignore to bypass mypy checking
        xy = np.array([xy[0], xy[1]])  # type: ignore
        pt = self.voxel_map.xy_to_grid_coords(xy)  # type: ignore
        return int(pt[0]), int(pt[1])

    def to_xy(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        """Converts a point from grid coordinates to continuous, world xy coordinates.

        Args:
            pt: The point in grid coordinates.

        Returns:
            The point in continuous xy coordinates.
        """
        # # type: ignore to bypass mypy checking
        pt = np.array([pt[0], pt[1]])  # type: ignore
        xy = self.voxel_map.grid_coords_to_xy(pt)  # type: ignore
        return float(xy[0]), float(xy[1])

    # def sample_navigation(self, start, planner: AStar, point, mode="navigation"):
    #     # plt.clf()
    #     if point is None:
    #         # start_pt = self.to_pt(start)
    #         return None
    #     goal = self.sample_target_point(start, point, planner)
    #     print("point:", point, "goal:", goal)
    #     return goal

    def sample_frontier(self, planner, start_pose=[0, 0, 0], text=None):
        (
            index,
            time_heuristics,
            total_heuristics,
        ) = self.sample_exploration(
            start_pose,
            planner,
            text=text,
            debug=False,
        )

        obstacles, explored = self.voxel_map.get_2d_map()
        return self.voxel_map.grid_coords_to_xyt(torch.tensor([index[0], index[1]]))
