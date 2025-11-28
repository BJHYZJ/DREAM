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

import os
import time
import timeit
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from threading import Lock, Thread
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import zmq
import dream.motion.constants as constants

from dream.agent.manipulation.dream_manipulation.dream_manipulation import (
    DreamManipulationWrapper as ManipulationWrapper,
)
from dream.agent.manipulation.dream_manipulation.grasper_utils import (
    capture_and_process_image,
    pickup,
    place,
)
from dream.core.interfaces import Observations
from dream.core.parameters import Parameters
from dream.core.robot import AbstractGraspClient, AbstractRobotClient
from dream.agent.zmq_client import RobotZmqClient
from dream.mapping.voxel import SparseVoxelMap
from dream.mapping.voxel import SparseVoxelMapNavigationSpace
from dream.motion.algo.a_star import AStar
from dream.motion import ConfigurationSpace, Planner, PlanResult
from dream.perception.detection.owl import OwlPerception
from dream.perception.encoders.siglip_encoder import MaskSiglipEncoder
from dream.perception.wrapper import OvmmPerception
from dream.utils.logger import Logger

logger = Logger(__name__)

class RobotAgent:
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
        robot: RobotZmqClient,
        parameters: Union[Parameters, Dict[str, Any]],
        semantic_sensor: Optional[OvmmPerception] = None,
        grasp_client: Optional[AbstractGraspClient] = None,
        manip_port: int = 5557,
        log: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
    ):
        self.reset_object_plans()
        if isinstance(parameters, Dict):
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
        else:
            raise RuntimeError(f"parameters of unsupported type: {type(parameters)}")
        self.robot = robot
        self.grasp_client = grasp_client

        self.semantic_sensor = semantic_sensor
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]

        self.rerun_visualizer = self.robot._rerun
        self.setup_custom_blueprint()

        # For placing
        self.owl_sam_detector = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists("dream_log"):
            os.makedirs("dream_log")

        if log is None:
            current_datetime = datetime.now()
            self.log = "dream_log/debug_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.log = "dream_log/" + log

        self._manipulation_radius = parameters["motion_planner"]["goals"]["manipulation_radius"]
        self._mllm = parameters["mllm"]
        self._with_mllm_verify = parameters["with_mllm_verify"]

        self.create_obstacle_map(parameters)

        # ==============================================
        self._realtime_updates = self.parameters["agent"]["use_realtime_updates"]

        # ==============================================
        # Update configuration
        # If true, the head will sweep on update, collecting more information.
        self._sweep_head_on_update = parameters["agent"]["sweep_head_on_update"]

        # ==============================================
        # Task-level parameters
        # Grasping parameters
        self.target_object = None
        self.target_receptacle = None
        # ==============================================

        # Parameters for feature matching and exploration
        self._is_match_threshold = parameters.get("encoder_args/feature_match_threshold", 0.05)
        self._grasp_match_threshold = parameters.get(
            "encoder_args/grasp_feature_match_threshold", 0.05
        )

        # Expanding frontier - how close to frontier are we allowed to go?
        self._default_expand_frontier_size = parameters["motion_planner"]["frontier"][
            "default_expand_frontier_size"
        ]
        self._frontier_min_dist = parameters["motion_planner"]["frontier"]["min_dist"]
        self._frontier_step_dist = parameters["motion_planner"]["frontier"]["step_dist"]
        
        self._voxel_size = parameters["voxel_size"]

        self._pose_trans_thresh: float = 0.03  # metre
        self._pose_rot_thresh: float = 3.0  # degree
        self._win_for_small_update: int = 50  # in 3hz, around 10 second, must has at least one frame in nearest 6 frame
        self._win_for_realtime_update: int = 10  # in 3hz, around 2 second
        self._max_obs_length = 500
        self._max_pose_graph_obs_length = self._max_obs_length // 5

        self._pose_graph_timestamp = None
        self._obs_timestamp = None

        context = zmq.Context()
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect("tcp://" + server_ip + ":" + str(manip_port))

        end_link = "gripper"
        self.transform_node = end_link
        self.manip_wrapper = ManipulationWrapper(
            self.robot, gripper_max=830, end_link=end_link
        )
        self.robot.move_to_nav_posture()

        self.reset_object_plans()

        # Store the current scene graph computed from detected objects
        self.scene_graph = None

        # Previously sampled goal during exploration
        self._previous_goal = None

        self._start_threads()


    def _start_threads(self):
        """Create threads and locks for real-time updates."""
        # Create Lock for voxel map
        self._voxel_map_lock = Lock()

        # Map updates
        self._update_map_thread = Thread(target=self.update_map_loop)
        self._update_map_thread.start()

        if self._realtime_updates:
            logger.alert("Using real-time updates!")
            # Get observations thread
            self._get_observations_thread = Thread(target=self.get_observations_loop)
            self._get_observations_thread.start()


    def get_observations_loop(self, verbose: bool=False, visualize_map: bool=True) -> None:
        while self.robot.running:
            if self.robot.in_task():
                time.sleep(0.5)
                continue
            t0 = timeit.default_timer()
            obs = self.robot.get_observation()
            obs_timestamp = obs.timestamp
            if self._obs_timestamp and obs_timestamp <= self._obs_timestamp:
                continue
            self._obs_timestamp = obs_timestamp
            with self._voxel_map_lock:
                self.voxel_map.process_rgbd_images(
                    rgb=obs.rgb, 
                    depth=obs.depth, 
                    intrinsics=obs.camera_K, 
                    camera_pose=obs.camera_in_map_pose,
                    base_pose=obs.base_in_map_pose,
                    obs_id=obs.obs_id,
                )

                if visualize_map:
                    if self.voxel_map.semantic_memory._points is not None and \
                            self.voxel_map.semantic_memory._points.nelement() != 0:
                        self.rerun_visualizer.update_voxel_map(space=self.space)
                    if self.voxel_map.semantic_memory._points is not None and \
                            len(self.voxel_map.semantic_memory._points) != 0:
                        self.rerun_visualizer.log_custom_pointcloud(
                            "world/semantic_memory/pointcloud",
                            self.voxel_map.semantic_memory._points.detach().cpu(),
                            self.voxel_map.semantic_memory._rgb.detach().cpu() / 255.0,
                            0.03,
                        )
                
            # Add a delay to make sure we don't get too many observations
            t1 = timeit.default_timer()
            if verbose:
                print(f"Done getting an observation, spend {t1 - t0}")
            time.sleep(0.5)


    def update_map_loop(self):
        """Threaded function that updates our voxel map in real-time."""
        while self.robot.running:
            self.update_map_with_pose_graph()

    def update_map_with_pose_graph(
        self, 
        verbose: bool=True,
        with_big_loop=False,
        with_medium_loop=True,
        with_small_loop=True,
    ) -> None:
        """ 
        Update our voxel pointcloud and semantic memory using a pose graph

        Modify node pose and add it to the scene with the new pose
            self._win_for_small_update: default 30, around 10 second
            self._win_for_realtime_update: default 6, around 2 second

        1. Check if a large loop closure has occurred. If so, directly call self.reset to reset the global scene. len(affected_ids) / len(shared_ids) > 1 / 3?
        2. If more than 1 / 5 frames in the last `self._win_for_small_update` frames also in affected_ids exist, and at least one frame falls within the last 
            `self._win_for_realtime_update` frames, then add all is_pose_graph_nodes from the last `self._win_for_small_update` frames to the scene with the updated pose.
        3. If neither of the above two situations occurs, and one frame in the last `self._win_for_realtime_update` frames has 1 ID in affected_ids, then re-add that frame to the scene.
        4. Do nothing.

        More Details
        1. When observations length is more than self._max_obs_length, we delete oldest node which `is_pose_graph_node` is false, and when `is_pose_graph_node` length is more than 
            self._max_pose_graph_obs_length, we also delete mode which `is_pose_graph_node` is true
        2. To reduce resource consumption during scene reconstruction and maintenance, each call to `process_rgbd_images` in `get_observations_loop` saves the features. 
            In the current function, based on the latest `is_pose_graph_node`, we delete all nodes in `self.observations` except for the latest `self._win_for_realtime_update` nodes 
            (the `self._win_for_realtime_update` nodes are a buffer because these nodes may be assigned features in the next round of `get_observations_loop`).
        """

        # Thanks to the design of clear_points, a `self.voxel_map.reset()` is not required, which effectively reduces computational overhead.
        # We simply re-added the most recent 10 frames to the scene according to the latest pose.
        
        if not self.voxel_map.observations or self.robot.in_task():
            time.sleep(0.5)
            return 
        pose_graph_data = self.robot.get_pose_graph()
        pose_graph_timestamp = pose_graph_data["timestamp"]
        if self._pose_graph_timestamp and pose_graph_timestamp <= self._pose_graph_timestamp:
            return
        self._pose_graph_timestamp = pose_graph_timestamp
        pose_graph = pose_graph_data["pose_graph"]
        pose_graph_ids = list(pose_graph.keys())
        
        with self._voxel_map_lock:
            t0 = timeit.default_timer()
            obs_ids = sorted(self.voxel_map.observations.keys())
            
            # Find the nodes in the pose graph. 
            shared_ids = (set(pose_graph_ids) & set(obs_ids))
            for sid in shared_ids:
                self.voxel_map.observations[sid].is_pose_graph_node = True

            # calculation pose change, and sign ids
            affected_ids = []
            if shared_ids:
                for sid in shared_ids:
                    obs = self.voxel_map.observations[sid]
                    assert obs.is_pose_graph_node, "The is_pose_graph_node should set to True when Node in pose_graph."
                    
                    camera_pose_now = torch.tensor(
                        pose_graph[sid], dtype=torch.float32)
                    
                    rot_origin = obs.camera_pose[:3, :3]
                    trans_origin = obs.camera_pose[:3, 3]
                    rot_now = camera_pose_now[:3, :3]
                    trans_now = camera_pose_now[:3, 3]

                    dr = rot_now @ rot_origin.T
                    # Robust rotation angle via atan2(sin, cos)
                    c = (torch.trace(dr) - 1.0) * 0.5
                    v = torch.stack([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]])
                    s = 0.5 * torch.linalg.norm(v)
                    trans_diff = torch.linalg.norm(trans_now - trans_origin).item()
                    rot_diff = torch.rad2deg(torch.atan2(s, c)).item()
                    assert np.isfinite(trans_diff) and np.isfinite(rot_diff), "assert trans_diff and rot_diff is finite"

                    if trans_diff > self._pose_trans_thresh or rot_diff > self._pose_rot_thresh:
                    # if trans_diff > 0 or rot_diff > 0:  # for DEBUG
                        affected_ids.append(sid)
                        # print(f"trans_diff: {trans_diff}, rot_diff: {rot_diff}")

            def re_add_semantic_memory(obs):
                # update obs pose from pose graph
                if obs.obs_id in pose_graph:
                    camera_pose_now = torch.tensor(
                        pose_graph[obs.obs_id], dtype=torch.float32)
                    obs.camera_pose = camera_pose_now

                features = obs.feats
                if self.voxel_map.compression_features and features is not None:
                    # decompress feature
                    features = self.voxel_map.feature_decompression(features)

                self.voxel_map.add_to_semantic_memory(
                    camera_pose=obs.camera_pose,
                    base_pose=obs.base_pose,
                    rgb=obs.rgb,
                    obs_id=obs.obs_id,
                    camera_K=obs.camera_K,
                    depth=obs.depth,
                    valid_depth=obs.valid_depth,
                    feats=features,
                )    

            update_length = 0
            if affected_ids:
                if with_big_loop and len(self.voxel_map.observations) > self._win_for_realtime_update and \
                    shared_ids and len(affected_ids) / len(shared_ids) > 2 / 3:
                    # big loop, reset voxel map semantic memory and re-add all PG nodes with optimized poses
                    self.voxel_map.reset()
                    for obs_id in obs_ids:  # sorted ids
                        obs = self.voxel_map.observations[obs_id]
                        assert obs.obs_id == obs_id
                        if not obs.is_pose_graph_node:
                            continue
                        re_add_semantic_memory(obs)
                        update_length += 1

                    if verbose:
                        t1 = timeit.default_timer()
                        print(f"[LOOP]: BIG LOOP happend! Spend time: {t1 - t0}, Update nodes: {update_length}")

                elif with_medium_loop and len(obs_ids) > self._win_for_small_update and \
                    len(set(obs_ids[-self._win_for_small_update:]) & set(affected_ids)) / self._win_for_small_update > 1 / 3 and \
                        len(set(obs_ids[-self._win_for_realtime_update:]) & set(affected_ids)) != 0:
                    # medium loop, reset voxel map semantic memory and re-add all PG nodes with optimized poses
                    for obs_id in obs_ids[-self._win_for_small_update:]:
                        obs = self.voxel_map.observations[obs_id]
                        assert obs.obs_id == obs_id
                        if not obs.is_pose_graph_node:
                            continue
                        re_add_semantic_memory(obs)
                        update_length += 1

                    if verbose:
                        t1 = timeit.default_timer()
                        print(f"[LOOP]: Medium LOOP happend! Spend time: {t1 - t0}, Update nodes: {update_length}")

                elif with_small_loop and len(set(obs_ids[-self._win_for_realtime_update:]) & set(affected_ids)) != 0:
                    for obs_id in obs_ids[-self._win_for_realtime_update:]:
                        obs = self.voxel_map.observations[obs_id]
                        assert obs.obs_id == obs_id
                        if not obs.is_pose_graph_node:
                            continue
                        re_add_semantic_memory(obs)
                        update_length += 1
                    if verbose:
                        t1 = timeit.default_timer()
                        print(f"[LOOP]: Small LOOP happend! Spend time: {t1 - t0}, Update nodes: {update_length}")
                else:
                    if verbose and False:
                        print(f"[No Loop Happend]: observations lenght: {len(self.voxel_map.observations)}, shared_ids length: {len(shared_ids)}, affected_ids: {affected_ids}")
                    pass
            # Limit the number of observations
            if len(self.voxel_map.observations) > self._max_obs_length:
                obs_ids_now = sorted(self.voxel_map.observations.keys())
                obs_length = len(obs_ids_now)
                need_del_lenght = obs_length - self._max_obs_length
                pose_graph_node_count = sum(1 for o in self.voxel_map.observations.values() if o.is_pose_graph_node)
                del_obs_ids = []  # need del obs ids
                for oid in obs_ids_now:
                    if need_del_lenght <= 0:
                        break
                    is_pose_graph_node = self.voxel_map.observations[oid].is_pose_graph_node
                    if (not is_pose_graph_node) or (is_pose_graph_node and pose_graph_node_count > self._max_pose_graph_obs_length):
                        del_obs_ids.append(oid)
                        need_del_lenght -= 1
                        if is_pose_graph_node:
                            pose_graph_node_count -= 1
                    
                for nd_id in del_obs_ids:
                    del self.voxel_map.observations[nd_id]
                

            # Clear feature when node is not pose graph node
            obs_ids_now = sorted(self.voxel_map.observations.keys())
            for obs_id in obs_ids_now[:-self._win_for_realtime_update]:
                obs = self.voxel_map.observations[obs_id]
                if not obs.is_pose_graph_node:
                    obs.feats = None

            if verbose and False:
                t2 = timeit.default_timer()
                print(f"[Update map with pose graph realtime] spend time: {t2 - t0}, observations length: {len(obs_ids_now)}")
                print("=" * 60)
        # print(f"Observations Len: {len(self.voxel_map.observations)}")
        time.sleep(1)

    def reset_object_plans(self):
        """Clear stored object planning information."""

        # Dictionary storing attempts to visit each object
        self._object_attempts: Dict[int, int] = {}
        self._cached_plans: Dict[int, PlanResult] = {}


    def start(
        self,
        goal: Optional[str] = None,
        can_move: bool = True,
        verbose: bool = True,
    ) -> None:

        # Call the robot's own startup hooks
        started = self.robot.start()
        if not started:
            # update here
            raise RuntimeError("Robot failed to start!")

        if verbose:
            print("ZMQ connection to robot started.")

        if can_move:
            # First, open the gripper...
            self.robot.switch_to_manipulation_mode()
            self.robot.open_gripper()

            # Tuck the arm away
            if verbose:
                print("Sending arm to home...")
            self.robot.move_to_nav_posture()
            if verbose:
                print("... done.")

        # Move the robot into navigation mode
        self.robot.switch_to_navigation_mode()
        if verbose:
            print("- Update map after switching to navigation posture")

    def create_obstacle_map(self, parameters):
        """
        This function creates the MaskSiglipEncoder, Owlv2 detector, voxel map util class and voxel map navigation space util class
        """

        # Initialize the encoder in different ways depending on the configuration
        # Use SIGLip-so400m for accurate inference
        # We personally feel that Siglipv1 is better than Siglipv2, but we still include the Siglipv2 in src/stretch/perception/encoders/ for future reference
        self.encoder = MaskSiglipEncoder(
            version="so400m", feature_matching_threshold=0.14, device=self.device
        )

        # You can see a clear difference in hyperparameter selection in different querying strategies
        # Running gpt4o is time consuming, so we don't want to waste more time on object detection or Siglip or voxelization
        # On the other hand querying by feature similarity is fast and we want more fine grained details in semantic memory
        if self._mllm:
            self.detection_model = OwlPerception(
                version="owlv2-B-p16", device=self.device, confidence_threshold=0.01
            )
            image_shape = (360, 270)
        else:
            self.detection_model = OwlPerception(
                version="owlv2-L-p14-ensemble", device=self.device, confidence_threshold=0.15
            )
            # image_shape = (480, 360)
            image_shape = (360, 720)
        self.voxel_map = SparseVoxelMap(
            voxel_resolution=parameters["voxel_size"],
            local_radius=parameters["local_radius"],
            ground_max_height=parameters["ground_max_height"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density=parameters["obs_min_density"],
            grid_resolution=parameters["grid_resolution"],
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            pad_obstacles=parameters["pad_obstacles"],  # in meter
            add_local_radius_points=parameters.get("add_local_radius_points", default=True),
            remove_visited_from_obstacles=parameters.get(
                "remove_visited_from_obstacles", default=False
            ),
            smooth_kernel_size=parameters.get("filters/smooth_kernel_size", -1),
            use_median_filter=parameters.get("filters/use_median_filter", False),
            median_filter_size=parameters.get("filters/median_filter_size", 5),
            median_filter_max_error=parameters.get("filters/median_filter_max_error", 0.01),
            use_derivative_filter=parameters.get("filters/use_derivative_filter", False),
            derivative_filter_threshold=parameters.get("filters/derivative_filter_threshold", 0.5),
            detection=self.detection_model,
            encoder=self.encoder,
            image_shape=image_shape,
            log=self.log,
            mllm=self._mllm,
            with_mllm_verify=self._with_mllm_verify,
        )
        self.space = SparseVoxelMapNavigationSpace(
            self.robot,
            self.voxel_map,
            rotation_step_size=parameters.get("motion_planner/rotation_step_size", 0.2),
            dilate_frontier_size=parameters.get("motion_planner/frontier/dilate_frontier_size", 2),
            dilate_obstacle_size=parameters.get("motion_planner/frontier/dilate_obstacle_size", 0),
            min_frontier_distance=self._manipulation_radius,
        )
        self.planner = AStar(self.space)

    def setup_custom_blueprint(self):
        main = rrb.Horizontal(
            rrb.Spatial3DView(name="3D View", origin="world"),
            rrb.Vertical(
                rrb.TextDocumentView(name="text", origin="robot_monologue"),
                rrb.Spatial2DView(name="image", origin="/observation_similar_to_text"),
            ),
            rrb.Vertical(
                # rrb.Spatial2DView(name="head_rgb", origin="/world/head_camera"),
                # rrb.Spatial2DView(name="ee_rgb", origin="/world/ee_camera"),
                rrb.Spatial2DView(name="rgb", origin="/world/camera/rgb"),
                rrb.Spatial2DView(name="rgb_servo", origin='world/camera/rgb_servo')
                # rrb.Spatial2DView(name="obj", origin="/world/camera/obj_mask"),
            ),
            column_shares=[2, 1, 1],
        )
        my_blueprint = rrb.Blueprint(
            rrb.Vertical(main, rrb.TimePanel(state=True)),
            collapse_panels=True,
        )
        rr.send_blueprint(my_blueprint)

    def compute_blur_metric(self, image):
        """
        Computes a blurriness metric for an image tensor using gradient magnitudes.

        Parameters:
        - image (torch.Tensor): The input image tensor. Shape is [H, W, C].

        Returns:
        - blur_metric (float): The computed blurriness metric.
        """

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients using the Sobel operator
        Gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        Gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        G = cv2.magnitude(Gx, Gy)

        # Compute the mean of gradient magnitudes
        blur_metric = G.mean()

        return blur_metric
    

    def update(self, visualize_map: bool=True):
        """Step the data collector. Get a single observation of the world. Remove bad points, such as those from too far or too near the camera. Update the 3d world representation."""
        # Sleep some time for the robot camera to focus
        # time.sleep(0.3)
        obs = self.robot.get_observation()
        with self._voxel_map_lock:
            self.voxel_map.process_rgbd_images(
                rgb=obs.rgb, 
                depth=obs.depth, 
                intrinsics=obs.camera_K, 
                camera_pose=obs.camera_in_map_pose,
                base_pose=obs.base_in_map_pose,
                obs_id=obs.obs_id,
            )
            if visualize_map:
                if self.voxel_map.semantic_memory._points is not None and \
                        len(self.voxel_map.semantic_memory._points) != 0:
                    self.rerun_visualizer.update_voxel_map(space=self.space)
                if self.voxel_map.semantic_memory._points is not None and \
                        len(self.voxel_map.semantic_memory._points) != 0:
                    self.rerun_visualizer.log_custom_pointcloud(
                        "world/semantic_memory/pointcloud",
                        self.voxel_map.semantic_memory._points.detach().cpu(),
                        self.voxel_map.semantic_memory._rgb.detach().cpu() / 255.0,
                        0.03,
                    )

    def look_around(self, speed: int=50):
        print("*" * 10, "Look around to check", "*" * 10)
        for angle in [
            constants.look_ahead,
            constants.look_down,
            constants.look_left_1,
            constants.look_left_2,
            constants.look_right_1, 
            constants.look_right_2,
            constants.look_front,
        ]:
            self.robot.arm_to(angle=angle, speed=speed, blocking=True)
            if not self._realtime_updates:
                self.update()
            else:
                time.sleep(0.1)  # waiting for 1 second

    def rotate_in_place(self, speed: int=50):
        print("*" * 10, "Rotate in place", "*" * 10)
        xyt = self.robot.get_base_in_map_xyt()
        self.robot.arm_to(angle=constants.look_down, speed=speed, blocking=True)
        for i in range(8):  # TODO range(8)
            xyt[2] += 2 * np.pi / 8
            self.robot.base_to(xyt, blocking=True)
            if not self._realtime_updates:
                self.update()

    def execute_action(
        self,
        text: str,
        arm_speed=40,
    ):
        # self.robot.look_front(speed=arm_speed)
        self.look_around(speed=arm_speed)
        # self.robot.look_front(speed=arm_speed)
        self.robot.switch_to_navigation_mode()

        start = self.robot.get_base_in_map_xyt()
        res = self.process_text(text, start)
        if len(res) == 0 and text != "" and text is not None:
            res = self.process_text("", start)

        if len(res) > 0:
            print("Plan successful!")
            goal_point = res[-1]
            if len(res) >= 2 and np.isnan(res[-2]).all():
                if len(res) > 2:
                    self.robot.execute_trajectory(
                        res[:-2],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                        blocking=True,
                        final_timeout=5.0,
                    )

                verified = self._verify_target_after_navigation(text, goal_point)
                if not verified:
                    return False, None
                return True, goal_point
            else:
                self.robot.execute_trajectory(
                    res,
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    blocking=True,
                    final_timeout=5.0,
                )
                return False, None
        else:
            print("Failed. Try again!")
            return None, None

    def _verify_target_after_navigation(self, text: Optional[str], end_point: Optional[np.ndarray]) -> bool:
        """Ensure the target still exists and is within manipulation range."""
        if text is None or text == "" or end_point is None:
            return True

        print("verifying target existence after navigation...")
        self.robot.look_at_target(tar_in_map=end_point, blocking=True)
        time.sleep(3)
        if not self._realtime_updates:
            self.update()

        if not self.voxel_map.observations:
            print("No observations available to verify target.")
            return False

        latest_obs_id = max(self.voxel_map.observations.keys())
        text_exist = self.voxel_map.detect_text(text=text, obs_id=latest_obs_id)
        if not text_exist:
            print("Target not found after navigation, continue navigation...")
            return False

        robot_xy = np.array(self.robot.get_base_in_map_xyt()[:2])
        target_xy = np.array(end_point[:2])
        dist = np.linalg.norm(robot_xy - target_xy)
        if dist > self._manipulation_radius:
            print(
                f"Target detected but outside manipulation radius ({dist:.2f} m), continue navigation..."
            )
            return False

        return True

    def run_exploration(self):
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        status, _ = self.execute_action("")
        if status is None:
            print("Exploration failed! Perhaps nowhere to explore!")
            return False
        return True

    def process_text(self, text, start_pose, step_num=24):
        """
        Process the text query and return the trajectory for the robot to follow.
        """

        print("Processing", text, "starts")

        self.rerun_visualizer.clear_identity("world/object")
        self.rerun_visualizer.clear_identity("world/robot_start_pose")
        self.rerun_visualizer.clear_identity("world/direction")
        self.rerun_visualizer.clear_identity("robot_monologue")
        self.rerun_visualizer.clear_identity("/observation_similar_to_text")

        debug_text = ""
        mode = "navigation"
        obs = None
        localized_point = None
        waypoints = None

        # add voxel map lock ensure voxel map not update when path planning
        with self._voxel_map_lock:  

            if text is not None and text != "" and self.space.traj is not None:
                print("saved traj", self.space.traj)
                traj_target_point = self.space.traj[-1]
                if hasattr(self.encoder, "feature_matching_threshold") and self.voxel_map.verify_point(
                    text,
                    traj_target_point,
                    similarity_threshold=self.encoder.feature_matching_threshold
                ):
                    localized_point = traj_target_point
                    debug_text += "## Last visual grounding results looks fine so directly use it.\n"

            print("Target verification finished")
            if text is not None and text != "" and localized_point is None:
                (
                    localized_point,
                    debug_text,
                    obs,
                    pointcloud,
                ) = self.voxel_map.localize_text(text, debug=True, return_debug=True)
                print("Target point selected!")

            # Do Frontier based exploration
            if text is None or text == "" or localized_point is None:
                debug_text += "## Navigation fails, so robot starts exploring environments.\n"
                localized_point = self.space.sample_frontier(self.planner, start_pose, text)
                mode = "exploration"

            if obs is not None and mode == "navigation":
                print(obs, len(self.voxel_map.observations))
                obs = self.voxel_map.find_obs_id_for_text(text)
                rgb = self.voxel_map.observations[obs].rgb
                self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", rgb)

            if localized_point is None:
                return []  # try to found object by frontier

            # # TODO: Do we really need this line?
            if len(localized_point) == 2:
                localized_point = np.array([localized_point[0], localized_point[1], 0])

            point = self.space.sample_target_point(
                start=start_pose, 
                point=localized_point, 
                planner=self.planner,
                debug=False,  # for visualization
            )

            # print("localized_point:", localized_point, "goal:", point)

            print("Navigation endpoint selected")

            waypoints = None

            if point is None:
                res = None
                print("Unable to find any target point, some exception might happen")
            else:
                res = self.planner.plan(start_pose, point)

        if res is not None and res.success:
            waypoints = [pt.state for pt in res.trajectory]
        elif res is not None:
            waypoints = None
            print("[FAILURE]", res.reason)

        traj = []
        close_enough = False
        if (
            waypoints is not None
            and mode == "navigation"
            and localized_point is not None
            and len(localized_point) >= 2
        ):
            start_xy = np.array(start_pose[:2], dtype=float)
            if isinstance(localized_point, torch.Tensor):
                localized_xy = localized_point.detach().cpu().numpy()
            else:
                localized_xy = np.asarray(localized_point)
            localized_xy = localized_xy[:2]
            close_enough = np.linalg.norm(start_xy - localized_xy) <= self._manipulation_radius

        if waypoints is not None:
            self.rerun_visualizer.log_custom_pointcloud(
                "world/object",
                [localized_point[0], localized_point[1], 0.5],
                torch.Tensor([0, 1, 0]),
                0.1,
            )

            finished = len(waypoints) <= step_num and mode == "navigation"
            if close_enough:
                finished = True
                debug_text += "## Robot already within manipulation radius; executing full trajectory.\n"

            if finished:
                self.space.traj = None
            else:
                self.space.traj = waypoints[step_num:] + [[np.nan, np.nan, np.nan], localized_point]
            if not finished:
                waypoints = waypoints[:step_num]
            traj = self.planner.clean_path_for_xy(waypoints)
            if finished:
                traj.append([np.nan, np.nan, np.nan])
                if isinstance(localized_point, torch.Tensor):
                    localized_point = localized_point.tolist()
                traj.append(localized_point)
            print("Planned trajectory:", traj)

        if text is not None and text != "":
            debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
        else:
            debug_text = "### I have not received any text query from human user.\n ### So, I plan to explore the environment with Frontier-based exploration.\n"
        
        debug_text = "# Robot's monologue: \n" + debug_text
        self.rerun_visualizer.log_text("robot_monologue", debug_text)

        if traj is not None:
            origins = []
            vectors = []
            for idx in range(len(traj)):
                if idx != len(traj) - 1:
                    origins.append([traj[idx][0], traj[idx][1], 0.5])
                    vectors.append(
                        [traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0]
                    )
            self.rerun_visualizer.log_arrow3D(
                "world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1
            )
            self.rerun_visualizer.log_custom_pointcloud(
                "world/robot_start_pose",
                [start_pose[0], start_pose[1], 0.5],
                torch.Tensor([0, 0, 1]),
                0.1,
            )

        return traj


    def get_voxel_map(self):
        """Return the voxel map in use by this model"""
        return self.voxel_map
    

    def navigate(self, text, max_step=20):
        # rr.init("Dream_robot", recording_id=uuid4(), spawn=True)
        finished = False
        step = 0
        end_point = None
        while not finished and step < max_step:
            print("*" * 20, "navigation step", step, "*" * 20)
            step += 1
            finished, end_point = self.execute_action(text)
            if finished is None:
                print("Navigation failed! The path might be blocked!")
                return None

        print("Navigation finished!")
        return end_point

    def place(        
        self,
        back_object: str,
        target_receptacle: str,
        target_point: None,
        skip_confirmation: bool = False,
    ):
        self.robot.switch_to_manipulation_mode()

        rotation, translation, theta_cumulative = capture_and_process_image(
            mode="place",
            obj=target_receptacle,
            tar_in_map=target_point,
            socket=self.manip_socket,
            manip_wrapper=self.manip_wrapper,
        )

        if rotation is None:
            print("(ಥ﹏ಥ) Try all pose but not suit pose for place.")
            return False

        if skip_confirmation or input("Do you want to do this place manipulation? Y or N ") != "N":
            self.robot.pause_slam()
            success = place(
                socket=self.manip_socket,
                manip_wrapper=self.manip_wrapper,
                back_object=back_object,
                translation=translation,
            )
            self.robot.resume_slam()
            if not success:
                print("(ಥ﹏ಥ) Place task failed.")
                return False

        # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
        if theta_cumulative != 0:
            self.manip_wrapper.move_to_position(base_theta=np.deg2rad(-theta_cumulative))

        return True


    def manipulate(
        self,
        target_object,
        target_point: Optional[np.ndarray]=None,
        skip_confirmation: bool=False,
        just_anygrasp: bool=False,
        just_heuristic: bool=False,
        two_stage: bool=True,
    ):
        """
        An API for running manipulation. By calling this API, human will ask the robot to pick up objects
        specified by target_object queries A
        - hello_robot: a wrapper for home-robot StretchClient controller
        - socoket: we use this to communicate with workstation to get estimated gripper pose
        - target_object: queries specifying target object
        - transform node: node name for coordinate systems of target gripper pose (usually the coordinate system on the robot gripper)
        - base node: node name for coordinate systems of estimated gipper poses given by anygrasp
        """

        self.robot.switch_to_manipulation_mode()

        rotation, translation, depth, width, obj_points, retry_flag, theta_cumulative = capture_and_process_image(
            mode="pick",
            obj=target_object,
            tar_in_map=target_point,
            socket=self.manip_socket,
            manip_wrapper=self.manip_wrapper,
        )

        if rotation is None and retry_flag != 3:
            print("(ಥ﹏ಥ) Try all pose but anygrasp is failed.")
            return False
        
        if not just_heuristic:
            just_heuristic = retry_flag == 3

        if skip_confirmation or input("Do you want to do this pickup manipulation? Y or N ") != "N":
            self.robot.pause_slam()
            success = pickup(
                self.manip_wrapper,
                rotation,
                translation,
                object_points=obj_points,
                just_heuristic=just_heuristic,
                just_anygrasp=just_anygrasp,
                two_stage=two_stage,
            )
            self.robot.resume_slam()
            if not success:
                print("(ಥ﹏ಥ) Pickup task failed.")
                return False

        # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
        if theta_cumulative != 0:
            self.manip_wrapper.move_to_position(base_theta=np.deg2rad(-theta_cumulative))

        return True
