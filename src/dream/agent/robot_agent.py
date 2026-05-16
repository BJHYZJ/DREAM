import os
import time
import timeit
from datetime import datetime
from typing import Any, Dict, Optional, Union
from uuid import uuid4
from threading import Lock, RLock, Thread
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
from dream.core.parameters import Parameters
from dream.core.robot import AbstractGraspClient
from dream.agent.zmq_client import RobotZmqClient
from dream.mapping.voxel import SparseVoxelMap
from dream.mapping.voxel import SparseVoxelMapNavigationSpace
from dream.motion.algo.a_star import AStar
from dream.motion import PlanResult
from dream.perception.detection.owl import OwlPerception
from dream.perception.encoders.siglip_encoder import MaskSiglipEncoder
from dream.perception.wrapper import OvmmPerception
from dream.utils.logger import Logger, color_text

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
        self._with_mllm_grounding = parameters["with_mllm_grounding"]
        self._with_mllm_verify = parameters["with_mllm_verify"]
        self._mllm_provider = parameters.get("mllm_provider", default="openai")
        self._mllm_model = parameters.get("mllm_model", default=None)

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
        self._focused_tracking_radius = parameters.get("agent/focused_tracking_radius", 2.0)
        self._focused_tracking_yaw_threshold = parameters.get(
            "agent/focused_tracking_yaw_threshold", 0.25
        )
        self._navigation_step_num = int(parameters.get("agent/navigation_step_num", 16))
        self._max_navigation_attempts = int(parameters.get("agent/max_navigation_attempts", 20))

        self._cached_navigation_goal = None
        self._cached_navigation_goal_lock = RLock()
        self._cached_navigation_goal_version = 0
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

        pose_graph_update_params = "agent/pose_graph_updates"
        self._pose_graph_updates_enabled: bool = parameters.get(
            f"{pose_graph_update_params}/enabled", True
        )
        self._with_big_loop_update: bool = parameters.get(
            f"{pose_graph_update_params}/with_big_loop", False
        )
        self._with_medium_loop_update: bool = parameters.get(
            f"{pose_graph_update_params}/with_medium_loop", True
        )
        self._with_small_loop_update: bool = parameters.get(
            f"{pose_graph_update_params}/with_small_loop", True
        )
        self._pose_trans_thresh: float = parameters.get(
            f"{pose_graph_update_params}/pose_trans_thresh", 0.03
        )  # metre
        self._pose_rot_thresh: float = parameters.get(
            f"{pose_graph_update_params}/pose_rot_thresh", 3.0
        )  # degree
        self._win_for_small_update: int = parameters.get(
            f"{pose_graph_update_params}/win_for_small_update", 50
        )  # in 3hz, around 10 second, must has at least one frame in nearest 6 frame
        self._win_for_realtime_update: int = parameters.get(
            f"{pose_graph_update_params}/win_for_realtime_update", 10
        )  # in 3hz, around 2 second
        self._max_obs_length = parameters.get(f"{pose_graph_update_params}/max_obs_length", 500)
        self._max_pose_graph_obs_length = parameters.get(
            f"{pose_graph_update_params}/max_pose_graph_obs_length",
            self._max_obs_length // 5,
        )
        self._pose_graph_poll_interval: float = parameters.get(
            f"{pose_graph_update_params}/poll_interval", 1.0
        )
        self._invalidate_cached_goal_on_pose_graph_update: bool = parameters.get(
            f"{pose_graph_update_params}/invalidate_cached_goal",
            parameters.get(f"{pose_graph_update_params}/invalidate_cached_traj", True),
        )

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
            time.sleep(0.05)


    def update_map_loop(self):
        """Threaded function that updates our voxel map in real-time."""
        while self.robot.running:
            self.update_map_with_pose_graph(
                with_big_loop=self._with_big_loop_update,
                with_medium_loop=self._with_medium_loop_update,
                with_small_loop=self._with_small_loop_update,
            )

    def update_map_with_pose_graph(
        self, 
        verbose: bool=True,
        with_big_loop: Optional[bool]=None,
        with_medium_loop: Optional[bool]=None,
        with_small_loop: Optional[bool]=None,
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
        
        if not self._pose_graph_updates_enabled:
            time.sleep(self._pose_graph_poll_interval)
            return

        if not self.voxel_map.observations or self.robot.in_task():
            time.sleep(self._pose_graph_poll_interval)
            return 
        pose_graph_data = self.robot.get_pose_graph()
        pose_graph_timestamp = pose_graph_data["timestamp"]
        if self._pose_graph_timestamp and pose_graph_timestamp <= self._pose_graph_timestamp:
            time.sleep(self._pose_graph_poll_interval)
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

            self._handle_pose_graph_map_update(update_length, verbose=verbose)

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
        time.sleep(self._pose_graph_poll_interval)

    def _handle_pose_graph_map_update(self, update_length: int, verbose: bool = False) -> None:
        """Invalidate the cached navigation goal after semantic-map changes."""
        if update_length <= 0:
            return

        if not self._invalidate_cached_goal_on_pose_graph_update:
            return

        self._clear_cached_navigation_goal()

        if verbose:
            logger.info(
                "Pose graph map update invalidated cached navigation goal."
            )

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
        # We personally feel that Siglipv1 is better than Siglipv2, but we still include the Siglipv2 in src/dream/perception/encoders/ for future reference
        self.encoder = MaskSiglipEncoder(
            version="so400m", feature_matching_threshold=0.14, device=self.device
        )

        # You can see a clear difference in hyperparameter selection in different querying strategies
        # Running gpt4o is time consuming, so we don't want to waste more time on object detection or Siglip or voxelization
        # On the other hand querying by feature similarity is fast and we want more fine grained details in semantic memory
        if self._with_mllm_grounding:
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
            with_mllm_grounding=self._with_mllm_grounding,
            with_mllm_verify=self._with_mllm_verify,
            mllm_provider=self._mllm_provider,
            mllm_model=self._mllm_model,
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
            # constants.look_ahead,
            constants.look_down,
            # constants.look_left_1,
            constants.look_left_2,
            # constants.look_right_1, 
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
        start = self.robot.get_base_in_map_xyt()
        cached_goal = self._get_cached_navigation_goal()
        focused_goal = self._get_focused_cached_navigation_goal(start)
        has_cached_goal = cached_goal is not None
        if focused_goal is not None:
            focused_tracking_text = (
                "### Using focused tracking near cached navigation goal; looking at target instead of sweeping."
            )
            logger.alert(focused_tracking_text)
            self.rerun_visualizer.log_text(
                "robot_monologue",
                "# Robot's monologue: \n" + focused_tracking_text,
            )
            finished, detected_goal_point = self._refresh_cached_navigation_goal_if_close(text)
            if finished:
                return True, detected_goal_point
        elif not has_cached_goal:
            self.look_around(speed=arm_speed)

        self.robot.move_to_nav_posture()

        start = self.robot.get_base_in_map_xyt()
        navigation_result = self.process_text(text, start, navigation_step_num=self._navigation_step_num)
        if (navigation_result is None or len(navigation_result) == 0) and text != "" and text is not None:
            navigation_result = self.process_text("", start, navigation_step_num=self._navigation_step_num)

        return self._execute_navigation_result(text, navigation_result)

    def _execute_navigation_result(
        self,
        text: str,
        navigation_result,
    ):
        if navigation_result is None or len(navigation_result) == 0:
            logger.error("Failed. Try again!")
            return None, None

        logger.alert("Plan successful!")
        if len(navigation_result) >= 2 and np.isnan(navigation_result[-2]).all():
            planned_goal_point = navigation_result[-1]
            if len(navigation_result) > 2:
                self.robot.execute_trajectory(
                    navigation_result[:-2],
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    blocking=True,
                    final_timeout=5.0,
                )

            verified, detected_goal_point = self._verify_target_after_navigation(text, planned_goal_point)
            if not verified:
                return False, None
            return True, detected_goal_point

        self.robot.execute_trajectory(
            navigation_result,
            pos_err_threshold=self.pos_err_threshold,
            rot_err_threshold=self.rot_err_threshold,
            blocking=True,
            final_timeout=5.0,
        )
        finished, detected_goal_point = self._refresh_cached_navigation_goal_if_close(text)
        if finished:
            return True, detected_goal_point
        return False, None

    def _verify_target_after_navigation(self, text: Optional[str], planned_goal_point: Optional[np.ndarray]):
        """Ensure the target still exists and is within manipulation range."""
        if text is None or text == "" or planned_goal_point is None:
            self._clear_cached_navigation_goal()
            return True, planned_goal_point

        logger.info("Verifying target existence after navigation...")
        detected_goal_point = self._focus_and_detect_goal(text, planned_goal_point)
        if detected_goal_point is None:
            self._clear_cached_navigation_goal()
            return False, None

        robot_xy = np.array(self.robot.get_base_in_map_xyt()[:2], dtype=float)
        planned_goal = self._target_to_numpy(planned_goal_point)
        planned_dist = (
            float(np.linalg.norm(robot_xy - planned_goal[:2]))
            if planned_goal is not None
            else float("nan")
        )
        detected_dist = float(np.linalg.norm(robot_xy - detected_goal_point[:2]))
        logger.info(
            "Target verification distance check: "
            f"robot_xy={np.array2string(robot_xy, precision=3)}, "
            f"planned_goal={np.array2string(planned_goal, precision=3) if planned_goal is not None else None}, "
            f"detected_goal={np.array2string(detected_goal_point, precision=3)}, "
            f"dist_to_planned={planned_dist:.2f} m, dist_to_detected={detected_dist:.2f} m"
        )
        if detected_dist > self._manipulation_radius:
            logger.warning(
                f"Target detected but outside manipulation radius ({detected_dist:.2f} m), continue navigation..."
            )
            self._clear_cached_navigation_goal()
            return False, None

        self._clear_cached_navigation_goal()
        logger.alert(f"Target verified within manipulation radius ({detected_dist:.2f} m).")
        return True, detected_goal_point

    def _target_to_numpy(self, point) -> Optional[np.ndarray]:
        if point is None:
            return None
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().numpy()
        point = np.asarray(point, dtype=float).reshape(-1)
        if len(point) < 2 or not np.all(np.isfinite(point[:2])):
            return None
        if len(point) == 2:
            point = np.array([point[0], point[1], 0.0], dtype=float)
        return point

    def _navigation_goal_marker(self):
        return [np.nan, np.nan, np.nan]

    def _is_navigation_goal_marker(self, point) -> bool:
        if point is None:
            return False
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().numpy()
        point = np.asarray(point, dtype=float).reshape(-1)
        return len(point) >= 2 and np.isnan(point[:2]).all()

    def _get_cached_navigation_goal(self) -> Optional[np.ndarray]:
        with self._cached_navigation_goal_lock:
            if self._cached_navigation_goal is None:
                return None
            return self._cached_navigation_goal.copy()

    def _get_cached_navigation_goal_snapshot(self):
        with self._cached_navigation_goal_lock:
            cached_goal = self._get_cached_navigation_goal()
            if cached_goal is None:
                return None, self._cached_navigation_goal_version
            return cached_goal, self._cached_navigation_goal_version

    def _cached_navigation_goal_version_matches(self, version: int) -> bool:
        with self._cached_navigation_goal_lock:
            return (
                self._cached_navigation_goal is not None
                and self._cached_navigation_goal_version == version
            )

    def _set_cached_navigation_goal(self, goal_point) -> None:
        goal = self._target_to_numpy(goal_point)
        with self._cached_navigation_goal_lock:
            if goal is None:
                self._cached_navigation_goal = None
            else:
                self._cached_navigation_goal = goal.copy()
            self._cached_navigation_goal_version += 1

    def _set_cached_navigation_goal_if_version(self, goal_point, version: int) -> bool:
        goal = self._target_to_numpy(goal_point)
        if goal is None:
            return False
        with self._cached_navigation_goal_lock:
            if self._cached_navigation_goal is None or self._cached_navigation_goal_version != version:
                return False
            self._cached_navigation_goal = goal.copy()
            self._cached_navigation_goal_version += 1
            return True

    def _clear_cached_navigation_goal(self) -> None:
        with self._cached_navigation_goal_lock:
            self._cached_navigation_goal = None
            self._cached_navigation_goal_version += 1

    def _target_distance(self, start_pose, goal_point: np.ndarray) -> float:
        start_xy = np.asarray(start_pose[:2], dtype=float)
        goal_xy = np.asarray(goal_point[:2], dtype=float)
        return float(np.linalg.norm(start_xy - goal_xy))

    def _get_focused_cached_navigation_goal(self, start_pose) -> Optional[np.ndarray]:
        cached_goal = self._get_cached_navigation_goal()
        if cached_goal is None:
            return None
        if self._target_distance(start_pose, cached_goal) > self._focused_tracking_radius:
            return None
        return cached_goal

    def _focus_and_detect_goal(self, text: Optional[str], expected_goal_point) -> Optional[np.ndarray]:
        if text is None or text == "":
            return None
        expected_goal = self._target_to_numpy(expected_goal_point)
        if expected_goal is None:
            return None

        prev_latest_obs_id = max(self.voxel_map.observations.keys()) if self.voxel_map.observations else None
        self._face_target_with_base(expected_goal)
        self.robot.look_at_target(tar_in_map=expected_goal, blocking=True)
        time.sleep(0.5)
        if not self._realtime_updates:
            self.update()

        if not self.voxel_map.observations:
            logger.error("No observations available to verify target.")
            return None

        obs_ids = sorted(self.voxel_map.observations.keys())
        new_obs_ids = (
            [obs_id for obs_id in obs_ids if obs_id > prev_latest_obs_id]
            if prev_latest_obs_id is not None
            else obs_ids
        )
        latest_obs_id = new_obs_ids[-1] if new_obs_ids else obs_ids[-1]
        text_exist, detected_goal_point = self.voxel_map.detect_text(
            text=text,
            obs_id=latest_obs_id,
            return_point=True,
            allow_feature_fallback=False,
        )
        if not text_exist:
            logger.warning("Target not found in focused view.")
            return None

        detected_goal_point = self._target_to_numpy(detected_goal_point)
        if detected_goal_point is None:
            logger.warning(
                "Target detected in focused view, but no valid current-frame target point was available."
            )
            return None
        return detected_goal_point

    def _refresh_cached_navigation_goal_if_close(self, text: Optional[str]):
        focused_goal = self._get_focused_cached_navigation_goal(self.robot.get_base_in_map_xyt())
        if focused_goal is None:
            return False, None

        detected_goal_point = self._focus_and_detect_goal(text, focused_goal)
        if detected_goal_point is None:
            self._clear_cached_navigation_goal()
            return False, None

        robot_xy = np.asarray(self.robot.get_base_in_map_xyt()[:2], dtype=float)
        detected_dist = float(np.linalg.norm(robot_xy - detected_goal_point[:2]))
        logger.info(
            "Focused target distance check: "
            f"robot_xy={np.array2string(robot_xy, precision=3)}, "
            f"detected_goal={np.array2string(detected_goal_point, precision=3)}, "
            f"dist_to_detected={detected_dist:.2f} m"
        )

        if detected_dist <= self._manipulation_radius:
            self._clear_cached_navigation_goal()
            logger.alert(f"Target verified within manipulation radius ({detected_dist:.2f} m).")
            return True, detected_goal_point

        self._set_cached_navigation_goal(detected_goal_point)
        logger.warning(
            f"Target detected but outside manipulation radius ({detected_dist:.2f} m), replanning from current pose..."
        )
        return False, None

    def _plan_to_cached_navigation_goal(self, start_pose, navigation_step_num: int):
        cached_goal, cache_version = self._get_cached_navigation_goal_snapshot()
        if cached_goal is None:
            return None

        navigation_viewpoint = self.space.sample_target_point(
            start=start_pose,
            point=cached_goal,
            planner=self.planner,
            debug=False,
        )
        if navigation_viewpoint is None:
            logger.warning("Unable to find a navigation viewpoint for cached navigation goal; replanning from text.")
            self._clear_cached_navigation_goal()
            return None

        plan_result = self.planner.plan(start_pose, navigation_viewpoint)
        if plan_result is None or not plan_result.success:
            if plan_result is not None:
                logger.error("[FAILURE]", plan_result.reason)
            self._clear_cached_navigation_goal()
            return None

        if not self._cached_navigation_goal_version_matches(cache_version):
            logger.warning("Cached navigation goal changed during replanning; discarding stale plan.")
            return None

        waypoints = [pt.state for pt in plan_result.trajectory]
        return self._build_navigation_result(
            waypoints=waypoints,
            navigation_goal=cached_goal,
            navigation_step_num=navigation_step_num,
            cache_target=True,
            expected_cache_version=cache_version,
        )

    def _build_navigation_result(
        self,
        waypoints,
        navigation_goal,
        navigation_step_num: int,
        cache_target: bool,
        expected_cache_version: Optional[int] = None,
    ):
        if waypoints is None:
            return []

        navigation_goal = self._target_to_numpy(navigation_goal)
        if navigation_goal is None:
            self._clear_cached_navigation_goal()
            return []

        if cache_target:
            finished = len(waypoints) <= navigation_step_num
            if not finished:
                if expected_cache_version is None:
                    self._set_cached_navigation_goal(navigation_goal)
                elif not self._set_cached_navigation_goal_if_version(navigation_goal, expected_cache_version):
                    logger.warning("Cached navigation goal changed during trajectory build; discarding stale plan.")
                    return None
                waypoints = waypoints[:navigation_step_num]
            elif (
                expected_cache_version is not None
                and not self._cached_navigation_goal_version_matches(expected_cache_version)
            ):
                logger.warning("Cached navigation goal changed before final verification; discarding stale plan.")
                return None
        else:
            finished = False
            waypoints = waypoints[:navigation_step_num]
            self._clear_cached_navigation_goal()

        if len(waypoints) == 0:
            return [self._navigation_goal_marker(), navigation_goal.tolist()] if finished else []

        navigation_result = self.planner.clean_path_for_xy(waypoints)
        if finished:
            navigation_result.append(self._navigation_goal_marker())
            navigation_result.append(navigation_goal.tolist())
        return navigation_result

    def _face_target_with_base(self, goal_point) -> None:
        goal = self._target_to_numpy(goal_point)
        if goal is None:
            return

        current_xyt = self.robot.get_base_in_map_xyt()
        vec_to_goal = np.asarray(goal[:2], dtype=float) - np.asarray(current_xyt[:2], dtype=float)
        if np.linalg.norm(vec_to_goal) < 1e-6:
            return

        target_yaw = float(np.arctan2(vec_to_goal[1], vec_to_goal[0]))
        yaw_error = (target_yaw - float(current_xyt[2]) + np.pi) % (2 * np.pi) - np.pi
        if abs(yaw_error) <= self._focused_tracking_yaw_threshold:
            return

        logger.alert(
            "Rotating base to face cached navigation goal before visual verification "
            f"({np.rad2deg(yaw_error):.1f} deg)."
        )
        self.robot.base_to(
            np.array([current_xyt[0], current_xyt[1], target_yaw]),
            blocking=True,
            timeout=8.0,
        )

    def _build_close_target_trajectory(self, start_pose, goal_point):
        """Stop regular navigation near the target, optionally rotating in place to face it."""
        start_xy = np.asarray(start_pose[:2], dtype=float)
        goal_xy = np.asarray(goal_point[:2], dtype=float)
        vec_to_goal = goal_xy - start_xy
        if np.linalg.norm(vec_to_goal) < 1e-6:
            return [[np.nan, np.nan, np.nan], goal_point.tolist()]

        target_bearing = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        base_yaw = float(start_pose[2])
        bearing_error = (target_bearing - base_yaw + np.pi) % (2 * np.pi) - np.pi

        target_xyt = [
            float(start_pose[0]),
            float(start_pose[1]),
            float(base_yaw + bearing_error),
        ]
        logger.alert(
            "Target is within manipulation radius; rotating base in place "
            f"({np.rad2deg(bearing_error):.1f} deg) instead of planning a nearby viewpoint."
        )
        return [target_xyt, [np.nan, np.nan, np.nan], goal_point.tolist()]

    def run_exploration(self):
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        status, _ = self.execute_action("")
        if status is None:
            print("Exploration failed! Perhaps nowhere to explore!")
            return False
        return True

    def process_text(
        self,
        text,
        start_pose,
        navigation_step_num: int,
        allow_early_stop: bool = True,
    ):
        """
        Process the text query and return the trajectory for the robot to follow.
        """
        logger.info("Processing", text, "starts")

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

        # Keep cached-goal replanning under the same map lock as normal text/frontier
        # planning because it also reads the 2D map and planner state.
        with self._voxel_map_lock:
            replanned_result = self._plan_to_cached_navigation_goal(start_pose, navigation_step_num)
            
        if replanned_result is not None:
            debug_text += "## Replan from current pose to cached navigation goal without semantic verification.\n"
            if text is not None and text != "":
                debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
            else:
                debug_text = "### I have not received any text query from human user.\n ### So, I plan to explore the environment with Frontier-based exploration.\n"
            debug_text = "# Robot's monologue: \n" + debug_text
            self.rerun_visualizer.log_text("robot_monologue", debug_text)
            print("Cached navigation goal replanned result:", replanned_result)
            return replanned_result

        # add voxel map lock ensure voxel map not update when path planning
        with self._voxel_map_lock:

            if text is not None and text != "" and localized_point is None:
                (
                    localized_point,
                    debug_text,
                    obs,
                    pointcloud,
                ) = self.voxel_map.localize_text(text, debug=True, return_debug=True)
                logger.alert("Target point selected!")

            # Do Frontier based exploration
            if text is None or text == "" or localized_point is None:
                self._clear_cached_navigation_goal()
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

            if allow_early_stop and mode == "navigation":
                start_xy = np.array(start_pose[:2], dtype=float)
                if isinstance(localized_point, torch.Tensor):
                    localized_goal = localized_point.detach().cpu().numpy()
                else:
                    localized_goal = np.asarray(localized_point)
                localized_goal = np.asarray(localized_goal, dtype=float).reshape(-1)
                finished_point = localized_goal.tolist()

                if np.linalg.norm(start_xy - localized_goal[:2]) <= self._manipulation_radius:
                    logger.alert(
                        "Robot already within manipulation radius; skipping target viewpoint planning."
                    )
                    debug_text += (
                        "## Robot already within manipulation radius; skip target viewpoint planning and verify for grasping.\n"
                    )
                    navigation_result = self._build_close_target_trajectory(start_pose, localized_goal)
                    self.rerun_visualizer.log_custom_pointcloud(
                        "world/object",
                        [finished_point[0], finished_point[1], 0.5],
                        torch.Tensor([0, 1, 0]),
                        0.1,
                    )

                    if text is not None and text != "":
                        debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
                    else:
                        debug_text = "### I have not received any text query from human user.\n ### So, I plan to explore the environment with Frontier-based exploration.\n"
                    debug_text = "# Robot's monologue: \n" + debug_text
                    self.rerun_visualizer.log_text("robot_monologue", debug_text)
                    return navigation_result

            navigation_viewpoint = self.space.sample_target_point(
                start=start_pose, 
                point=localized_point, 
                planner=self.planner,
                debug=False,  # for visualization
            )

            # print("localized_point:", localized_point, "navigation_viewpoint:", navigation_viewpoint)

            logger.info("Navigation endpoint selected")

            waypoints = None

            if navigation_viewpoint is None:
                plan_result = None
                logger.warning("Unable to find any target point, some exception might happen")
            else:
                plan_result = self.planner.plan(start_pose, navigation_viewpoint)

        if plan_result is not None and plan_result.success:
            waypoints = [pt.state for pt in plan_result.trajectory]
        elif plan_result is not None:
            waypoints = None
            logger.error("[FAILURE]", plan_result.reason)

        navigation_result = []
        if waypoints is not None:
            self.rerun_visualizer.log_custom_pointcloud(
                "world/object",
                [localized_point[0], localized_point[1], 0.5],
                torch.Tensor([0, 1, 0]),
                0.1,
            )

            navigation_result = self._build_navigation_result(
                waypoints=waypoints,
                navigation_goal=localized_point,
                navigation_step_num=navigation_step_num,
                cache_target=mode == "navigation",
            )
            print("Planned trajectory:", navigation_result)

        if text is not None and text != "":
            debug_text = "### The goal is to navigate to " + text + ".\n" + debug_text
        else:
            debug_text = "### I have not received any text query from human user.\n ### So, I plan to explore the environment with Frontier-based exploration.\n"
        
        debug_text = "# Robot's monologue: \n" + debug_text
        self.rerun_visualizer.log_text("robot_monologue", debug_text)

        if navigation_result is not None:
            origins = []
            vectors = []
            for idx in range(len(navigation_result) - 1):
                start_pt = navigation_result[idx]
                end_pt = navigation_result[idx + 1]
                if isinstance(start_pt, torch.Tensor):
                    start_pt = start_pt.detach().cpu().numpy()
                if isinstance(end_pt, torch.Tensor):
                    end_pt = end_pt.detach().cpu().numpy()
                start_pt = np.asarray(start_pt, dtype=float).reshape(-1)
                end_pt = np.asarray(end_pt, dtype=float).reshape(-1)
                if (
                    len(start_pt) < 2
                    or len(end_pt) < 2
                    or not np.all(np.isfinite(start_pt[:2]))
                    or not np.all(np.isfinite(end_pt[:2]))
                ):
                    continue
                origins.append([start_pt[0], start_pt[1], 0.5])
                vectors.append([end_pt[0] - start_pt[0], end_pt[1] - start_pt[1], 0])
            if origins:
                self.rerun_visualizer.log_arrow3D(
                    "world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1
                )
            self.rerun_visualizer.log_custom_pointcloud(
                "world/robot_start_pose",
                [start_pose[0], start_pose[1], 0.5],
                torch.Tensor([0, 0, 1]),
                0.1,
            )

        return navigation_result


    def get_voxel_map(self):
        """Return the voxel map in use by this model"""
        return self.voxel_map
    

    def navigate(self, text):
        # rr.init("Dream_robot", recording_id=uuid4(), spawn=True)
        finished = False
        navigation_attempt = 0
        end_point = None
        while not finished and navigation_attempt < self._max_navigation_attempts:
            logger.info(
                color_text("*" * 20, "cyan"),
                f"navigation attempt {navigation_attempt + 1}/{self._max_navigation_attempts}",
                color_text("*" * 20, "cyan"),
            )
            navigation_attempt += 1
            finished, end_point = self.execute_action(text)
            if finished is None:
                logger.error("Navigation failed! The path might be blocked!")
                return None

        if not finished:
            logger.warning(
                "Navigation reached the maximum number of attempts "
                f"({self._max_navigation_attempts}) before verifying the target."
            )
            return None

        logger.alert("Navigation finished!")
        return end_point

    def place(        
        self,
        back_object: str,
        target_receptacle: str,
        target_point: None,
        skip_confirmation: bool = False,
    ):
        self.robot.switch_to_manipulation_mode()
        theta_cumulative = 0.0
        slam_paused = False
        try:
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
                slam_paused = True
                success = place(
                    socket=self.manip_socket,
                    manip_wrapper=self.manip_wrapper,
                    back_object=back_object,
                    translation=translation,
                )
                self.robot.resume_slam()
                slam_paused = False
                if not success:
                    print("(ಥ﹏ಥ) Place task failed.")
                    return False

            return True
        finally:
            if slam_paused:
                self.robot.resume_slam()

            # Shift base back even if manipulation fails midway.
            if theta_cumulative:
                self.manip_wrapper.move_to_position(base_theta=np.deg2rad(-theta_cumulative))

            # Ensure camera/head pose is reset for the next trial.
            self.manip_wrapper.robot.look_front()


    def manipulate(
        self,
        target_object,
        target_point: Optional[np.ndarray]=None,
        skip_confirmation: bool=False,
        just_anygrasp: bool=False,
        just_heuristic: bool=False,
        two_stage: bool=True,
    ):

        self.robot.switch_to_manipulation_mode()
        theta_cumulative = 0.0
        slam_paused = False
        try:
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
                slam_paused = True
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
                slam_paused = False
                if not success:
                    print("(ಥ﹏ಥ) Pickup task failed.")
                    return False

            return True
        finally:
            if slam_paused:
                self.robot.resume_slam()

            # Shift base back even if manipulation fails midway.
            if theta_cumulative:
                self.manip_wrapper.move_to_position(base_theta=np.deg2rad(-theta_cumulative))

            # Ensure camera/head pose is reset for the next trial.
            self.manip_wrapper.robot.look_front()
