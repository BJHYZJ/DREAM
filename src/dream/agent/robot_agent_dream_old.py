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
from dream.agent.robot_agent import RobotAgent as RobotAgentBase
from dream.audio.text_to_speech import get_text_to_speech
from dream.core.interfaces import Observations
from dream.core.parameters import Parameters
from dream.core.robot import AbstractGraspClient, AbstractRobotClient
from dream.mapping.instance import Instance
from dream.mapping.voxel import SparseVoxelMap
from dream.mapping.voxel import SparseVoxelMapNavigationSpace
from dream.mapping.voxel import SparseVoxelMapProxy
from dream.motion.algo.a_star import AStar
from dream.motion import ConfigurationSpace, Planner, PlanResult
from dream.perception.detection.owl import OwlPerception
from dream.perception.encoders.siglip_encoder import MaskSiglipEncoder
from dream.perception.wrapper import OvmmPerception
from dream.utils.logger import Logger

logger = Logger(__name__)

# Manipulation hyperparameters
INIT_LIFT_POS = 0.45
INIT_WRIST_PITCH = -1.57
INIT_ARM_POS = 0
INIT_WRIST_ROLL = 0
INIT_WRIST_YAW = 0
INIT_HEAD_PAN = -1.57
INIT_HEAD_TILT = -0.65


class RobotAgent:
    """Basic demo code. Collects everything that we need to make this work."""

    def __init__(
        self,
        robot: AbstractRobotClient,
        parameters: Union[Parameters, Dict[str, Any]],
        semantic_sensor: Optional[OvmmPerception] = None,
        grasp_client: Optional[AbstractGraspClient] = None,
        debug_instances: bool = True,
        show_instances_detected: bool = False,
        use_instance_memory: bool = False,
        # re: int = 3,
        manip_port: int = 5557,
        log: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        mllm: bool = False,
        manipulation_only: bool = False,
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
        self.debug_instances = debug_instances
        self.show_instances_detected = show_instances_detected

        self.semantic_sensor = semantic_sensor
        self.pos_err_threshold = parameters["trajectory_pos_err_threshold"]
        self.rot_err_threshold = parameters["trajectory_rot_err_threshold"]

        self.rerun_visualizer = self.robot._rerun
        self.setup_custom_blueprint()

        self.mllm = mllm
        self.manipulation_only = manipulation_only
        # For placing
        self.owl_sam_detector = None

        # if self.parameters.get("encoder", None) is not None:
        #     self.encoder: BaseImageTextEncoder = get_encoder(
        #         self.parameters["encoder"], self.parameters.get("encoder_args", {})
        #     )
        # else:
        #     self.encoder: BaseImageTextEncoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists("dream_log"):
            os.makedirs("dream_log")

        if log is None:
            current_datetime = datetime.now()
            self.log = "dream_log/debug_" + current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.log = "dream_log/" + log

        self.create_obstacle_map(parameters)

        # ==============================================
        self.guarantee_instance_is_reachable = self.parameters.guarantee_instance_is_reachable
        self.use_scene_graph = self.parameters["use_scene_graph"]
        self.tts = get_text_to_speech(self.parameters["tts_engine"])
        self._use_instance_memory = use_instance_memory
        self._realtime_updates = self.parameters["agent"]["use_realtime_updates"]

        # ==============================================
        # Update configuration
        # If true, the head will sweep on update, collecting more information.
        self._sweep_head_on_update = parameters["agent"]["sweep_head_on_update"]

        # ==============================================
        # Task-level parameters
        # Grasping parameters
        self.current_receptacle: Instance = None
        self.current_object: Instance = None
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
        self._manipulation_radius = parameters["motion_planner"]["goals"]["manipulation_radius"]
        self._voxel_size = parameters["voxel_size"]

        self._pose_trans_thresh: float = 0.01  # metre
        self._pose_rot_thresh: float = 1.0  # degree

        # self.image_processor = VoxelMapImageProcessor(
        #     rerun=True,
        #     rerun_visualizer=self.robot._rerun,
        #     log="dream_log/" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        #     robot=self.robot,
        # )  # type: ignore
        # self.encoder = self.image_processor.get_encoder()
        context = zmq.Context()
        self.manip_socket = context.socket(zmq.REQ)
        self.manip_socket.connect("tcp://" + server_ip + ":" + str(manip_port))

        # if re == 1 or re == 2:
        #     stretch_gripper_max = 0.3
        #     end_link = "link_straight_gripper"
        # else:
        #     stretch_gripper_max = 0.64
        #     end_link = "link_gripper_s3_body"
        end_link = "gripper"
        self.transform_node = end_link
        self.manip_wrapper = ManipulationWrapper(
            self.robot, gripper_max=830, end_link=end_link
        )
        self.robot.move_to_nav_posture()

        self.reset_object_plans()

        # self.re = re

        # Store the current scene graph computed from detected objects
        self.scene_graph = None

        # Previously sampled goal during exploration
        self._previous_goal = None

        self._start_threads()


    def _start_threads(self):
        """Create threads and locks for real-time updates."""

        # Track if we are still running
        self._voxel_map_lock = Lock()
        self._obs_history_lock = Lock()
        self.obs_history: List[Observations] = []

        # Map updates
        self._update_map_thread = Thread(target=self.update_map_loop)
        if self._realtime_updates:
            self._update_map_thread = Thread(target=self.update_map_loop_realtime)
        self._update_map_thread.start()

        if self._realtime_updates:
            logger.alert("Using real-time updates!")

            # Prune old observations
            # self._prune_old_observations_thread = Thread(target=self.prune_old_observations_loop)
            # self._prune_old_observations_thread.start()

            # Match pose graph
            # self._match_pose_graph_thread = Thread(target=self.match_pose_graph_loop)
            # self._match_pose_graph_thread.start()

            # Get observations thread
            self._get_observations_thread = Thread(target=self.get_observations_loop)
            self._get_observations_thread.start()


    # def get_observations_loop(self, verbose=False) -> None:
    #     while self.robot.running:
    #         t0 = timeit.default_timer()
    #         with self._voxel_map_lock:
    #             self.update()
    #         t1 = timeit.default_timer()
    #         if verbose:
    #             print(f"[LOOP Logger] Get observations loop spend {t1 - t0}")
    #         time.sleep(0.5)

    def get_observations_loop(self, verbose=False) -> None:
        while self.robot.running:
            t0 = timeit.default_timer()
            obs = self.robot.get_observation(wait_for_new=True)
            # with self._obs_history_lock:
            #     self.obs_history.append(obs)
            with self._voxel_map_lock:
                self.voxel_map.process_rgbd_images(
                    rgb=obs.rgb, 
                    depth=obs.depth, 
                    intrinsics=obs.camera_K, 
                    camera_pose=obs.camera_in_map_pose,
                    base_pose=obs.base_in_map_pose, 
                    local_tf=obs.camera_in_base_pose, 
                    node_id=obs.node_id,
                )
            # Add a delay to make sure we don't get too many observations
            t1 = timeit.default_timer()
            if verbose:
                print(f"Done getting an observation, spend {t1 - t0}")
            time.sleep(0.5)


    # def match_pose_graph_loop(self, verbose=False) -> None:
    #     while self.robot.running:
    #         t0 = timeit.default_timer()
    #         self.pose_graph = self.robot.get_pose_graph(wait_for_new=True)
    #         with self._voxel_map_lock:
    #             pass

    def update_map_loop_realtime(self):
        """Threaded function that updates our voxel map in real-time."""
        while self.robot.running:
            self.update_map_with_pose_graph_realtime()
        time.sleep(0.5)

    def update_map_loop(self):
        """Threaded function that updates our voxel map in real-time."""
        while self.robot.running:
            # self.update_map_with_pose_graph()
            pass
        time.sleep(0.5)



    def update_map_with_pose_graph_realtime(
        self, 
        obs_length_win: int=5, 
        verbose: bool=False
    ) -> None:
        """Update our voxel pointcloud and semantic memory using a pose graph"""
        # Thanks to the design of clear_points, a `self.voxel_map.reset()` is not required, which effectively reduces computational overhead.
        # We simply re-added the most recent 10 frames to the scene according to the latest pose.
        if len(self.obs_history) == 0: return
        t0 = timeit.default_timer()
        pose_graph = self.robot.get_pose_graph(wait_for_new=True)
        pose_graph_ids = list(pose_graph.keys())
        
        with self._obs_history_lock:
            obs_ids = [obs.node_id for obs in self.obs_history]
            
            # Find the nodes in the pose graph. 
            shared_ids = (set(pose_graph_ids) &  set(obs_ids))
            for sid in shared_ids:
                self.obs_history[obs_ids.index(sid)].is_pose_graph_node = True

            # calculation pose change, and update it
            if shared_ids:
                for obs in self.obs_history:
                    if obs.node_id not in shared_ids:
                        continue
                    optimized_base_in_map_pose = torch.tensor(
                        pose_graph[obs.node_id].matrix(), dtype=torch.float32)
                    camera_pose_origin = obs.camera_pose
                    camera_pose_now = optimized_base_in_map_pose @ obs.local_tf
                    
                    rot_origin = camera_pose_origin[:3, :3]
                    trans_origin = camera_pose_origin[:3, 3]
                    rot_now = camera_pose_now[:3, :3]
                    trans_now = camera_pose_now[:3, 3]

                    trans_diff = np.linalg.norm(trans_now - trans_origin)
                    dr = rot_now @ rot_origin.T
                    trace = np.clip((np.trace(dr) - 1) / 2.0, -1.0, 1.0)
                    rot_diff = np.rad2deg(np.arccos(trace))

                    # if trans_diff > self._pose_trans_thresh or rot_diff > self._pose_rot_thresh:
                    if trans_diff > 0 or rot_diff > 0:  # for DEBUG
                        obs.camera_pose = optimized_base_in_map_pose @ obs.local_tf
                        obs.base_pose = optimized_base_in_map_pose      
            
                t1 = timeit.default_timer()
                if verbose:
                    print(f"Done updating past observations. Time: {t1 - t0}")

                if len(self.obs_history) > 0:
                    obs_history_win = self.obs_history[-obs_length_win:]
                    blurness = [self.compute_blur_metric(obs.rgb) for obs in obs_history_win]
                    observation = obs_history_win[blurness.index(max(blurness))]
                else:
                    observation = None

                t2 = timeit.default_timer()
                if verbose:
                    print(f"Done get an observation for process. Time: {t2 - t1}")

                feats = None
                if observation.feats is not None:
                    feats = self.voxel_map.feature_decompression(observation.feats)

                with self._voxel_map_lock:
                    self.voxel_map.process_rgbd_images(
                        rgb=observation.rgb, 
                        depth=observation.depth, 
                        intrinsics=observation.K, 
                        camera_pose=observation.camera_pose,
                        base_pose=observation.base_pose, 
                        local_tf=observation.local_tf, 
                        node_id=observation.node_id,
                    )

        t3 = timeit.default_timer()
        if verbose:
            logger.info(f"update map with observation spend times: {t3 - t2}")

        time.sleep(0.5)

    # def update_map_with_pose_graph_realtime(
    #     self, 
    #     nearset_frame_lenght: int=20, 
    #     verbose: bool=False
    # ) -> None:
    #     """Update our voxel pointcloud and semantic memory using a pose graph"""
    #     # Thanks to the design of clear_points, a `self.voxel_map.reset()` is not required, which effectively reduces computational overhead.
    #     # We simply re-added the most recent 10 frames to the scene according to the latest pose.
    #     if len(self.voxel_map.observations) == 0: return
    #     t0 = timeit.default_timer()
    #     pose_graph = self.robot.get_pose_graph(wait_for_new=True)
    #     pose_graph_ids = list(pose_graph.keys())
    #     with self._voxel_map_lock:
    #         num_obs = len(self.voxel_map.observations)
    #         voxel_map_ids = [obs.node_id for obs in self.voxel_map.observations]
            
    #         # Find the nodes in the pose graph. 
    #         shared_ids = (set(pose_graph_ids) &  set(voxel_map_ids))
    #         for sid in shared_ids:
    #             self.voxel_map.observations[voxel_map_ids.index(sid)].is_pose_graph_node = True

    #         window_size = min(nearset_frame_lenght, num_obs)
    #         start_win = num_obs - window_size
    #         voxel_map_ids_win = voxel_map_ids[start_win:]
            

    #         assert len(set(pose_graph_ids)) == len(pose_graph_ids) and len(set(voxel_map_ids_win)) == len(voxel_map_ids_win)
    #         shared_ids_win = set(pose_graph_ids) & set(voxel_map_ids_win)
    #         # calculation pose change
    #         effected_ids = []  # Record the smallest ID that meets the conditions.
    #         if shared_ids_win:
    #             for frame in self.voxel_map.observations[start_win:]:
    #                 if frame.node_id not in shared_ids_win:
    #                     continue
    #                 optimized_base_in_map_pose = torch.tensor(
    #                     pose_graph[frame.node_id].matrix(), dtype=torch.float32)
    #                 camera_pose_origin = frame.camera_pose
    #                 camera_pose_now = optimized_base_in_map_pose @ frame.local_tf
                    
    #                 rot_origin = camera_pose_origin[:3, :3]
    #                 trans_origin = camera_pose_origin[:3, 3]
    #                 rot_now = camera_pose_now[:3, :3]
    #                 trans_now = camera_pose_now[:3, 3]

    #                 trans_diff = np.linalg.norm(trans_now - trans_origin)
    #                 dr = rot_now @ rot_origin.T
    #                 trace = np.clip((np.trace(dr) - 1) / 2.0, -1.0, 1.0)
    #                 rot_diff = np.rad2deg(np.arccos(trace))

    #                 # if trans_diff > self._pose_trans_thresh or rot_diff > self._pose_rot_thresh:
    #                 if trans_diff > 0 or rot_diff > 0:  # for DEBUG
    #                     effected_ids.append(frame.node_id)
            
    #         if effected_ids:
    #             # small_offset = voxel_map_ids_win.index(effected_ids[0])
    #             # start_index = start_win + small_offset
    #             start_index = voxel_map_ids.index(effected_ids[0])
                
    #             # re-add observations to the map by new pose
    #             for frame in self.voxel_map.observations[start_index:]:
    #                 if frame.node_id in effected_ids:
    #                     optimized_base_in_map_pose = torch.tensor(
    #                         pose_graph[frame.node_id].matrix(), dtype=torch.float32)
    #                     frame.camera_pose = optimized_base_in_map_pose @ frame.local_tf
    #                     frame.base_pose = optimized_base_in_map_pose             

    #                 self.voxel_map.add_to_voxel_pointcloud(
    #                     camera_pose=frame.camera_pose,
    #                     rgb=frame.rgb,
    #                     depth=frame.depth,
    #                     valid_depth=frame.valid_depth,
    #                     camera_K=frame.camera_K,
    #                     base_pose=frame.base_pose,
    #                 )

    #                 self.voxel_map.add_to_semantic_memory(
    #                     camera_pose=frame.camera_pose,
    #                     rgb=frame.rgb,
    #                     depth=frame.depth,
    #                     valid_depth=frame.valid_depth,
    #                     camera_K=frame.intrinsics,
    #                 )
                            
    #             t1 = timeit.default_timer()
    #             if verbose:
    #                 logger.info(f"update map with pose graph with {len(self.voxel_map.observations[start_index:])} observation spend times: {t1 - t0}")
    #     time.sleep(0.5)


    # def update_map_with_pose_graph(
    #     self, 
    #     nearset_frame_lenght: int=20, 
    #     verbose: bool=False
    # ):
    #     """Update our voxel pointcloud and semantic memory using a pose graph"""
    #     # Thanks to the design of clear_points, a `self.voxel_map.reset()` is not required, which effectively reduces computational overhead.
    #     # We simply re-added the most recent 10 frames to the scene according to the latest pose.
    #     if len(self.voxel_map.observations) == 0: return
    #     t0 = timeit.default_timer()
    #     pose_graph = self.robot.get_pose_graph(wait_for_new=True)
    #     with self._voxel_map_lock:
    #         num_obs = len(self.voxel_map.observations)
    #         window_size = min(nearset_frame_lenght, num_obs)
    #         start_win = num_obs - window_size
    #         voxel_map_ids = [obs.node_id for obs in self.voxel_map.observations[start_win:]]
    #         pose_graph_ids = list(pose_graph.keys())

    #         assert len(set(pose_graph_ids)) == len(pose_graph_ids) and len(set(voxel_map_ids)) == len(voxel_map_ids)
    #         shared_ids = set(pose_graph_ids) & set(voxel_map_ids)
    #         # calculation pose change
    #         effected_ids = []  # Record the smallest ID that meets the conditions.
    #         if shared_ids:
    #             for frame in self.voxel_map.observations[start_win:]:
    #                 if frame.node_id not in shared_ids:
    #                     continue
    #                 optimized_base_in_map_pose = torch.tensor(
    #                     pose_graph[frame.node_id].matrix(), dtype=torch.float32)
    #                 camera_pose_origin = frame.camera_pose
    #                 camera_pose_now = optimized_base_in_map_pose @ frame.local_tf
                    
    #                 rot_origin = camera_pose_origin[:3, :3]
    #                 trans_origin = camera_pose_origin[:3, 3]
    #                 rot_now = camera_pose_now[:3, :3]
    #                 trans_now = camera_pose_now[:3, 3]

    #                 trans_diff = np.linalg.norm(trans_now - trans_origin)
    #                 dr = rot_now @ rot_origin.T
    #                 trace = np.clip((np.trace(dr) - 1) / 2.0, -1.0, 1.0)
    #                 rot_diff = np.rad2deg(np.arccos(trace))

    #                 if trans_diff > self._pose_trans_thresh or rot_diff > self._pose_rot_thresh:
    #                 # if trans_diff > 0 or rot_diff > 0:  # for DEBUG
    #                     effected_ids.append(frame.node_id)
            
    #         if effected_ids:
    #             small_offset = voxel_map_ids.index(effected_ids[0])
    #             start_index = start_win + small_offset
    #             # re-add observations to the map by new pose
    #             for frame in self.voxel_map.observations[start_index:]:
    #                 if frame.node_id in effected_ids:
    #                     optimized_base_in_map_pose = torch.tensor(
    #                         pose_graph[frame.node_id].matrix(), dtype=torch.float32)
    #                     frame.camera_pose = optimized_base_in_map_pose @ frame.local_tf
    #                     frame.base_pose = optimized_base_in_map_pose             

    #                 self.voxel_map.add_to_voxel_pointcloud(
    #                     camera_pose=frame.camera_pose,
    #                     rgb=frame.rgb,
    #                     camera_K=frame.camera_K,
    #                     depth=frame.depth,
    #                     base_pose=frame.base_pose,
    #                 )

    #                 self.voxel_map.add_to_semantic_memory(
    #                     camera_pose=frame.camera_pose,
    #                     rgb=frame.rgb,
    #                     camera_K=frame.camera_K,
    #                     depth=frame.depth,
    #                     feats=frame.feats,
    #                 )
                            
    #             t1 = timeit.default_timer()
    #             if verbose:
    #                 logger.info(f"update map with pose graph with {len(self.voxel_map.observations[start_index:])} observation spend times: {t1 - t0}")
    #     time.sleep(0.5)


    def reset_object_plans(self):
        """Clear stored object planning information."""

        # Dictionary storing attempts to visit each object
        self._object_attempts: Dict[int, int] = {}
        self._cached_plans: Dict[int, PlanResult] = {}

        # Objects that cannot be reached
        self.unreachable_instances = set()


    def start(
        self,
        goal: Optional[str] = None,
        visualize_map_at_start: bool = False,
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

        # Add some debugging stuff - show what 3d point clouds look like
        if visualize_map_at_start:
            if not self._realtime_updates:
                # self.update(visualize_map=False)  # Append latest observations
                self.update()
            print("- Visualize map after updating")
            with self._voxel_map_lock:
                self.voxel_map.show(
                    orig=np.zeros(3),
                    xyt=self.robot.get_base_in_map_xyt(),
                    footprint=self.robot.get_robot_model().get_footprint(),
                    instances=self.semantic_sensor is not None,
                )
            self.print_found_classes(goal)


    def print_found_classes(self, goal: Optional[str] = None, verbose: bool = False):
        """Helper. print out what we have found according to detic."""
        if self.semantic_sensor is None:
            logger.warning("Tried to print classes without semantic sensor!")
            return

        with self._voxel_map_lock:
            instances = self.voxel_map.get_instances()
            if goal is not None:
                print(f"Looking for {goal}.")
            print("So far, we have found these classes:")
            for i, instance in enumerate(instances):
                oid = int(instance.category_id.item())
                name = self.semantic_sensor.get_class_name_for_id(oid)
                print(i, name, instance.score)
                if goal is not None:
                    if self.is_match_by_feature(instance, goal):
                        if verbose:
                            from matplotlib import pyplot as plt

                            plt.imshow(instance.get_best_view().cropped_image.int())
                            plt.show()


    def create_obstacle_map(self, parameters):
        """
        This function creates the MaskSiglipEncoder, Owlv2 detector, voxel map util class and voxel map navigation space util class
        """

        # Initialize the encoder in different ways depending on the configuration
        if self.manipulation_only:
            self.encoder = None
        else:
            # Use SIGLip-so400m for accurate inference
            # We personally feel that Siglipv1 is better than Siglipv2, but we still include the Siglipv2 in src/stretch/perception/encoders/ for future reference
            self.encoder = MaskSiglipEncoder(
                version="so400m", feature_matching_threshold=0.14, device=self.device
            )

        # You can see a clear difference in hyperparameter selection in different querying strategies
        # Running gpt4o is time consuming, so we don't want to waste more time on object detection or Siglip or voxelization
        # On the other hand querying by feature similarity is fast and we want more fine grained details in semantic memory
        voxel_resolution = None
        if self.manipulation_only:
            self.detection_model = None
            image_shape = (360, 270)
        elif self.mllm:
            self.detection_model = OwlPerception(
                version="owlv2-B-p16", device=self.device, confidence_threshold=0.01
            )
            image_shape = (360, 270)
        else:
            self.detection_model = OwlPerception(
                version="owlv2-L-p14-ensemble", device=self.device, confidence_threshold=0.15
            )
            voxel_resolution = 0.05
            image_shape = (480, 360)
        self.voxel_map = SparseVoxelMap(
            voxel_resolution=parameters["voxel_size"] if voxel_resolution is None else voxel_resolution,
            local_radius=parameters["local_radius"],
            obs_min_height=parameters["obs_min_height"],
            obs_max_height=parameters["obs_max_height"],
            obs_min_density=parameters["obs_min_density"],
            neg_obs_height=parameters["neg_obs_height"],
            use_negative_obstacles=parameters["use_negative_obstacles"],
            grid_resolution=0.1,
            min_depth=parameters["min_depth"],
            max_depth=parameters["max_depth"],
            pad_obstacles=parameters["pad_obstacles"],
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
            mllm=self.mllm,
        )
        self.space = SparseVoxelMapNavigationSpace(
            self.robot,
            self.voxel_map,
            rotation_step_size=parameters.get("motion_planner/rotation_step_size", 0.2),
            dilate_frontier_size=parameters.get("motion_planner/frontier/dilate_frontier_size", 2),
            dilate_obstacle_size=parameters.get("motion_planner/frontier/dilate_obstacle_size", 0),
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
        obs = self.robot.get_observation(wait_for_new=True)
        # Since the pose graph only contains the poses of the base_in_map_pose, 
        # the camera_in_base_pose need to be stored separately for subsequent pose updates.
        (
            rgb,
            depth,
            K,
            camera_pose,
            base_pose,
            local_tf,
            node_id,
        ) = (
            obs.rgb,
            obs.depth,
            obs.camera_K,
            obs.camera_in_map_pose,
            obs.base_in_map_pose,
            obs.camera_in_base_pose,
            obs.node_id,
        )

        self.voxel_map.process_rgbd_images(
            rgb=rgb, 
            depth=depth, 
            intrinsics=K, 
            camera_pose=camera_pose,
            base_pose=base_pose, 
            local_tf=local_tf, 
            node_id=node_id,
        )
        if visualize_map:
            if self.voxel_map.voxel_pointcloud._points is not None and \
                    len(self.voxel_map.voxel_pointcloud._points) != 0:
                self.rerun_visualizer.update_voxel_map(space=self.space)
            if self.voxel_map.semantic_memory._points is not None and \
                    len(self.voxel_map.semantic_memory._points) != 0:
                self.rerun_visualizer.log_custom_pointcloud(
                    "world/semantic_memory/pointcloud",
                    self.voxel_map.semantic_memory._points.detach().cpu(),
                    self.voxel_map.semantic_memory._rgb.detach().cpu() / 255.0,
                    0.03,
                )

    def look_around(self):
        print("*" * 10, "Look around to check", "*" * 10)
        # for pan in [0.6, -0.2, -1.0, -1.8]:
        #     tilt = -0.6
        for angle in [
            constants.look_ahead, constants.look_front, constants.look_down, 
            constants.look_left_1, constants.look_left_2, 
            constants.look_right_1, constants.look_right_2
        ]:
            self.robot.arm_to(angle=angle, blocking=True)
            if not self._realtime_updates:
                self.update()

    def rotate_in_place(self):
        print("*" * 10, "Rotate in place", "*" * 10)
        xyt = self.robot.get_base_in_map_xyt()
        # self.robot.arm_to(head_pan=0, head_tilt=-0.6, blocking=True)
        for i in range(0):  # TODO range(8)
            xyt[2] += 2 * np.pi / 8
            self.robot.base_to(xyt, blocking=True)
            if not self._realtime_updates:
                self.update()

    def execute_action(
        self,
        text: str,
    ):
        if not self._realtime_updates:
            self.robot.look_front()
            self.look_around()
            self.robot.look_front()
            self.robot.switch_to_navigation_mode()

        self.robot.switch_to_navigation_mode()

        start = self.robot.get_base_in_map_xyt()
        res = self.process_text(text, start)
        if len(res) == 0 and text != "" and text is not None:
            res = self.process_text("", start)

        if len(res) > 0:
            print("Plan successful!")
            if len(res) >= 2 and np.isnan(res[-2]).all():
                if len(res) > 2:
                    self.robot.execute_trajectory(
                        res[:-2],
                        pos_err_threshold=self.pos_err_threshold,
                        rot_err_threshold=self.rot_err_threshold,
                        blocking=True,
                    )

                return True, res[-1]
            else:
                # print(res)
                # res[-1][2] += np.pi / 2
                self.robot.execute_trajectory(
                    res,
                    pos_err_threshold=self.pos_err_threshold,
                    rot_err_threshold=self.rot_err_threshold,
                    blocking=True,
                )
                return False, None
        else:
            print("Failed. Try again!")
            return None, None

    def run_exploration(self):
        """Go through exploration. We use the voxel_grid map created by our collector to sample free space, and then use our motion planner (RRT for now) to get there. At the end, we plan back to (0,0,0).

        Args:
            visualize(bool): true if we should do intermediate debug visualizations"""
        status, _ = self.execute_action("")
        if status is None:
            print("Exploration failed! Perhaps nowhere to explore!")
            return False
        return True

    def process_text(self, text, start_pose):
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

        if text is not None and text != "" and self.space.traj is not None:
            print("saved traj", self.space.traj)
            traj_target_point = self.space.traj[-1]
            with self._voxel_map_lock:
                if self.voxel_map.verify_point(text, traj_target_point):
                    localized_point = traj_target_point
                    debug_text += "## Last visual grounding results looks fine so directly use it.\n"

        print("Target verification finished")

        if text is not None and text != "" and localized_point is None:
            with self._voxel_map_lock:
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
            with self._voxel_map_lock:
                print(obs, len(self.voxel_map.observations))
                obs = self.voxel_map.find_obs_id_for_text(text)
                rgb = self.voxel_map.observations[obs - 1].rgb
            self.rerun_visualizer.log_custom_2d_image("/observation_similar_to_text", rgb)

        if localized_point is None:
            return []

        # TODO: Do we really need this line?
        if len(localized_point) == 2:
            localized_point = np.array([localized_point[0], localized_point[1], 0])

        point = self.space.sample_navigation(
            start=start_pose, 
            point=localized_point, 
            planner=self.planner
        )

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
        # If we are navigating to some object of interest, send (x, y, z) of
        # the object so that we can make sure the robot looks at the object after navigation
        traj = []
        if waypoints is not None:

            self.rerun_visualizer.log_custom_pointcloud(
                "world/object",
                [localized_point[0], localized_point[1], 1.5],
                torch.Tensor([0, 1, 0]),
                0.1,
            )

            finished = len(waypoints) <= 8 and mode == "navigation"
            if finished:
                self.space.traj = None
            else:
                self.space.traj = waypoints[8:] + [[np.nan, np.nan, np.nan], localized_point]
            if not finished:
                waypoints = waypoints[:8]
            traj = self.planner.clean_path_for_xy(waypoints)
            if finished:
                traj.append([np.nan, np.nan, np.nan])
                if isinstance(localized_point, torch.Tensor):
                    localized_point = localized_point.tolist()
                traj.append(localized_point)
            print("Planned trajectory:", traj)

        # Talk about what you are doing, as the robot.
        if self.robot is not None:
            if text is not None and text != "":
                self.robot.say("I am looking for a " + text + ".")
            else:
                self.robot.say("I am exploring the environment.")

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
                    origins.append([traj[idx][0], traj[idx][1], 1.5])
                    vectors.append(
                        [traj[idx + 1][0] - traj[idx][0], traj[idx + 1][1] - traj[idx][1], 0]
                    )
            self.rerun_visualizer.log_arrow3D(
                "world/direction", origins, vectors, torch.Tensor([0, 1, 0]), 0.1
            )
            self.rerun_visualizer.log_custom_pointcloud(
                "world/robot_start_pose",
                [start_pose[0], start_pose[1], 1.5],
                torch.Tensor([0, 0, 1]),
                0.1,
            )

        return traj


    def get_voxel_map(self):
        """Return the voxel map"""
        return self.voxel_map
    

    def navigate(self, text, max_step=10):
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
            success = place(
                socket=self.manip_socket,
                manip_wrapper=self.manip_wrapper,
                back_object=back_object,
                translation=translation,
            )
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
        target_point: None,
        skip_confirmation: bool = False,
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

        rotation, translation, depth, width, obj_points, theta_cumulative = capture_and_process_image(
            mode="pick",
            obj=target_object,
            tar_in_map=target_point,
            socket=self.manip_socket,
            manip_wrapper=self.manip_wrapper,
        )

        if rotation is None:
            print("(ಥ﹏ಥ) Try all pose but anygrasp is failed.")
            return False
        
        if skip_confirmation or input("Do you want to do this pickup manipulation? Y or N ") != "N":
            success = pickup(
                self.manip_wrapper,
                rotation,
                translation,
                object_points=obj_points,
            )
            if not success:
                print("(ಥ﹏ಥ) Pickup task failed.")
                return False

        # Shift the base back to the original point as we are certain that original point is navigable in navigation obstacle map
        if theta_cumulative != 0:
            self.manip_wrapper.move_to_position(base_theta=np.deg2rad(-theta_cumulative))

        return True
