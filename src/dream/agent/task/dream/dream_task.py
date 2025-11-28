# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from dream.agent.robot_agent import RobotAgent
from dream.core import AbstractRobotClient, Parameters
from dream.agent import RobotZmqClient
from dream.perception import create_semantic_sensor
from dream.utils.image import numpy_image_to_bytes

# Mapping and perception
from dream.utils.logger import Logger

logger = Logger(__name__)


def compute_tilt(camera_xyz, target_xyz):
    """
    a util function for computing robot head tilts so the robot can look at the target object after navigation
    - camera_xyz: estimated (x, y, z) coordinates of camera
    - target_xyz: estimated (x, y, z) coordinates of the target object
    """
    if not isinstance(camera_xyz, np.ndarray):
        camera_xyz = np.array(camera_xyz)
    if not isinstance(target_xyz, np.ndarray):
        target_xyz = np.array(target_xyz)
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))


class DreamTaskExecutor:
    def __init__(
        self,
        robot: AbstractRobotClient | RobotZmqClient,
        parameters: Parameters,
        match_method: str = "feature",
        device_id: int = 0,
        output_path: Optional[str] = None,
        server_ip: Optional[str] = "127.0.0.1",
        skip_confirmations: bool = True,
        explore_iter: int = 5,
        back_object:str=None
    ) -> None:
        """Initialize the executor."""
        self.robot = robot
        self.parameters = parameters

        # Other parameters
        self.match_method = match_method
        self.skip_confirmations = skip_confirmations
        self.explore_iter = explore_iter

        # Do type checks
        if not isinstance(self.robot, AbstractRobotClient):
            raise TypeError(f"Expected AbstractRobotClient, got {type(self.robot)}")

        self.parameters["encoder"] = None
        self.semantic_sensor = None

        self.back_object = back_object  # object has been pickup and place to back

        print("- Start robot agent with data collection")
        self.agent = RobotAgent(
            self.robot,
            self.parameters,
            self.semantic_sensor,
            log=output_path,
            server_ip=server_ip,
        )
        self.agent.start()

    def _find(self, target_object: str) -> np.ndarray:
        """Find an object. This is a helper function for the main loop.

        Args:
            target_object: The object to find.

        Returns:
            The point where the object is located.
        """
        self.robot.switch_to_navigation_mode()
        point = self.agent.navigate(target_object)
        # `filename` = None means write to default log path (the datetime you started to run the process)
        self.agent.voxel_map.write_to_pickle(filename=None)
        if point is None:
            logger.error("Navigation Failure: Could not find the object {}".format(target_object))
            return None
        cv2.imwrite(self.agent.voxel_map.log + "/" + target_object + ".jpg", self.robot.get_observation().rgb[:, :, [2, 1, 0]])
        return point

    def _pickup(
        self,
        target_object: str,
        point: Optional[np.ndarray] = None,
        skip_confirmations: bool = False,
    ) -> None:
        """Pick up an object."""
        self.robot.switch_to_manipulation_mode()
        print("Using self.agent to grasp object:", target_object)
        success = self.agent.manipulate(target_object=target_object, target_point=point, skip_confirmation=skip_confirmations)
        if not success:
            self.robot.open_gripper()
        else:
            self.back_object = target_object
        self.robot.look_front()


    def _place(
        self,
        target_receptacle: str, 
        point: Optional[np.ndarray]=None, 
        skip_confirmations: bool = False
    ) -> None:
        """Place an object."""
        self.robot.switch_to_manipulation_mode()
        assert self.back_object, "back must have object."
        self.agent.place(
            back_object=self.back_object, 
            target_receptacle=target_receptacle, 
            target_point=point, 
            skip_confirmation=skip_confirmations
        )
        self.robot.look_front()


    def __call__(self, response: List[Tuple[str, str]], channel=None) -> bool:
        """Execute the list of commands given by the LLM bot.

        Args:
            response: A list of tuples, where the first element is the command and the second is the argument.

        Returns:
            True if we should keep going, False if we should stop.
        """
        i = 0

        if response is None or len(response) == 0:
            logger.error("No commands to execute!")
            return True

        # Dynamem aims to life long robot, we should not reset the robot's memory.
        # logger.info("Resetting agent...")
        # self.agent.reset()

        # Loop over every command we have been given
        # Pull out pickup and place as a single arg if they are in a row
        # Else, execute things as they come
        while i < len(response):
            command, args = response[i]
            logger.info(f"Command: {i} {command} {args}")
            if command == "pause_slam":
                logger.info(f"[Pause SLAM]")
                self.robot.pause_slam()
            elif command == "resume_slam":
                logger.info(f"[Resume SLAM]")
                self.robot.resume_slam()
            elif command == "look_around":
                logger.info(f"look_around")
                self.agent.look_around()
            elif command == "base_to_relative":
                logger.info(f"BASE TO {args}")
                self.robot.base_to(
                    xyt=np.array(args),
                    relative=True,
                    reliable=True
                )
            elif command == "pickup_only":
                logger.info(f"[Pickup task] Pickup: {args}")
                target_object = args
                # Pick up
                self._pickup(target_object, skip_confirmations=self.skip_confirmations)
            elif command == "place_only":
                logger.info(f"[Pickup task] Place: {args}")
                target_object = args
                # Placing
                self._place(target_object, skip_confirmations=self.skip_confirmations)
 
            elif command == "pickup":
                logger.info(f"[Pickup task] Pickup: {args}")
                target_object = args
                next_command, next_args = response[i]

                # Navigation

                # Either we wait for users to confirm whether to run navigation, or we just directly control the robot to navigate.
                if self.skip_confirmations or (
                    not self.skip_confirmations
                    and input("Do you want to run navigation? [Y/n]: ").upper() != "N"
                ):
                    self.robot.move_to_nav_posture()
                    point = self._find(args)
                # Or the user explicitly tells that he or she does not want to run navigation.
                else:
                    point = None

                # Pick up
                if self.skip_confirmations:
                    if point is not None:
                        self._pickup(target_object, point=point, skip_confirmations=self.skip_confirmations)
                    else:
                        logger.error("Could not find the object.")
                        i += 1
                        continue
                else:
                    if input("Do you want to run picking? [Y/n]: ").upper() != "N":
                        self._pickup(target_object, point=point)
                    else:
                        logger.info("Skip picking!")
                        i += 1
                        continue

            elif command == "place":
                logger.info(f"[Pickup task] Place: {args}")
                target_object = args
                next_command, next_args = response[i]

                # Navigation

                # Either we wait for users to confirm whether to run navigation, or we just directly control the robot to navigate.
                if self.skip_confirmations or (
                    not self.skip_confirmations
                    and input("Do you want to run navigation? [Y/n]: ").upper() != "N"
                ):
                    point = self._find(args)
                # Or the user explicitly tells that he or she does not want to run navigation.
                else:
                    point = None

                # Placing

                if self.skip_confirmations:
                    if point is not None:
                        self._place(target_object, point=point, skip_confirmations=self.skip_confirmations)
                    else:
                        logger.error("Could not find the object.")
                        i += 1
                        continue
                else:
                    if input("Do you want to run placement? [Y/n]: ").upper() != "N":
                        self._place(target_object, point=point)
                    else:
                        logger.info("Skip placing!")
                        i += 1
                        continue
            elif command == "hand_over":
                self._hand_over()
            elif command == "rotate_in_place":
                logger.info("Rotate in place to scan environments.")
                self.agent.rotate_in_place()
                # `filename` = None means write to default log path (the datetime you started to run the process)
                self.agent.voxel_map.write_to_pickle(filename=None)
            elif command == "read_from_pickle":
                logger.info(f"Load the semantic memory from past runs, pickle file name: {args}.")
                self.agent.voxel_map.read_from_pickle(args)
            elif command == "go_home":
                logger.info("[Pickup task] Going home.")
                if self.agent.get_voxel_map().is_empty():
                    logger.warning("No map data available. Cannot go home.")
                else:
                    self.agent.go_home()
            elif command == "explore":
                logger.info("[Pickup task] Exploring.")
                for _ in range(self.explore_iter):
                    self.agent.run_exploration()
            elif command == "find":
                logger.info("[Pickup task] Finding {}.".format(args))
                point = self._find(args)
            elif command == "quit":
                logger.info("[Pickup task] Quitting.")
                self.robot.stop()
                return False
            elif command == "end":
                logger.info("[Pickup task] Ending.")
                break
            else:
                logger.error(f"Skipping unknown command: {command}")

            i += 1
        # If we did not explicitly receive a quit command, we are not yet done.
        return True
