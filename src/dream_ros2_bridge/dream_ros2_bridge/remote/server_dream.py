#!/usr/bin/env python
# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

# (c) 2024 Hello Robot, under MIT license

import time
from typing import Any, Dict
import threading
import os
import click
import numpy as np
import rclpy
from overrides import override

import dream.utils.compression as compression
import dream.utils.logger as logger
from dream.core.server import BaseZmqServer
from dream.utils.image import adjust_gamma, scale_camera_matrix
from dream_ros2_bridge.remote import DreamClient
# from dream_ros2_bridge.ros.map_saver import MapSerializerDeserializer


class ZmqServer(BaseZmqServer):
    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ROS2 client interface
        self.client = DreamClient()
        # self.use_d405 = use_d405

        # Check if the robot is homed
        if not self.client.is_homed:
            raise RuntimeError("Robot is not homed. Please home the robot first.")

        # Check if the robot is runstopped
        if self.client.is_runstopped:
            raise RuntimeError("Robot is runstopped. Please unstop the robot first.")

        # Map saver - write and load map information from SLAM
        # self.map_saver = MapSerializerDeserializer()

    def shutdown(self):
        """Shutdown the server and clean up resources"""
        print("Shutting down server...")
        
        # Shutdown the client (ROS interface)
        if hasattr(self, 'client') and hasattr(self.client, 'shutdown'):
            self.client.shutdown()
        
        # Call parent shutdown if it exists
        if hasattr(super(), 'shutdown'):
            super().shutdown()
        
        print("Server shutdown complete")

    @override
    def is_running(self) -> bool:
        return self.is_running and rclpy.ok()

    @override
    def get_control_mode(self) -> str:
        """Get the current control mode of the robot. Can be navigation, manipulation, or none.

        Returns:
            str: The current control mode of the robot.
        """
        if self.client.in_manipulation_mode():
            control_mode = "manipulation"
        elif self.client.in_navigation_mode():
            control_mode = "navigation"
        else:
            control_mode = "none"
        return control_mode

    @override
    def get_full_observation_message(self) -> Dict[str, Any]:
        # get information
        # Still about 0.01 seconds to get observations
        obs = self.client.get_full_observation()
        if obs is None:
            return None
        data = {
            "timestamp": obs.timestamp,
            "compass": obs.compass,
            "gps": obs.gps,
            "node_id": obs.node_id,
            "is_history_node": obs.is_history_node,
            "rgb": obs.rgb_compressed,
            "depth": obs.depth_compressed,
            "camera_K": obs.camera_K,
            "pose_graph": obs.pose_graph,
            "base_in_map_pose": obs.base_in_map_pose,
            "camera_in_base_pose": obs.camera_in_base_pose,
            "recv_address": self.recv_address,
            "step": self._last_step,
            "at_goal": self.client.at_goal(),
        }
        return data

    @override
    def get_state_message(self) -> Dict[str, Any]:
        obs = self.client.get_state_observation()
        if obs is None:
            return None
        # """Get the state message for the robot."""
        # q, dq, eff = self.client.get_joint_state()
        message = {
            "gps": obs.gps,
            "compass": obs.compass,
            "base_in_map_pose": obs.base_in_map_pose,
            "arm_base_in_map_pose": obs.arm_base_in_map_pose,
            "camera_in_arm_base_pose": obs.camera_in_arm_base_pose,
            "camera_in_base_pose": obs.camera_in_base_pose,
            "camera_in_map_pose": obs.camera_in_map_pose,
            "ee_in_arm_base_pose": obs.ee_in_arm_base_pose,
            "ee_in_base_pose": obs.ee_in_base_pose,
            "ee_in_map_pose": obs.ee_in_map_pose,
            "joint_states": obs.joint_states,
            "joint_velocities": obs.joint_velocities,
            "joint_forces": obs.joint_forces,
            "joint_positions": obs.joint_positions,
            "at_goal": obs.at_goal,
            "is_homed": obs.is_homed,
            "is_runstopped": obs.is_runstopped,
            "control_mode": self.get_control_mode(),
            "step": self._last_step,
        }
        return message


    def get_servo_message(self) -> Dict[str, Any]:
        obs = self.client.get_servo_observation()
        if obs is None:
            return None
        color_image, depth_image = self._rescale_color_and_depth(
            obs.rgb, obs.depth, self.image_scaling
        )
        depth_image = (depth_image * 1000).astype(np.uint16)
        compressed_color_image = compression.to_jpg(color_image)
        compressed_depth_image = compression.to_jp2(depth_image)
        
        message = {
            "rgb": compressed_color_image,
            "depth": compressed_depth_image,
            "camera_K": scale_camera_matrix(
                self.client.rgb_cam.get_K(), self.image_scaling
            ),
            "depth_K": scale_camera_matrix(
                self.client.dpt_cam.get_K(), self.image_scaling
            ),
            "image_scaling": self.image_scaling,
            "depth_scaling": self.depth_scaling,
            "color_shape": color_image.shape,
            "depth_shape": depth_image.shape,
            "step": self._last_step,
        }
        # message.update(d405_output)
        return message
    

    @override
    def handle_action(self, action: Dict[str, Any]):
        """Handle an action from the client."""

        if "posture" in action:
            if action["posture"] == "manipulation":
                self.client.stop()
                self.client.switch_to_busy_mode()
                self.client.move_to_manip_posture()
                self.client.switch_to_manipulation_mode()
            elif action["posture"] == "navigation":
                self.client.stop()
                self.client.switch_to_busy_mode()
                self.client.move_to_nav_posture()
                self.client.switch_to_navigation_mode()
            else:
                print(
                    " - posture",
                    action["posture"],
                    "not recognized or supported.",
                )
        elif "control_mode" in action:
            if action["control_mode"] == "manipulation":
                self.client.switch_to_manipulation_mode()
                self.control_mode = "manipulation"
            elif action["control_mode"] == "navigation":
                self.client.switch_to_navigation_mode()
                self.control_mode = "navigation"
            else:
                print(
                    " - control mode",
                    action["control_mode"],
                    "not recognized or supported.",
                )
        elif "save_map" in action:
            self.client.save_map(action["save_map"])
        elif "load_map" in action:
            self.client.load_map(action["load_map"])
        elif "say" in action:
            # Text to speech from the robot, not the client/agent device
            print("Saying:", action["say"])
            # self.text_to_speech.say_async(action["say"])
        elif "say_sync" in action:
            print("Saying:", action["say_sync"])
            # self.text_to_speech.say(action["say_sync"])
        elif "xyt" in action:
            # Check control mode
            if not self.client.in_navigation_mode():
                self.client.switch_to_navigation_mode()
            if self.verbose:
                print(
                    "Is robot in navigation mode?",
                    self.client.in_navigation_mode(),
                )
                print(f"{action['xyt']} {action['nav_relative']} {action['nav_blocking']}")
            self.client.base_to(
                action["xyt"],
                relative=action["nav_relative"],
            )
        elif "base_velocity" in action:
            base_velocity_action = action["base_velocity"]
            if self.verbose:
                print(
                    f"Setting base velocity to translation={base_velocity_action['v']} and rotation={base_velocity_action['w']}"
                )
            if "v" in base_velocity_action:
                v = base_velocity_action["v"]
            if "w" in base_velocity_action:
                w = action["base_velocity"]["w"]
            self.client.nav.set_velocity(v, w)

        elif action.get("slam_pause"):
            timeout = action.get("slam_timeout", 2.0)
            success = self.client.pause_slam(timeout=timeout)
            if self.verbose:
                print(f"Pausing SLAM (timeout={timeout}) -> {success}")
            if not success:
                logger.warning("Failed to pause SLAM via RTAB-Map service.")

        elif action.get("slam_resume"):
            timeout = action.get("slam_timeout", 2.0)
            success = self.client.resume_slam(timeout=timeout)
            if self.verbose:
                print(f"Resuming SLAM (timeout={timeout}) -> {success}")
            if not success:
                logger.warning("Failed to resume SLAM via RTAB-Map service.")

        # elif "joint" in action:
        #     # This allows for executing motor commands on the robot relatively quickly
        #     if self.verbose:
        #         print(f"Moving arm to config={action['joint']}")
        #     if "gripper" in action:
        #         gripper_cmd = action["gripper"]
        #     else:
        #         gripper_cmd = None
        #     if "target_point" in action:
        #         target_point = action["target_point"]
        #     else:
        #         target_point = None

        #     # I found currently the blocking in arm to does not
        #     # serve any actual purpose so maybe we should use this line instead

        #     # _is_blocking = action.get("blocking", False) or action.get("manip_blocking", False)
        #     _is_blocking = action.get("blocking", False)

        #     # Now send all command fields here
        #     self.client.arm_to(
        #         action["joint"],
        #         gripper=gripper_cmd,
        #         target_point=target_point,
        #         blocking=_is_blocking,
        #     )
        elif "move_to_positions" in action:
            if self.verbose:
                print(f"Moving to positions {action['move_to_positions']}")
            if not self.client.in_navigation_mode():
                self.client.switch_to_manipulation_mode()
            _is_wait = action.get("wait", False)
            self.client.move_to_positions(
                action["move_to_positions"],
                wait=_is_wait,
            )
        elif "servo_angle" in action:
            # This will send head without anything else
            if self.verbose or True:
                print(f"Moving head to {action['servo_angle']}")
            if not self.client.in_manipulation_mode():
                self.client.switch_to_navigation_mode()
            self.client.manip.set_servo_angle(
                angle=action["servo_angle"],
                wait=action['wait'],
            )
        elif "gripper" in action:
            if self.verbose or True:
                print(f"Moving gripper to {action['gripper']}")
            self.client.manip.set_gripper(action["gripper"], wait=action['wait'])
        else:
            logger.warning(" - action not recognized or supported.")
            logger.warning(action)



@click.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--send_port", default=4401, help="Port to send observations to")
@click.option("--recv_port", default=4402, help="Port to receive actions from")
@click.option("--local", is_flag=True, help="Run code locally on the robot.")
def main(
    send_port: int = 4401,
    recv_port: int = 4402,
    local: bool = False,
):
    # try:
    rclpy.init()
    server = ZmqServer(
        send_port=send_port,
        recv_port=recv_port,
        use_remote_computer=(not local),
    )
    server.start()
    try:
        while rclpy.ok() and server.running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    finally:
        server.shutdown()
            
    # except Exception as e:
    #     print(f"Error in main: {e}")
    # finally:
    #     if rclpy.ok():
    #         rclpy.shutdown()


if __name__ == "__main__":
    main()
