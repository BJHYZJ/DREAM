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
        # rgb, depth = obs.rgb, obs.depth
        # width, height = rgb.shape[:2]

        # # Convert depth into int format
        # depth = (depth * 1000).astype(np.uint16)

        # # Make both into jpegs
        # rgb = compression.to_jpg(rgb)
        # depth = compression.to_jp2(depth)

        # Get the other fields from an observation
        # rgb = compression.to_webp(rgb)
        data = {
            # "rgb": obs.rgb,
            # "depth": obs.depth,
            # "gps": obs.gps,
            # "compass": obs.compass,
            # "camera_pose_in_map": obs.camera_pose_in_map.matrix(),
            # "camera_pose_in_arm": obs.camera_pose_in_arm.matrix(),
            # "camera_pose_in_base": obs.camera_pose_in_base.matrix(),
            # "joint": obs.joint,
            # "joint_velocities": obs.joint_velocities,
            # "camera_K": obs.camera_K.cpu().numpy(),
            # "ee_pose_in_map": obs.ee_pose_in_map.matrix(),
            # "rgb_width": width,
            # "rgb_height": height,
            # "lidar_points": obs.lidar_points,
            # "lidar_timestamp": obs.lidar_timestamp,
            # "pose_graph": self.client.get_pose_graph(),

            "timestamp": obs.timestamp,
            "compass": obs.compass,
            "gps": obs.gps,
            "node_id": obs.node_id,
            "rgb": obs.rgb_compressed,
            "depth": obs.depth_compressed,
            "lidar_points": obs.laser_compressed,
            "camera_K": obs.camera_K,
            "pose_graph": obs.pose_graph,
            "base_in_map_pose": obs.base_in_map_pose,
            "camera_in_map_pose": obs.camera_in_map_pose,


            # "last_motion_failed": self.client.last_motion_failed(),
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
            "ee_in_map_pose": obs.ee_in_map_pose,
            "joint_positions": obs.joint_positions,
            "joint_velocities": obs.joint_velocities,
            "joint_efforts": obs.joint_efforts,
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
            "ee_in_map_pose": obs.ee_in_map_pose,
            "camera_in_map_pose": obs.camera_in_map_pose,
            "camera_K": scale_camera_matrix(
                self.client.rgb_cam.get_K(), self.image_scaling
            ),
            "depth_K": scale_camera_matrix(
                self.client.dpt_cam.get_K(), self.image_scaling
            ),
            "color_image": compressed_color_image,
            "depth_image": compressed_depth_image,
            "color_shape": color_image.shape,
            "depth_shape": depth_image.shape,
            "image_scaling": self.image_scaling,
            "depth_scaling": self.depth_scaling,
            "joint_positions": obs.joint_positions,
            "joint_velocities": obs.joint_velocities,
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
            self.text_to_speech.say_async(action["say"])
        elif "say_sync" in action:
            print("Saying:", action["say_sync"])
            self.text_to_speech.say(action["say_sync"])
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
            self.client.move_base_to(
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
        elif "joint" in action:
            # This allows for executing motor commands on the robot relatively quickly
            if self.verbose:
                print(f"Moving arm to config={action['joint']}")
            if "gripper" in action:
                gripper_cmd = action["gripper"]
            else:
                gripper_cmd = None
            if "head_to" in action:
                head_pan_cmd, head_tilt_cmd = action["head_to"]
            else:
                head_pan_cmd, head_tilt_cmd = None, None

            # I found currently the blocking in arm to does not
            # serve any actual purpose so maybe we should use this line instead

            # _is_blocking = action.get("blocking", False) or action.get("manip_blocking", False)
            _is_blocking = action.get("blocking", False)

            # Now send all command fields here
            self.client.arm_to(
                action["joint"],
                gripper=gripper_cmd,
                head_pan=head_pan_cmd,
                head_tilt=head_tilt_cmd,
                blocking=_is_blocking,
            )
        elif "head_to" in action:
            # This will send head without anything else
            if self.verbose or True:
                print(f"Moving head to {action['head_to']}")
            self.client.head_to(
                action["head_to"][0],
                action["head_to"][1],
                blocking=True,
            )
        elif "gripper" in action and "joint" not in action:
            if self.verbose or True:
                print(f"Moving gripper to {action['gripper']}")
            self.client.manip.move_gripper(action["gripper"])
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
    try:
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
            
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
