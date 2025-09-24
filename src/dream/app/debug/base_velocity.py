#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time

import click

from dream.agent.zmq_client import HomeRobotZmqClient
from dream.core import get_parameters


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--parameter_file", default="default_planner.yaml", help="Path to parameter file")
def main(
    robot_ip: str = "192.168.1.15",
    local: bool = False,
    parameter_file: str = "config/default_planner.yaml",
):
    """Set up the robot and send it to home (0, 0, 0)."""
    parameters = get_parameters(parameter_file)
    robot = HomeRobotZmqClient(
        robot_ip=robot_ip,
        use_remote_computer=(not local),
        parameters=parameters,
        enable_rerun_server=False,
    )
    robot.set_base_velocity(forward=0.1, rotational=0.0)
    time.sleep(1.0)
    robot.set_base_velocity(forward=-0.1, rotational=0.0)
    time.sleep(1.0)
    robot.set_base_velocity(forward=0.0, rotational=0.0)

    robot.stop()


if __name__ == "__main__":
    main()
