# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

import click
import traceback
from dream.agent.task.dream import DreamTaskExecutor
from dream.agent.zmq_client import RobotZmqClient
from dream.core.parameters import get_parameters
from dream.llms import get_llm_choices, get_llm_client
from dream.motion import constants

@click.command()
# by default you are running these codes on your workstation, not on your robot.
@click.option("--server_ip", "--server-ip", default="127.0.0.1", type=str)
@click.option("--manual-wait", default=False, is_flag=True)
@click.option("--random-goals", default=False, is_flag=True)
@click.option("--explore-iter", default=3)
@click.option("--method", default="dream", type=str)
@click.option("--mode", default="", type=click.Choice(["navigation", "manipulation", "save", ""]))

@click.option(
    "--llm",
    default="openai",
    help="Client to use for language model.",
    type=click.Choice(get_llm_choices()),
)
@click.option("--debug_llm", "--debug-llm", is_flag=True, help="Set to debug the language model")
@click.option(
    "--use_voice",
    "--use-voice",
    is_flag=True,
    help="Set to use voice input",
)
@click.option(
    "--visual_servo",
    "--vs",
    "-V",
    "--visual-servo",
    default=False,
    is_flag=True,
    help="Use visual servoing grasp",
)
@click.option(
    "--robot_ip", type=str, default="", help="Robot IP address (leave empty for saved default)"
)
@click.option("--target_object", type=str, default=None, help="Target object to grasp")
@click.option(
    "--target_receptacle", "--receptacle", type=str, default=None, help="Target receptacle to place"
)
@click.option(
    "--skip_confirmations",
    "--skip",
    "-S",
    "-y",
    "--yes",
    is_flag=True,
    help="Skip many confirmations",
)
@click.option(
    "--input-path",
    type=click.Path(),
    default=None,
    help="Input path with default value None",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=None,
    help="Input path with default value None",
)
@click.option(
    "--match-method",
    "--match_method",
    type=click.Choice(["class", "feature"]),
    default="class",
    help="match method for visual servoing",
)
@click.option("--device_id", default=0, type=int, help="Device ID for semantic sensor")
def main(
    server_ip,
    manual_wait,
    explore_iter: int = 3,
    mode: str = "navigation",
    match_method: str = "class",
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    robot_ip: str = "",
    visual_servo: bool = False,
    skip_confirmations: bool = True,
    device_id: int = 0,
    target_object: str = None,
    target_receptacle: str = None,
    use_voice: bool = False,
    debug_llm: bool = False,
    llm: str = "qwen25-3B-Instruct",
    **kwargs,
):
    """
    Including only some selected arguments here.

    Args:
        random_goals(bool): randomly sample frontier goals instead of looking for closest
    """

    print("- Load parameters")
    parameters = get_parameters("dream_config.yaml")

    print("- Create robot client")
    robot = RobotZmqClient(
        robot_ip=robot_ip,
        parameters=parameters,
        output_path=output_path,
    )

    print("- Create task executor")
    executor = DreamTaskExecutor(
        robot,
        parameters,
        match_method=match_method,
        device_id=device_id,
        output_path=output_path,
        server_ip=server_ip,
        skip_confirmations=skip_confirmations,
    )

    if False:
        # command = [("pause_slam", "")]
        # executor(command)
        # command = [("resume_slam", "")]
        # executor(command)
        # command = [("look_around", "")]
        # executor(command)
        # command = [("base_to_relative", [0.8, 0, 0])]
        # executor(command)
        # command = [("base_to_relative", [-0.8, 0, 0])]
        # executor(command)
        # command = [("base_to_relative", [0, 0, 3.14 / 2])]
        # executor(command)

        # command = [("find", "red pepper")]
        # executor(command)
        # import time
        # while True:
        #     command = [("pickup_only", "corn")]
        #     executor(command)
        #     time.sleep(10)

        # command = [("place_only", "green bowl")]
        # executor(command)
        import time
        while True:
            time.sleep(0.1)



    if input_path is None:
        start_command = [("rotate_in_place", "")]
    else:
        start_command = [("read_from_pickle", input_path)]
    executor(start_command)



    # Parse things and listen to the user
    ok = True
    while ok:
        try:
            # Call the LLM client and parse
            explore = input(
                "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place)]: "
            )
            if explore.upper() == "E":
                llm_response = [("explore", None)]
            else:
                if target_object is None or len(target_object) == 0:
                    target_object = input("Enter the target object: ")
                if target_receptacle is None or len(target_receptacle) == 0:
                    target_receptacle = input("Enter the target receptacle: ")
                llm_response = [("pickup", target_object), ("place", target_receptacle)]


            ok = executor(llm_response)
            target_object = None
            target_receptacle = None
        except Exception as e:
            print(e)
            traceback.print_exc()
            ok = False

    print("Stopping robot...")
    robot.stop()
    print("Robot stopped successfully.")


if __name__ == "__main__":
    main()
