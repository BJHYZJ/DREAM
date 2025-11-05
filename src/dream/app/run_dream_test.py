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

from dream.agent.task.dream import DreamTaskExecutor
from dream.agent.zmq_client_dream import DreamRobotZmqClient
from dream.core.parameters import get_parameters
from dream.llms import LLMChatWrapper, PickupPromptBuilder, get_llm_choices, get_llm_client
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
    "--use_llm",
    "--use-llm",
    is_flag=True,
    help="Set to use the language model",
)
@click.option(
    "--llm",
    # default="gemma2b",
    default="qwen25-3B-Instruct",
    help="Client to use for language model. Recommended: gemma2b, openai",
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
@click.option(
    "--mllm-for-visual-grounding",
    "--mllm",
    "-M",
    is_flag=True,
    help="Use GPT4o for visual grounding",
)
@click.option("--device_id", default=0, type=int, help="Device ID for semantic sensor")
@click.option(
    "--manipulation-only", "--manipulation", is_flag=True, help="For debugging manipulation"
)
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
    use_llm: bool = False,
    use_voice: bool = False,
    debug_llm: bool = False,
    llm: str = "qwen25-3B-Instruct",
    manipulation_only: bool = False,
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
    robot = DreamRobotZmqClient(
        robot_ip=robot_ip,
        parameters=parameters,
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
        mllm=kwargs["mllm_for_visual_grounding"],
        manipulation_only=manipulation_only,
    )

    # if not manipulation_only:
    #     if input_path is None:
    #         start_command = [("rotate_in_place", "")]
    #     else:
    #         start_command = [("read_from_pickle", input_path)]
    #     executor(start_command)

    # Create the prompt we will use to control the robot
    prompt = PickupPromptBuilder()

    # Get the LLM client
    llm_client = None
    if use_llm:
        llm_client = get_llm_client(llm, prompt=prompt)
        chat_wrapper = LLMChatWrapper(llm_client, prompt=prompt, voice=use_voice)

    # Parse things and listen to the user
    ok = True

    while ok:
        # say_this = None
        # if llm_client is None:
        #     # Call the LLM client and parse
        #     explore = input(
        #         "Enter desired mode [E (explore and mapping) / M (Open vocabulary pick and place)]: "
        #     )
        #     if explore.upper() == "E":
        #         llm_response = [("explore", None)]
        #     else:
        #         if target_object is None or len(target_object) == 0:
        #             target_object = input("Enter the target object: ")
        #         if target_receptacle is None or len(target_receptacle) == 0:
        #             target_receptacle = input("Enter the target receptacle: ")
        #         llm_response = [("pickup", target_object), ("place", target_receptacle)]
        # else:
        #     # Call the LLM client and parse
        #     llm_response = chat_wrapper.query(verbose=debug_llm)
        #     if debug_llm:
        #         print("Parsed LLM Response:", llm_response)

        # llm_response = [("pickup", 'eggplant')]
        llm_response = [("place", 'chair')]
        executor.back_object = "eggplant"
        ok = executor(llm_response)
        target_object = None
        target_receptacle = None

    print("Stopping robot...")
    robot.stop()
    print("Robot stopped successfully.")


if __name__ == "__main__":
    main()
