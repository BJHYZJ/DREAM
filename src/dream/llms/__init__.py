# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.
from typing import Union

from .base import AbstractLLMClient, AbstractPromptBuilder
from .openai_client import OpenaiClient

# This is a list of all the modules that are imported when you use the import * syntax.
# The __all__ variable is used to define what symbols get exported when from a module when you use the import * syntax.
__all__ = [
    "OpenaiClient",
]

llms = {
    "openai": OpenaiClient,
}



def get_prompt_builder(prompt_type: str) -> AbstractPromptBuilder:
    """Return a prompt builder of the specified type.

    Args:
        prompt_type: The type of prompt builder to create.

    Returns:
        A prompt builder.
    """
    if prompt_type not in prompts:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    return prompts[prompt_type]()


def get_prompt_choices():
    """Return a list of available prompt builders."""
    return prompts.keys()


def get_llm_choices():
    """Return a list of available LLM clients."""
    return llms.keys()


def get_llm_client(
    client_type: str, prompt: Union[str, AbstractPromptBuilder], **kwargs
) -> AbstractLLMClient:
    """Return an LLM client of the specified type.

    Args:
        client_type: The type of client to create.
        kwargs: Additional keyword arguments to pass to the client constructor.

    Returns:
        An LLM client.
    """
    if client_type == "openai":
        return OpenaiClient(prompt, **kwargs)
    else:
        raise ValueError(f"Invalid client type: {client_type}")
