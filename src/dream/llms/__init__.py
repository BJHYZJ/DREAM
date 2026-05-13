from typing import Union

from .base import AbstractLLMClient, AbstractPromptBuilder
from .openai_client import OpenaiClient
from .qwen_client import QwenClient

# This is a list of all the modules that are imported when you use the import * syntax.
# The __all__ variable is used to define what symbols get exported when from a module when you use the import * syntax.
__all__ = [
    "OpenaiClient",
    "QwenClient",
]

llms = {
    "openai": OpenaiClient,
    "qwen": QwenClient,
}


def get_llm_choices():
    """Return a list of available LLM clients."""
    return llms.keys()


def get_llm_client(
    client_type: str, prompt: Union[str, AbstractPromptBuilder], **kwargs
) -> AbstractLLMClient:
    """Return an LLM client of the specified type."""
    if client_type not in llms:
        raise ValueError(f"Invalid client type: {client_type}")
    return llms[client_type](prompt, **kwargs)
