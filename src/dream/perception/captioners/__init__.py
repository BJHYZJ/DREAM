from typing import Any

from .blip_captioner import BlipCaptioner
from .qwen_captioner import QwenCaptioner

captioners = ["qwen", "blip"]


def get_captioner(captioner_name, args: Any):
    """Get captioner."""
    if captioner_name == "qwen":
        return QwenCaptioner(**args)
    elif captioner_name == "blip":
        return BlipCaptioner(**args)
    else:
        raise ValueError(
            f"Captioner {captioner_name} not implemented or not supported. Should be one of {captioners}."
        )
