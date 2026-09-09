import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib import error, request

import numpy as np
from PIL import Image

try:
    from dream.llms.base import AbstractLLMClient, AbstractPromptBuilder
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from dream.llms.base import AbstractLLMClient, AbstractPromptBuilder


class QwenClient(AbstractLLMClient):
    """Client for Qwen VL models via DashScope OpenAI-compatible HTTP API."""

    model_choices = [
        "qwen-vl-max-latest",
        "qwen-vl-plus-latest",
        "qwen3-vl-flash",
        "qwen3-vl-plus",
        "qwen2.5-vl-72b-instruct",
        "qwen2.5-vl-32b-instruct",
        "qwen2.5-vl-7b-instruct",
    ]

    def __init__(
        self,
        prompt: Union[str, AbstractPromptBuilder],
        prompt_kwargs: Optional[Dict[str, Any]] = None,
        model: str = "qwen-vl-max-latest",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_sec: int = 60,
    ):
        super().__init__(prompt, prompt_kwargs)
        self.model = model
        self.timeout_sec = timeout_sec
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for QwenClient.")

        default_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL", default_base)).rstrip("/")
        self.chat_endpoint = f"{self.base_url}/chat/completions"

        if self.model not in self.model_choices:
            print("Your Qwen model:", self.model)

    def _process_input(self, command, verbose=False):
        """Transform user command to OpenAI-compatible multimodal message content."""
        if isinstance(command, str):
            user_commands = command
        else:
            user_commands = []  # type:ignore
            for c in command:
                if isinstance(c, dict):
                    user_commands.append(c)
                elif isinstance(c, str):
                    user_commands.append({"type": "text", "text": c})
                elif isinstance(c, Image.Image) or isinstance(c, np.ndarray):
                    if isinstance(c, np.ndarray):
                        image = Image.fromarray(c.astype(np.uint8), mode="RGB")
                    else:
                        image = c

                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_bytes = buffered.getvalue()
                    base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
                    user_commands.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_encoded}"},
                        }
                    )
                else:
                    raise NotImplementedError("We only support text and image for now!")

        if verbose:
            print("input to the model:")
            if isinstance(user_commands, str):
                print(user_commands)
            else:
                for (idx, user_command) in enumerate(user_commands):
                    if "image_url" in user_command:
                        print(idx, ".", user_command["type"])
                    else:
                        print(idx, ".", user_command["type"], user_command["text"])
        return user_commands

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = request.Request(
            self.chat_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Qwen HTTPError {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Qwen URL error: {exc}") from exc

    @staticmethod
    def _extract_text(response: Dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Invalid Qwen response: missing choices. raw={response}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"Invalid Qwen response: missing message.content. raw={response}")
        return content

    def __call__(self, command: Union[str, list], verbose: bool = False):
        if verbose:
            print(f"{self.system_prompt=}")

        command = self._process_input(command, verbose=verbose)  # type:ignore
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command},
            ],
        }
        response = self._post_chat_completion(payload)
        output_text = self._extract_text(response)
        if verbose:
            print(f"output_text={output_text}")
        return output_text

    def sample(self, command: Union[str, list], n_samples: int, verbose: bool = False):
        if verbose:
            print(f"{self.system_prompt=}")

        command = self._process_input(command, verbose=verbose)  # type:ignore
        payload = {
            "model": self.model,
            "temperature": 1,
            "n": n_samples,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": command},
            ],
        }
        response = self._post_chat_completion(payload)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"Invalid Qwen response: missing choices. raw={response}")
        if verbose:
            print(f"choices={choices}")
        return choices


def main():
    # ---- Hardcoded test data: edit these values directly ----
    test_model = "qwen-vl-max-latest"
    test_system_prompt = "You are a precise visual assistant. Answer briefly."
    test_image_path = "/home/yanzj/workspace/code/DREAM/dream_log/debug_2026-04-22_19-33-59/rgb2654.jpg"
    test_user_text = "what are you see?"
    test_timeout_sec = 60
    test_verbose = False
    # ---------------------------------------------------------

    client = QwenClient(
        prompt=test_system_prompt,
        model=test_model,
        timeout_sec=test_timeout_sec,
    )

    if Path(test_image_path).exists():
        image = Image.open(test_image_path).convert("RGB")
        command = [image, test_user_text]
    else:
        print(f"[QwenClient main] image not found: {test_image_path}; testing text-only.")
        command = test_user_text

    response = client(command, verbose=test_verbose)
    print("qwen:", response)


if __name__ == "__main__":
    main()
