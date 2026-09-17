"""File operations shared by execution, restart, and result evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def configure_vulkan() -> None:
    """Respect an explicit renderer; otherwise locate an installed Mesa ICD."""
    if os.environ.get("VK_ICD_FILENAMES"):
        return
    for name in ("lvp_icd.json", "lvp_icd.x86_64.json"):
        path = Path("/usr/share/vulkan/icd.d") / name
        if path.is_file():
            os.environ["VK_ICD_FILENAMES"] = str(path)
            return


def digest(path: Path) -> str:
    """Hash a file without retaining large recording archives in the page cache."""
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.suffix == ".tar" and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return result


def atomic_json(path: Path, value: Any) -> None:
    """Replace one scheduler-owned JSON record only after serialization succeeds."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
