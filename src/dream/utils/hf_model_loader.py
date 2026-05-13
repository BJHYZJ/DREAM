import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class HFLoadPlan:
    load_target: str
    local_only: bool
    local_snapshot: Optional[str]


def hf_hub_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("TRANSFORMERS_CACHE"):
        return Path(os.environ["TRANSFORMERS_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_valid_snapshot(path: Path) -> bool:
    if not path.exists():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    return has_config and has_weights


def find_local_hf_snapshot(repo_id: str) -> Optional[str]:
    model_dir_name = "models--" + repo_id.replace("/", "--")
    base = hf_hub_root() / model_dir_name
    snapshots = base / "snapshots"
    if not snapshots.exists():
        return None

    ref_main = base / "refs" / "main"
    if ref_main.exists():
        commit = ref_main.read_text().strip()
        candidate = snapshots / commit
        if is_valid_snapshot(candidate):
            return str(candidate)

    candidates = sorted(
        [p for p in snapshots.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if is_valid_snapshot(candidate):
            return str(candidate)
    return None


def build_hf_load_plan(repo_id: str, friendly_name: str) -> HFLoadPlan:
    local_snapshot = find_local_hf_snapshot(repo_id)
    load_target = local_snapshot if local_snapshot is not None else repo_id
    local_only = local_snapshot is not None
    return HFLoadPlan(
        load_target=load_target,
        local_only=local_only,
        local_snapshot=local_snapshot,
    )
