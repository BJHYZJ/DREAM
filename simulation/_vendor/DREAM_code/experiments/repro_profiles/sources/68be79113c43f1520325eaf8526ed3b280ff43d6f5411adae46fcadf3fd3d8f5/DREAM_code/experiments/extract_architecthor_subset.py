#!/usr/bin/env python3
"""Range-extract the ten ArchitecTHOR scenes from ManiSkill's official ZIP.

The published AI2THOR archive contains more than 10k scenes.  This utility
resolves the exact assets referenced by the ten packaged ArchitecTHOR configs
and downloads only those ZIP members.  Assets are not redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from remotezip import RemoteZip


URL = "https://huggingface.co/datasets/haosulab/AI2THOR/resolve/main/ai2thor.zip"
SCENES = (
    "ArchitecTHOR-Test-03", "ArchitecTHOR-Val-02", "ArchitecTHOR-Val-01",
    "ArchitecTHOR-Test-02", "ArchitecTHOR-Val-04", "ArchitecTHOR-Val-03",
    "ArchitecTHOR-Val-00", "ArchitecTHOR-Test-01", "ArchitecTHOR-Test-04",
    "ArchitecTHOR-Test-00",
)
ROOT = "ai2thor"
CONFIG_ROOT = f"{ROOT}/ai2thor-hab/configs/scenes/ArchitecTHOR"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != ROOT:
        raise ValueError(f"unsafe/unexpected ZIP member: {name}")


def resolve_members(remote: RemoteZip) -> tuple[list[str], dict[str, object]]:
    available = set(remote.namelist())
    members = {f"{ROOT}/ai2thor-hab/configs/object_semantic_id_mapping.json"}
    scene_counts = {}
    for scene in SCENES:
        config_name = f"{CONFIG_ROOT}/{scene}.scene_instance.json"
        config = json.loads(remote.read(config_name))
        stage = config["stage_instance"]["template_name"]
        members |= {config_name, f"{ROOT}/ai2thor-hab/assets/{stage}.glb"}
        templates = {item["template_name"] for item in config["object_instances"]}
        scene_counts[scene] = len(templates)
        for template in templates:
            members.add(f"{ROOT}/ai2thorhab-uncompressed/assets/{template}.glb")
            members.add(f"{ROOT}/ai2thorhab-uncompressed/configs/{template}.object_config.json")
    missing = sorted(members - available)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} referenced members missing from official ZIP:\n"
            + "\n".join(missing[:20])
        )
    ordered = sorted(members)
    return ordered, {
        "source_url": URL,
        "scene_ids": list(SCENES),
        "navigable_positions_in_official_archive": False,
        "navigation_mesh_policy": "derive floor raster from referenced stage mesh",
        "member_count": len(ordered),
        "unique_object_templates_per_scene": scene_counts,
        "compressed_bytes": sum(remote.getinfo(name).compress_size for name in ordered),
        "uncompressed_bytes": sum(remote.getinfo(name).file_size for name in ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-parent", type=Path,
        default=Path("../.maniskill_assets/data/scene_datasets"),
        help="parent under which the ZIP's ai2thor/ directory is created",
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with RemoteZip(URL) as remote:
        members, manifest = resolve_members(remote)
        print(json.dumps(manifest, indent=2), flush=True)
        if args.list_only:
            return
        args.output_parent.mkdir(parents=True, exist_ok=True)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    thread_state = threading.local()
    opened: list[RemoteZip] = []
    opened_lock = threading.Lock()

    def download(name: str) -> tuple[str, dict[str, object]]:
        remote = getattr(thread_state, "remote", None)
        if remote is None:
            remote = RemoteZip(URL)
            thread_state.remote = remote
            with opened_lock:
                opened.append(remote)
        info = remote.getinfo(name)
        target = args.output_parent / name
        if not target.exists() or target.stat().st_size != info.file_size:
            _safe_member(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            partial.write_bytes(remote.read(name))
            partial.replace(target)
        return name, {"bytes": target.stat().st_size, "sha256": _sha256(target)}

    artifacts = {}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, name): name for name in members}
            for index, future in enumerate(as_completed(futures), 1):
                name, record = future.result()
                artifacts[name] = record
                print(json.dumps({"member": index, "total": len(members), "path": name}), flush=True)
    finally:
        for remote in opened:
            remote.close()
    manifest["artifacts"] = artifacts
    manifest_path = args.output_parent / ROOT / "ARCHITECTHOR_SUBSET_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
