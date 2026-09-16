"""Free temporary camera recordings only after comparing durable archive bytes."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

from dream_sim.io import atomic_json, digest


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.sha.update(data)
        return data


def is_camera_recording(relative: Path) -> bool:
    return relative.suffix == ".mp4" or (
        relative.suffix == ".npz"
        and relative.name.startswith(("observation_", "navigation_depth_", "visibility_"))
    )


def prune_recording(run: Path, durable: Path, name: str, *, execute=False) -> dict:
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Expected a task directory name")
    source = (run / name).resolve(strict=True)
    receipt_path = durable / name / "recording_archive.json"
    receipt = json.loads(receipt_path.read_text())
    archive = (durable / receipt["archive"]).resolve(strict=True)
    if not archive.is_relative_to(durable.resolve()):
        raise ValueError("Archive escapes its study")
    verified = []
    seen = set()
    with archive.open("rb") as raw:
        reader = HashedReader(raw)
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                relative = Path(member.name).relative_to(name)
                if relative in seen or ".." in relative.parts or relative.is_absolute():
                    raise ValueError("Unsafe or duplicated archive member")
                seen.add(relative)
                path = source / relative
                if not is_camera_recording(relative) or not path.exists():
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(source):
                    raise ValueError("Recording escapes its task directory")
                with tar.extractfile(member) as content:
                    expected = hashlib.file_digest(content, "sha256").hexdigest()
                if digest(path) != expected:
                    raise ValueError(f"Temporary recording differs from archive: {relative}")
                verified.append((path, path.stat().st_size, expected))
        while reader.read(1024 * 1024):
            pass
        # These immutable archives are read once for verification. Do not
        # retain hundreds of GiB in the host page cache during a large cleanup.
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(raw.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if reader.sha.hexdigest() != receipt["archive_sha256"]:
            raise ValueError("Durable archive checksum mismatch; no recordings removed")
    report = dict(
        name=name,
        archive=str(archive),
        archive_sha256=receipt["archive_sha256"],
        files=len(verified),
        bytes=sum(row[1] for row in verified),
        executed=execute,
        verified_unix_s=time.time(),
        removed_relative_paths=[],
    )
    if execute:
        for path, size, expected in verified:
            # Refuse to remove a file changed since the archive comparison.
            if digest(path) != expected:
                raise ValueError(f"Recording changed during pruning: {path.name}")
            path.unlink()
            report["removed_relative_paths"].append(path.relative_to(source).as_posix())
        atomic_json(durable / name / "recording_pruned.json", report)
    return report
