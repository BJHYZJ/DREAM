"""Verify and extract the controller source trees used by the simulation profiles."""
from __future__ import annotations

import argparse
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
ARCHIVES = PROJECT / "reproducibility" / "source_archives"
LOCK = ARCHIVES / "selected_profiles_v1.lock.json"
EVIDENCE = PROJECT / "reproducibility" / "evidence"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_member(name: str) -> Path:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name:
        raise ValueError(f"Unsafe archived path: {name}")
    return Path(*path.parts)


def verify_tree(root: Path, members: dict) -> None:
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected source symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(members):
        raise ValueError("Expanded source member set changed; use a new source-cache directory")
    for name, expected in members.items():
        path = root / safe_member(name)
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"Expanded source changed: {name}; existing data will not be overwritten")


@lru_cache(maxsize=1)
def source_root() -> Path:
    if not LOCK.is_file():
        raise FileNotFoundError("Use an editable installation from a complete simulation-branch checkout")
    lock = json.loads(LOCK.read_text())
    archive = ARCHIVES / safe_member(lock["archive"])
    if sha(archive) != lock["sha256"]:
        raise ValueError("Historical source archive checksum mismatch")
    cache = Path(os.environ.get("DREAM_SIM_SOURCE_CACHE", PROJECT / ".runtime" / "sources")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / lock["sha256"]
    # Serialize extraction so concurrent workers see a complete, verified tree.
    with (cache / "extract.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if not destination.exists():
            with tempfile.TemporaryDirectory(prefix="expand-", dir=cache) as temporary:
                staging = Path(temporary) / "sources"
                staging.mkdir()
                with zipfile.ZipFile(archive) as bundle:
                    names = bundle.namelist()
                    if len(names) != len(set(names)) or set(names) != set(lock["members"]):
                        raise ValueError("Source archive member set changed")
                    for info in bundle.infolist():
                        relative = safe_member(info.filename)
                        expected = lock["members"][info.filename]
                        if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
                            raise ValueError("Only regular source files are allowed")
                        if info.file_size != expected["bytes"]:
                            raise ValueError("Archived source size changed")
                        payload = bundle.read(info)
                        if hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                            raise ValueError("Archived source checksum mismatch")
                        target = staging / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("xb") as stream:
                            stream.write(payload)
                verify_tree(staging, lock["members"])
                staging.rename(destination)
        else:
            verify_tree(destination, lock["members"])
    # Legacy helper imports must not add bytecode files to a verified tree.
    sys.dont_write_bytecode = True
    return destination


def engine_root() -> Path:
    return source_root() / "engine" / "DREAM_code"


def audit_root() -> Path:
    return source_root() / "audit_sources"


def verify_public_configs() -> dict:
    catalog_root = engine_root() / "experiments" / "repro_profiles"
    original = catalog_root / "profiles.json"
    index = json.loads((PROJECT / "configs" / "cases.json").read_text())
    if sha(original) != index["catalog_sha256"] or index["cases"] != json.loads(original.read_text())["cases"]:
        raise ValueError("Public case index differs from the frozen profiles")
    verified = 0
    for name in ("tasks", "locks"):
        public = PROJECT / "configs" / name
        frozen = catalog_root / name
        expected = {p.relative_to(frozen).as_posix(): sha(p) for p in frozen.rglob("*") if p.is_file()}
        actual = {p.relative_to(public).as_posix(): sha(p) for p in public.rglob("*") if p.is_file()}
        if actual != expected or any(p.is_symlink() for p in public.rglob("*")):
            raise ValueError(f"Public {name} differ from the frozen release")
        verified += len(expected)
    return {"public_case_index_verified": True, "public_config_files_verified": verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[f"{i:02d}" for i in range(1, 11)],
                        help="Print the exact, inspectable controller path for this video")
    args = parser.parse_args()
    root = source_root()
    report = {"source_root": str(root), "engine": str(engine_root()),
              "new_policy_execution": False, **verify_public_configs()}
    if args.case:
        catalog_root = engine_root() / "experiments" / "repro_profiles"
        catalog = json.loads((catalog_root / "profiles.json").read_text())
        case = next(row for row in catalog["cases"] if row["id"] == args.case)
        report.update(case=args.case, source_id=case["source_id"],
                      controller=str(catalog_root / catalog["sources"][case["source_id"]]["repository"]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
