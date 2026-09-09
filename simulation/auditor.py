"""Resolve the exact record-review version used by a selected original video.

Policy and record-review versions are independently pinned. Selecting the
policy snapshot's older reviewer can incorrectly classify a valid later
rediscovery using the first moving-target detection instead.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_reviewer(case: dict, catalog_root: Path) -> Path:
    original = json.loads((catalog_root / case["records"]["record_review.json"]).read_text())
    expected = original["review_script_sha256"]
    candidates = sorted(catalog_root.glob("sources/*/DREAM_code/experiments/review_instruction_record.py"))
    candidates.append(Path(__file__).parent / "_audit_sources" / "record_v3" / "DREAM_code" /
                      "experiments" / "review_instruction_record.py")
    matches = [path for path in candidates if path.is_file() and sha(path) == expected]
    if not matches:
        raise ValueError(f"Missing exact recorded reviewer for case {case['id']}: {expected}")
    chosen = matches[0]
    if case["id"] == "05":
        lock_path = Path(__file__).parent / "_audit_sources" / "record_v3" / "source_hashes.json"
        hashes = json.loads(lock_path.read_text())
        source = chosen.parents[1]
        actual = {path.relative_to(source).as_posix() for path in
                  [*source.glob("experiments/*.py"), *source.glob("src/dream/**/*.py")]}
        if actual != set(hashes) or any(sha(source / name) != value for name, value in hashes.items()):
            raise ValueError("Historical record-review v3 source changed")
    return chosen
