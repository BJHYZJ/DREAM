#!/usr/bin/env python3
"""Download the production DREAM base encoders with immutable revision locks.

Network access happens only here. Experiment runners use this cache offline.
No hosted inference service, user images, or credentials are used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

from huggingface_hub import snapshot_download

MODELS = ("google/siglip-base-patch16-224", "google/owlv2-base-patch16-ensemble")
PRODUCTION_MODELS = ("google/siglip-so400m-patch14-384", "google/owlv2-large-patch14-ensemble")
LOCKED_REVISIONS = {
    "google/siglip-base-patch16-224": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    "google/owlv2-base-patch16-ensemble": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
    "google/siglip-so400m-patch14-384": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    "google/owlv2-large-patch14-ensemble": "95e26936e865f87db1742128404b3c035d47d89d",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--production", action="store_true",
                        help="Also prepare the exact models used in RobotAgent.create_obstacle_map")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.cache_dir / "dream_models.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    requested = MODELS + PRODUCTION_MODELS if args.production else MODELS
    for model in requested:
        if model in lock and lock[model]!=LOCKED_REVISIONS[model]:
            raise ValueError(f"Existing cache lock differs from the released experiment: {model}")
        lock[model] = LOCKED_REVISIONS[model]
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"locked_revisions": lock}), flush=True)

    def download(model):
        path = snapshot_download(
            model, revision=lock[model], cache_dir=str(args.cache_dir),
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"],
            max_workers=2,
        )
        print(json.dumps({"model": model, "snapshot": path}), flush=True)
        return path

    # snapshot_download already parallelizes files. Nesting model downloads
    # triggers a tqdm global-lock race in some reproducibility environments.
    for model in requested:
        download(model)


if __name__ == "__main__":
    main()
