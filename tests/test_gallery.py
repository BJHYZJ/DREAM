"""Public-label changes must not relabel the original experimental cases."""
import copy
import json
from pathlib import Path

import pytest

from dream_sim.verify_evidence import validate_gallery_manifest


def records():
    root = Path(__file__).resolve().parents[1] / "reproducibility" / "evidence" / "gallery"
    manifest = json.loads((root / "manifest.json").read_text())
    catalog = json.loads((root / manifest["profile_catalog"]).read_text())
    return manifest, catalog


def test_public_labels_keep_exact_original_profiles():
    validate_gallery_manifest(*records())


@pytest.mark.parametrize("key", ["scene", "seed", "video", "source_id"])
def test_rejects_relabelled_case(key):
    original, catalog = records()
    changed = copy.deepcopy(original)
    changed["cases"][0][key] = "incorrect"
    with pytest.raises(ValueError):
        validate_gallery_manifest(changed, catalog)
