"""Regression coverage for packaging; not claims of new task execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from dream_sim.sources import LOCK, safe_member, source_root, verify_tree, verify_public_configs


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../escape", "a\\b", "", "a//b", "./a"])
def test_unsafe_archive_paths_rejected(name):
    with pytest.raises(ValueError):
        safe_member(name)


def test_complete_expanded_release_verified():
    lock = json.loads(LOCK.read_text())
    verify_tree(source_root(), lock["members"])
    assert len(lock["members"]) == 2193
    assert verify_public_configs()["public_case_index_verified"]


@pytest.mark.parametrize("mutation", ["bytes", "extra", "symlink", "missing"])
def test_modified_source_cache_is_rejected(tmp_path, mutation):
    file = tmp_path / "source.py"
    file.write_bytes(b"original")
    members = {"source.py": {"bytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}}
    verify_tree(tmp_path, members)
    if mutation == "bytes":
        file.write_bytes(b"modified")
    elif mutation == "extra":
        (tmp_path / "extra.py").write_bytes(b"extra")
    elif mutation == "symlink":
        (tmp_path / "alias").symlink_to(file)
    else:
        file.unlink()
    with pytest.raises(ValueError):
        verify_tree(tmp_path, members)


@pytest.mark.parametrize("module", ["run", "audit", "render", "video", "prepare", "profile", "sources"])
def test_public_module_help(module):
    result = subprocess.run([sys.executable, "-m", "dream_sim." + module, "--help"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
