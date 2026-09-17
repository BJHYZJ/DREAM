"""Rendering must keep the caller's installed Vulkan backend."""

from dream_sim.io import configure_vulkan
from dream_sim.limited_worker import ALLOWED_GPUS, worker_environment


def test_replay_preserves_explicit_vulkan_driver(monkeypatch):
    monkeypatch.setenv("VK_ICD_FILENAMES", "/custom/driver.json")
    configure_vulkan()
    import os

    assert os.environ["VK_ICD_FILENAMES"] == "/custom/driver.json"


def test_worker_preserves_explicit_vulkan_driver(monkeypatch):
    monkeypatch.setenv("VK_ICD_FILENAMES", "/custom/driver.json")
    assert worker_environment(ALLOWED_GPUS[0])["VK_ICD_FILENAMES"] == "/custom/driver.json"
