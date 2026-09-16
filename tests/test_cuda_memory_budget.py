import json
from types import SimpleNamespace

import pytest
import torch

from dream_sim.limited_worker import configure_cuda_memory_budget


def test_cuda_allocator_cap_is_applied_and_recorded_without_reservation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: calls.append((fraction, device)),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=96 * 1024**3),
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(torch_cuda_memory_fraction=0.08)))
    receipt = tmp_path / "task.runtime.json"
    configure_cuda_memory_budget(receipt, config)
    row = json.loads(receipt.with_suffix(".cuda.json").read_text())
    assert calls == [(0.08, 0)]
    assert row["torch_allocator_limit_bytes"] == int(0.08 * 96 * 1024**3)
    assert row["memory_reserved_by_setting"] is False


@pytest.mark.parametrize("fraction", [0, -0.1, 0.11, 1, "0.08"])
def test_invalid_budget_is_rejected_before_cuda_is_touched(tmp_path, monkeypatch, fraction):
    def forbidden(*a, **k):
        raise AssertionError("Invalid budget must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", forbidden)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(torch_cuda_memory_fraction=fraction)))
    with pytest.raises(ValueError, match="ten percent"):
        configure_cuda_memory_budget(tmp_path / "receipt.json", config)
