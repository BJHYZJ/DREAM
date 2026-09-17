"""Regression checks for the separately validated release-recovery candidate."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "controllers/continuous_return/experiments/instruction_policy.py"
)


def method():
    tree = ast.parse(SOURCE.read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "wait_for_release_stillness"
    )
    namespace = {"np": np, "array": np.asarray}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["wait_for_release_stillness"]


class ReleaseFixture:
    def __init__(self, torso=0.18, reachable_height=0.30, lost_after=None):
        self.reachable_height = reachable_height
        self.lost_after = lost_after
        self.controls = 0
        self.retention_checks = 0
        self.height_changes = []
        self.events = []
        self.position = np.array([0.027, 0.0, 1.0])
        self.pose = SimpleNamespace(p=self.position[None, :], to_transformation_matrix=self.matrix)
        self.io = SimpleNamespace(
            base=SimpleNamespace(control_freq=20),
            body=np.array([0.0, 0.0, torso]),
            robot=SimpleNamespace(tcp_pose=self.pose),
            settle_torso=self.settle,
        )
        self.instruction = SimpleNamespace(placement_relation="on")
        self.perceived_payload_radius = 0.03

    def matrix(self):
        matrix = np.eye(4)
        matrix[:3, 3] = self.position
        return matrix[None, :, :]

    def event(self, name, **values):
        self.events.append((name, values))

    def require_retained_payload(self):
        self.retention_checks += 1
        if self.lost_after is not None and self.controls >= self.lost_after:
            raise RuntimeError("Closed gripper is empty")

    def settle(self, *, check, hold_tcp):
        assert hold_tcp is True
        check()
        self.height_changes.append(float(self.io.body[2]))

    def hold_release_pose_step(self, goal, rotation):
        self.controls += 1
        # A reach limit persists until the torso lifts enough. The fixture
        # then exposes a stable object center exactly over the observed goal.
        self.position[:] = [
            0.0 if self.io.body[2] >= self.reachable_height - 1e-9 else 0.027,
            0.0,
            1.0,
        ]

    def run(self):
        return method()(
            self,
            np.array([0.0, 0.0, 1.0]),
            np.eye(3),
            release=np.array([0.0, 0.0, 1.0]),
            geometry={"observed_safe_radius_m": 0.10},
            offset=np.zeros(3),
        )


def test_late_reachable_release_is_recovered_with_retention_checks():
    fixture = ReleaseFixture()
    assert fixture.run() is True
    assert fixture.height_changes == pytest.approx([0.24, 0.30])
    assert fixture.retention_checks >= fixture.controls
    assert fixture.events[-1][0] == "release_pose_settled"


def test_unreachable_release_preserves_alignment_threshold_and_height_ceiling():
    fixture = ReleaseFixture(reachable_height=0.50)
    assert fixture.run() is False
    assert fixture.io.body[2] <= 0.385
    assert fixture.events[-1][0] == "release_pose_unsettled"
    assert fixture.events[-1][1]["horizontal_error_m"] == pytest.approx(0.027)


def test_payload_loss_stops_before_further_pose_commands():
    fixture = ReleaseFixture(lost_after=45)
    with pytest.raises(RuntimeError, match="gripper is empty"):
        fixture.run()
    assert fixture.controls == 45
    assert not any(event == "release_pose_settled" for event, _ in fixture.events)


def test_reachable_initial_release_does_not_lift_the_torso():
    fixture = ReleaseFixture(reachable_height=0.10)
    assert fixture.run() is True
    assert fixture.height_changes == []
