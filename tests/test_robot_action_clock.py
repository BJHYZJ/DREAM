import importlib.util
from pathlib import Path

import pytest

p = (
    Path(__file__).resolve().parents[1]
    / "controllers/compact_v1/experiments/instruction_execution_clock.py"
)
spec = importlib.util.spec_from_file_location("action_clock", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_exact_900_second_boundary_uses_control_frequency():
    for hz in (10, 20, 30):
        clock = m.RobotActionBudget(900, hz)
        clock.before_step(900 * hz - 1)
        assert not clock.report(900 * hz)["robot_action_timeout"]
        with pytest.raises(TimeoutError):
            clock.before_step(900 * hz)
        assert clock.report(900 * hz)["robot_action_seconds"] == 900
        assert clock.report(900 * hz)["robot_action_timeout"]


def test_waiting_and_reporting_cannot_consume_controls():
    clock = m.RobotActionBudget(900, 20)
    for _ in range(1000):
        clock.before_step(10)
    assert clock.report(10)["robot_action_seconds"] == 0.5


@pytest.mark.parametrize("seconds,hz", [(0, 20), (900, 0), (float("inf"), 20), (900, float("nan"))])
def test_invalid_clocks_rejected(seconds, hz):
    with pytest.raises(ValueError):
        m.RobotActionBudget(seconds, hz)
