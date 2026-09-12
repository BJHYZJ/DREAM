import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def controller():
    path = Path(__file__).resolve().parents[1] / 'controllers/recovery_v3/experiments/maniskill_learned_probe.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SimulatorIO')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'smooth_body_target')
    namespace = dict(np=np, array=np.asarray)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    robot = SimpleNamespace(get_qlimits=lambda: np.array([[[0., 1.]] * 3 + [[0., .4]]]))
    io = SimpleNamespace(robot=SimpleNamespace(robot=robot, control_mode="pd_joint_pos"), body=np.array([.2, .3, .385]),
                         _torso_reference=.18, _torso_reference_velocity=0., _torso_dt=.05)
    return io, namespace['smooth_body_target']


def test_reversed_torso_request_obeys_velocity_and_acceleration_limits():
    io, advance = controller()
    velocities = [0.]
    positions = [.18]
    for step in range(700):
        if step == 35:
            io.body[2] = .01
        if step == 240:
            io.body[2] = .30
        target = advance(io)
        assert np.array_equal(target[:2], io.body[:2])
        velocities.append(io._torso_reference_velocity)
        positions.append(target[2])
    assert np.max(np.abs(velocities)) <= .04 + 1e-12
    assert np.max(np.abs(np.diff(velocities) / io._torso_dt)) <= .10 + 1e-12
    assert np.max(np.abs(np.diff(positions))) <= .04 * io._torso_dt + 1e-12
    assert abs(positions[-1] - .30) < 1e-5


def test_out_of_range_torso_goal_stays_within_joint_limits_after_settling():
    io, advance = controller()
    io.body[2] = 10.
    for _ in range(400):
        target = advance(io)
    assert abs(target[2] - .39) < 1e-5


def settle_method():
    path = Path(__file__).resolve().parents[1] / 'controllers/recovery_v3/experiments/maniskill_learned_probe.py'
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'SimulatorIO')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'settle_torso')
    namespace = dict(np=np, array=np.asarray)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['settle_torso']


def test_settling_accepts_the_clamped_fetch_goal_and_checks_payload():
    io, _ = controller()
    io.robot.robot.get_qlimits = lambda: np.array([[[0., 1.]] * 3 + [[0., .38615]]])
    io.body[2] = .385
    io._torso_reference = .37615
    io.torso_trace = lambda: dict(torso_position_m=.3753, torso_velocity_m_s=.00005)
    calls = []
    io.command = lambda: calls.append('step')
    settle_method()(io, check=lambda: calls.append('payload'))
    assert calls.count('step') == 20
    assert calls.count('payload') == 3


def test_settling_rejects_an_unsettled_physical_joint():
    import pytest
    io, _ = controller()
    io._torso_reference = .385
    io.torso_trace = lambda: dict(torso_position_m=.25, torso_velocity_m_s=-.12)
    io.command = lambda: None
    with pytest.raises(RuntimeError, match='did not settle'):
        settle_method()(io, max_steps=30)
