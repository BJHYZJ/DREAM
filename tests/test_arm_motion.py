import sys
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dream_sim.fold_review import review_return_stages

ROOT = Path(__file__).resolve().parents[1]
controller = ROOT / "controllers/continuous_return/experiments"
spec = spec_from_file_location(
    "tested_native_motion_return", controller / "instruction_arm_return.py"
)
arm = module_from_spec(spec)
sys.modules[spec.name] = arm
sys.path.insert(0, str(controller))
try:
    spec.loader.exec_module(arm)
    motion = import_module("instruction_arm_motion")
finally:
    sys.path.pop(0)
JointMotionSampler, MEASUREMENT = motion.JointMotionSampler, motion.MEASUREMENT


class IO:
    def __init__(self, biased=False, wiggle=False):
        self.position = np.zeros((1, 15))
        self.position[0, list(arm.ARM_INDICES)] = [-0.25, -1.14, 0.53, 1.15, -0.34, 1.43, 0.26]
        model = SimpleNamespace(
            get_qpos=lambda: self.position.copy(),
            get_qvel=self.velocities,
            get_qlimits=lambda: np.tile([-np.pi, np.pi], (1, 15, 1)),
            get_links=lambda: [],
        )
        self.robot = SimpleNamespace(robot=model)
        self.base = SimpleNamespace(
            agent=self.robot,
            control_freq=20,
            scene=SimpleNamespace(px=SimpleNamespace(timestep=0.01), get_contacts=lambda: []),
            _after_simulation_step=lambda: None,
        )
        self.arm = self.position[0, list(arm.ARM_INDICES)].copy()
        self.grasp_seed_arm = np.array([0.6, 0, 0, 0, 0, 0, 0])
        self.step_id = 0
        self.trace = []
        self.commands = []
        self.biased, self.wiggle = biased, wiggle

    def velocities(self):
        values = np.zeros_like(self.position)
        if self.biased and getattr(self, "arm_return_stage", "") == "stow":
            values[0, arm.ARM_INDICES[-1]] = 0.09
        return values

    def pose(self):
        return np.zeros(3)

    def hold_measured_arm(self):
        self.arm = self.position[0, list(arm.ARM_INDICES)].copy()

    def command(self):
        initial = self.position[0, list(arm.ARM_INDICES)].copy()
        for tick in range(1, 6):
            measured = initial + (self.arm - initial) * tick / 5
            if self.wiggle and tick == 2 and self.arm_return_stage == "stow":
                measured[0] += 0.004
            self.position[0, list(arm.ARM_INDICES)] = measured
            self.base._after_simulation_step()
        self.step_id += 1
        self.commands.append(self.arm.copy())
        self.trace.append(
            dict(
                step=self.step_id,
                arm_joint_positions_rad=self.position[0, list(arm.ARM_INDICES)].tolist(),
                arm_joint_velocities_rad_s=self.velocities()[0, list(arm.ARM_INDICES)].tolist(),
            )
        )


def record(biased=True):
    io = IO(biased=biased)
    hook = io.base._after_simulation_step
    receipt = arm.execute_return(io)
    assert io.base._after_simulation_step is hook
    event = dict(step=io.step_id, **receipt)
    actions = {row["step"]: row for row in io.trace}
    replay = dict(passed=True, control_steps=list(actions))
    return io, event, actions, replay


def test_normal_native_velocity_return_needs_no_alternate_measurement():
    _, event, actions, _ = record(biased=False)
    assert all("settling_measurement" not in stage for stage in event["stages"])
    assert review_return_stages(event, actions)


def test_static_pose_with_tgs_velocity_bias_requires_independent_native_replay():
    io, event, actions, replay = record()
    stage = event["stages"][-1]
    assert stage["settling_measurement"] == MEASUREMENT
    assert stage["native_physics_samples"] == 100
    assert stage["measured_maximum_speed_rad_s"] == 0
    assert io.trace[-1]["arm_joint_velocities_rad_s"][-1] == 0.09
    assert not review_return_stages(event, actions)
    assert review_return_stages(event, actions, replay)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "gap",
        "endpoint",
        "oscillation",
        "nan",
        "dt",
        "sample_count",
        "receipt_speed",
        "replay_failed",
        "replay_missing",
    ],
)
def test_incomplete_fabricated_or_moving_native_windows_are_rejected(damage):
    _, event, actions, replay = record()
    row = actions[event["step"] - 5]
    native = row["arm_joint_motion"]
    if damage == "missing":
        del row["arm_joint_motion"]
    elif damage == "gap":
        native["initial_joint_positions_rad"][0] += 0.00001
    elif damage == "endpoint":
        row["arm_joint_positions_rad"][0] += 0.00001
    elif damage == "oscillation":
        native["joint_positions_rad"][2][0] += 0.004
    elif damage == "nan":
        native["joint_positions_rad"][2][0] = float("nan")
    elif damage == "dt":
        native["physics_dt_s"] *= 2
    elif damage == "sample_count":
        native["joint_positions_rad"].pop()
    elif damage == "receipt_speed":
        event["stages"][-1]["measured_maximum_speed_rad_s"] = 0.079
    elif damage == "replay_failed":
        replay["passed"] = False
    else:
        replay["control_steps"].remove(row["step"])
    assert not review_return_stages(event, actions, replay)


def test_controller_rejects_motion_hidden_between_control_endpoints():
    io = IO(biased=True, wiggle=True)
    hook = io.base._after_simulation_step
    with pytest.raises(arm.ArmReturnError, match="did not settle"):
        arm.execute_return(io)
    assert io.base._after_simulation_step is hook


def test_sampler_does_not_modify_robot_and_restores_callback_on_exception():
    io = IO()
    original = io.base._after_simulation_step
    initial = io.position.copy()
    with pytest.raises(RuntimeError):
        with JointMotionSampler(io) as sampler:
            io.base._after_simulation_step()
            assert len(sampler.records[1]["joint_positions_rad"]) == 1
            raise RuntimeError("stop")
    np.testing.assert_array_equal(initial, io.position)
    assert io.base._after_simulation_step is original


def test_interpolation_retains_limit_margin_without_clipping_or_slower_global_limits():
    initial = np.zeros(7)
    points = np.zeros((4, 7))
    points[:, 0] = [0.4, 0.5, 0.5, 0.4]
    points[:, 1] = [0.1, 0.2, 0.3, 0.4]
    limits = np.tile([-3.0, 3.0], (7, 1))
    limits[0, 1] = 0.508
    unconstrained = arm.continuous_joint_reference(initial, points, 20)
    assert unconstrained[:, 0].max() > 0.5
    bounded = arm.continuous_joint_reference(initial, points, 20, joint_limits=limits)
    assert bounded[:, 0].max() <= 0.5 + 1e-12
    np.testing.assert_allclose(bounded[-1], points[-1])
    velocity = np.diff(np.vstack([initial, initial, bounded, bounded[-1]]), axis=0) * 20
    acceleration = np.diff(velocity, axis=0) * 20
    assert np.abs(velocity).max() <= arm.MAX_SPEED
    assert np.abs(acceleration).max() <= arm.MAX_ACCELERATION


@pytest.mark.parametrize("can_clear", [True, False])
def test_failed_open_hand_approach_clears_support_before_navigation_resumes(can_clear):
    import ast

    path = ROOT / "controllers/continuous_return/experiments/instruction_policy.py"
    tree = ast.parse(path.read_text())
    policy = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InstructionSearchPilot"
    )
    method = next(
        node
        for node in policy.body
        if isinstance(node, ast.FunctionDef) and node.name == "defer_ungraspable_pickup"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"
        ),
        namespace,
    )
    calls = []

    def clear_support(*, loaded):
        assert loaded is False
        calls.append("clear_support")
        if not can_clear:
            raise RuntimeError("No safe clearance path")
        return {"settled": True}

    pilot = SimpleNamespace(
        cached=np.array([1.0, 2.0, 3.0]),
        grasp_motion_attempted=True,
        grasp_closure_attempted=False,
        deferred_pickup_regions=[],
        io=SimpleNamespace(hold_measured_arm=lambda: calls.append("hold")),
        fold_arm_in_place=clear_support,
        event=lambda name, **values: calls.append(name),
        memory=SimpleNamespace(reject=lambda *args, **kwargs: calls.append("reject")),
        instruction=SimpleNamespace(pickup_query="mug"),
    )
    if can_clear:
        assert namespace["defer_ungraspable_pickup"](pilot)
        assert calls.index("clear_support") < calls.index("reject")
        assert pilot.grasp_motion_attempted is False
        assert pilot.cached is None and pilot.task_stage == "pickup_search"
    else:
        with pytest.raises(RuntimeError, match="No safe clearance path"):
            namespace["defer_ungraspable_pickup"](pilot)
        assert calls == ["hold", "clear_support"]
        assert pilot.grasp_motion_attempted is True and pilot.cached is not None
