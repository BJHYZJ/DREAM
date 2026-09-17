"""Arm-return sequencing and failure handling without a simulator or GPU."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

implementations = {}
for name in ("staged_return", "continuous_return"):
    path = Path(__file__).resolve().parents[1] / "controllers" / name / "experiments/instruction_arm_return.py"
    spec = spec_from_file_location("tested_" + name, path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    implementations[name] = module
arm_return = implementations["staged_return"]


@pytest.fixture(autouse=True, params=implementations)
def controller(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "arm_return", implementations[request.param])


class RobotFeedback:
    def __init__(self):
        self.position = np.zeros((1, 15))
        self.position[0, list(arm_return.ARM_INDICES)] = [
            -0.25,
            -1.14,
            0.53,
            1.15,
            -0.34,
            1.43,
            0.26,
        ]
        self.links = [
            SimpleNamespace(name=name, _objs=[SimpleNamespace(entity=object())])
            for name in ("torso", "forearm")
        ]

    def get_qpos(self):
        return self.position.copy()

    def get_qvel(self):
        return np.zeros_like(self.position)

    def get_qlimits(self):
        return np.tile([-np.pi, np.pi], (1, 15, 1))

    def get_links(self):
        return self.links


class FeedbackIO:
    def __init__(self, fault=None):
        self.model = RobotFeedback()
        self.robot = SimpleNamespace(robot=self.model)
        self.base = SimpleNamespace(
            agent=self.robot,
            control_freq=20,
            scene=SimpleNamespace(
                get_contacts=lambda: self.contacts, px=SimpleNamespace(timestep=0.005)
            ),
            _after_simulation_step=lambda: None,
        )
        self.arm = self.model.position[0, list(arm_return.ARM_INDICES)].copy()
        self.grasp_seed_arm = np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.step_id = 0
        self.commands = []
        self.trace = []
        self.contacts = []
        self.fault = fault
        self.holds = 0

    def pose(self):
        return np.array([0.03 if self.fault == "base" and self.step_id > 10 else 0.0, 0.0, 0.0])

    def hold_measured_arm(self):
        self.arm = self.model.position[0, list(arm_return.ARM_INDICES)].copy()
        self.holds += 1

    def command(self):
        next_step = self.step_id + 1
        self.commands.append((self.arm_return_stage, self.arm.copy()))
        initial = self.model.position[0, list(arm_return.ARM_INDICES)].copy()
        endpoint = initial.copy() if self.fault == "tracking" else self.arm.copy()
        if self.fault == "stalled":
            endpoint[3] -= 0.06
        if self.fault == "collision" and next_step > 10:
            self.contacts = [
                SimpleNamespace(
                    bodies=[link._objs[0] for link in self.model.links],
                    points=[SimpleNamespace(impulse=np.array([0.01, 0.0, 0.0]))],
                )
            ]
        substeps = round(1 / (self.base.control_freq * self.base.scene.px.timestep))
        for tick in range(1, substeps + 1):
            self.model.position[0, list(arm_return.ARM_INDICES)] = (
                endpoint if tick == substeps else initial + (endpoint - initial) * tick / substeps
            )
            self.base._after_simulation_step()
        self.step_id = next_step
        self.trace.append(dict(step=self.step_id,
            arm_joint_positions_rad=self.model.get_qpos()[0, list(arm_return.ARM_INDICES)].tolist(),
            arm_joint_velocities_rad_s=self.model.get_qvel()[0, list(arm_return.ARM_INDICES)].tolist()))


def test_arm_shortens_before_shoulder_turn_and_confirms_every_intermediate_pose():
    io = FeedbackIO()
    pan = float(io.arm[0])
    receipt = arm_return.execute_return(io)
    short = next(stage for stage in receipt["stages"] if stage["stage"] == "shorten")
    assert all(command[0] == pytest.approx(pan) for _, command in io.commands[: short["end_step"]])
    assert len(receipt["stages"]) > 1
    assert all(stage["settled"] for stage in receipt["stages"])
    assert np.max(np.abs(arm_return.joint_error(arm_return.TRANSPORT, io.arm))) < 0.025
    differences = np.diff(np.array([command for _, command in io.commands]), axis=0)
    assert np.max(np.abs(differences)) * io.base.control_freq <= arm_return.MAX_SPEED + 1e-6
    assert receipt["settle_window_s"] >= 1.0


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("collision", "self-contact"),
        ("tracking", "tracking error"),
        ("base", "base moved"),
        ("stalled", "stalled"),
    ],
)
def test_failed_return_holds_measured_arm_and_does_not_claim_completion(fault, reason):
    io = FeedbackIO(fault)
    original_hook = io.base._after_simulation_step
    with pytest.raises(arm_return.ArmReturnError, match=reason):
        arm_return.execute_return(io)
    assert io.holds >= 2
    assert io.arm_return_stage is None
    assert io.base._after_simulation_step is original_hook
    np.testing.assert_allclose(io.arm, io.model.position[0, list(arm_return.ARM_INDICES)])


def test_lost_payload_aborts_before_another_stage():
    io = FeedbackIO()

    def retained():
        if io.step_id >= 12:
            raise RuntimeError("payload lost")

    with pytest.raises(RuntimeError, match="payload lost"):
        arm_return.execute_return(io, check=retained)
    assert io.step_id == 12
    assert io.holds >= 2


def test_unknown_joint_feedback_and_unreachable_targets_are_rejected():
    io = FeedbackIO()
    limits = io.model.get_qlimits()[0, list(arm_return.ARM_INDICES)]
    invalid = io.arm.copy()
    invalid[0] = np.nan
    with pytest.raises(ValueError):
        arm_return.return_stages(invalid, 0.6, limits)
    limits[3, 1] = 1.5
    with pytest.raises(ValueError, match="exceeds joint"):
        arm_return.return_stages(io.arm, 0.6, limits)


def test_continuous_joint_turns_use_the_nearest_equivalent_angle():
    error = arm_return.joint_error(
        np.array([0.0, 0.0, -np.pi + 0.02, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, np.pi - 0.02, 0.0, 0.0, 0.0, 0.0]),
    )
    assert error[2] == pytest.approx(0.04)


def test_unreachable_withdrawal_is_rejected_before_moving(monkeypatch):
    io = FeedbackIO()
    matrix = np.eye(4)
    matrix[:3, 3] = [0.65, 0.0, 1.2]
    io.robot.tcp_pose = SimpleNamespace(to_transformation_matrix=lambda: matrix[None].copy())

    class InfeasibleKinematics:
        def __init__(self, robot):
            pass

        def forward(self, joints):
            return matrix.copy()

        def path(self, *args):
            raise arm_return.ArmReturnError("no joint-limit-feasible clearance path")

    monkeypatch.setattr(arm_return, "ArmKinematics", InfeasibleKinematics)
    with pytest.raises(arm_return.ArmReturnError, match="joint-limit-feasible"):
        arm_return.execute_return(io, withdraw=True)
    assert io.step_id == 0
    assert io.holds >= 2


@pytest.mark.parametrize("frequency", [1, 20, 60])
def test_continuous_reference_bounds_curved_paths_and_starts_and_ends_at_rest(frequency):
    if not hasattr(arm_return, "continuous_joint_reference"):
        pytest.skip("Continuous timing is only provided by the derived controller")
    initial = np.zeros(7)
    initial[2] = np.pi - .02
    waypoints = np.zeros((21, 7))
    t = np.linspace(0., 1., 21)
    waypoints[:, 0] = .6 * t
    waypoints[:, 1] = .2 * np.sin(np.pi * t)
    waypoints[:, 2] = -np.pi + .02 * t
    reference = arm_return.continuous_joint_reference(initial, waypoints, frequency)
    assert np.isfinite(reference).all()
    np.testing.assert_allclose(arm_return.joint_error(waypoints[-1], reference[-1]), 0., atol=1e-12)
    assert np.min(reference[:, 2]) > 3.0  # Cross the wrap, never rotate a full turn.
    positions = np.vstack([initial, initial, reference, reference[-1]])
    speed = np.diff(positions, axis=0) * frequency
    acceleration = np.diff(speed, axis=0) * frequency
    assert np.max(np.abs(speed)) <= arm_return.MAX_SPEED + 1e-9
    assert np.max(np.abs(acceleration)) <= arm_return.MAX_ACCELERATION + 1e-9


def test_initial_stow_preserves_scan_cadence_and_cannot_skip_a_real_return():
    if not hasattr(arm_return, "confirm_initial_stow"):
        pytest.skip("Separate startup confirmation is only provided by the derived controller")
    io = FeedbackIO()
    io.model.position = io.model.position.astype(np.float32)
    io.model.position[0, list(arm_return.ARM_INDICES)] = arm_return.TRANSPORT
    io.arm = arm_return.TRANSPORT.copy()
    io.arm_return_stage = None
    io.grip = 1.
    io.robot.set_control_mode = lambda mode: None
    io.robot.controller = SimpleNamespace(reset=lambda: None)
    receipt = arm_return.confirm_initial_stow(io)
    assert receipt["stages"][0]["end_step"] == 40
    assert receipt["arm_return_protocol"] == "feedback_gated"
    assert len(io.commands) == 40  # 20 ramp commands + 20 settling samples.
    assert not arm_return.confirm_initial_stow(io)  # Later calls need the full sequence.
    io.step_id = 0
    io.grip = -1.
    assert not arm_return.confirm_initial_stow(io)
    io.grip = 1.
    io.model.position[0, arm_return.ARM_INDICES[3]] -= .04
    assert not arm_return.confirm_initial_stow(io)
    assert len(io.commands) == 40


def test_mesh_projection_certifies_the_same_clearance_with_transformed_meshes():
    if not hasattr(arm_return.ArmKinematics, "mesh_clearance"):
        pytest.skip("Mesh refinement is only provided by the derived controller")
    from itertools import product

    model = arm_return.ArmKinematics.__new__(arm_return.ArmKinematics)
    vertices = np.asarray(list(product([-.05, .05], repeat=3)))
    normals = np.eye(3)
    model.head_meshes = [(vertices, normals)]
    model.arm_meshes = [("forearm", vertices, normals)]
    frame = np.eye(4)
    frame[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    for gap in [.02, .01, -.01]:
        frame[0, 3] = .10 + gap
        measured = model.mesh_clearance({"forearm": frame})
        assert measured.min() == pytest.approx(gap)
        assert bool(measured.min() >= .015) == (gap == .02)


@pytest.mark.parametrize("loaded", [False, True])
def test_return_uses_the_requested_load_profile_and_keeps_all_settling_windows(loaded):
    if not hasattr(arm_return, "LOADED_SPEED"):
        pytest.skip("Separate speed profiles are only provided by the derived controller")
    io = FeedbackIO()
    io.grip = -1. if loaded else 1.
    receipt = arm_return.execute_return(io)
    limit = arm_return.LOADED_SPEED if loaded else arm_return.MAX_SPEED
    assert receipt["maximum_reference_speed_rad_s"] == limit
    commands = np.asarray([command for _, command in io.commands])
    assert np.max(np.abs(np.diff(commands, axis=0))) * io.base.control_freq <= limit + 1e-9
    assert len(receipt["stages"]) == 7
    assert all(stage["settled"] for stage in receipt["stages"])
    assert receipt["stages"][0]["end_step"] == 6  # Already raised; still confirm 0.3 s.
    assert receipt["settle_window_s"] == 1.
