"""Feedback-gated Fetch arm retraction with stationary-base and self-contact checks.

The sequence first shortens the arm at its current shoulder bearing. Only then
may the shoulder turn toward the transport posture. All targets refer to the
robot's joints; no task-actor pose is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ARM_INDICES = (5, 7, 8, 9, 10, 11, 12)
CONTINUOUS = (2, 4, 6)
TRANSPORT = np.array([-1.253857, 1.35, 0.4, 1.9, -0.013637, 1.200051, 0.000009])
POSTURE_ID = "recovery_v2_tighter_belly_v1"
MAX_SPEED = 0.45
MAX_ACCELERATION = 0.7


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def joint_error(target, measured):
    error = np.asarray(target, dtype=float) - measured
    error[list(CONTINUOUS)] = np.arctan2(
        np.sin(error[list(CONTINUOUS)]), np.cos(error[list(CONTINUOUS)])
    )
    return error


@dataclass(frozen=True)
class ReturnStage:
    name: str
    joints: np.ndarray


def return_stages(initial, shoulder_seed, limits):
    """Construct explicit lift, shorten, turn, lower, and stow targets."""
    initial = np.asarray(initial, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if initial.shape != (7,) or limits.shape != (7, 2) or not np.isfinite(initial).all():
        raise ValueError("Expected seven finite arm positions and seven joint limits")
    if not math.isfinite(shoulder_seed) or np.isnan(limits).any():
        raise ValueError("Invalid shoulder seed or joint limits")
    target = initial + joint_error(TRANSPORT, initial)
    if np.max(np.abs(joint_error(target, initial))) < 0.025:
        return [ReturnStage("stow", target)]
    raised = initial.copy()
    raised[1] = min(initial[1], -0.55)
    shortened = target.copy()
    shortened[0] = initial[0]
    shortened[1] = raised[1]
    shortened[3] = target[3] + 0.15
    shortened[5] = 0.8
    turned = shortened.copy()
    turned[0] = 0.5 * (shoulder_seed + target[0] - 0.25)
    lowered = turned.copy()
    lowered[1] = 0.8
    outside = target.copy()
    outside[0] -= 0.25
    outside[3] -= 0.15
    aligned = outside.copy()
    aligned[0] = target[0]
    stages = [
        ReturnStage("raise", raised),
        ReturnStage("shorten", shortened),
        ReturnStage("turn", turned),
        ReturnStage("lower", lowered),
        ReturnStage("approach_stow", outside),
        ReturnStage("align", aligned),
        ReturnStage("stow", target),
    ]
    for stage in stages:
        for index in range(7):
            if (
                index not in CONTINUOUS
                and not limits[index, 0] <= stage.joints[index] <= limits[index, 1]
            ):
                raise ValueError(f"Arm-return stage {stage.name} exceeds joint {index} limits")
    return stages


class ArmReturnError(RuntimeError):
    """The arm must hold position and the task must not begin transport."""


class SelfContactGuard:
    """Monitor only pairs of robot links at every physics substep.

    This is simulated robot self-contact feedback. Scene objects and task-actor
    handles never enter the controller. Independent replay still checks payload
    retention and contacts with the environment.
    """

    def __init__(self, base):
        self.base = base
        self.links = {link._objs[0].entity: link.name for link in base.agent.robot.get_links()}
        self.original = base._after_simulation_step
        self.stage = "start"
        self.maximum_force = 0.0

    def inspect(self):
        for contact in self.base.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            if not all(entity in self.links for entity in entities):
                continue
            force = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
            force /= float(self.base.scene.px.timestep)
            self.maximum_force = max(self.maximum_force, force)
            if force >= 0.5:
                names = [self.links[entity] for entity in entities]
                raise ArmReturnError(
                    f"Arm return {self.stage}: robot self-contact {names}, {force:.3f} N"
                )

    def after_step(self):
        self.original()
        self.inspect()

    def __enter__(self):
        self.inspect()
        self.base._after_simulation_step = self.after_step
        return self

    def __exit__(self, *_):
        self.base._after_simulation_step = self.original


class ArmKinematics:
    """Robot-description FK and bounded IK; never changes simulator state."""

    def __init__(self, robot):
        import xml.etree.ElementTree as ET

        from scipy.spatial.transform import Rotation

        self.rotation = Rotation
        root = ET.parse(robot.urdf_path).getroot()
        by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
        chain = []
        link = robot.ee_link_name
        while link != "torso_lift_link":
            joint = by_child[link]
            origin = joint.find("origin")
            matrix = np.eye(4)
            if origin is not None:
                matrix[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                matrix[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            index = (
                robot.arm_joint_names.index(joint.get("name"))
                if joint.get("type") != "fixed"
                else None
            )
            axis = (
                np.fromstring(joint.find("axis").get("xyz"), sep=" ") if index is not None else None
            )
            chain.append((matrix, index, axis, link))
            link = joint.find("parent").get("link")
        self.chain = list(reversed(chain))
        self.root = array(robot.torso_lift_link.pose.to_transformation_matrix())[0].copy()

        from scipy.spatial import ConvexHull

        self.obstacles = []
        self.arm_bounds = []
        for robot_link in robot.robot.get_links():
            name = robot_link.name
            if name not in {
                "head_pan_link",
                "head_tilt_link",
                "elbow_flex_link",
                "forearm_roll_link",
                "wrist_flex_link",
                "wrist_roll_link",
                "gripper_link",
            }:
                continue
            for shape in robot_link._objs[0].get_collision_shapes():
                if not hasattr(shape, "vertices"):
                    raise ArmReturnError("Arm return needs convex robot collision geometry")
                vertices = np.asarray(shape.vertices) * np.asarray(getattr(shape, "scale", 1.0))
                transform = shape.local_pose.to_transformation_matrix()
                points = vertices @ transform[:3, :3].T + transform[:3, 3]
                if name.startswith("head_"):
                    world = array(robot_link.pose.to_transformation_matrix())[0]
                    points = points @ world[:3, :3].T + world[:3, 3]
                    self.obstacles.append(ConvexHull(points).equations)
                else:
                    center = (points.min(0) + points.max(0)) / 2
                    radius = float(np.linalg.norm(points[:, 1:] - center[1:], axis=1).max())
                    samples = np.tile(center, (12, 1))
                    samples[:, 0] = np.linspace(points[:, 0].min(), points[:, 0].max(), 12)
                    spacing = (points[:, 0].max() - points[:, 0].min()) / 11
                    self.arm_bounds.append((name, samples, radius + spacing / 2))

    def frames(self, joints):
        matrix = self.root.copy()
        frames = {}
        for origin, index, axis, link in self.chain:
            matrix = matrix @ origin
            if index is not None:
                rotation = np.eye(4)
                rotation[:3, :3] = self.rotation.from_rotvec(axis * joints[index]).as_matrix()
                matrix = matrix @ rotation
            frames[link] = matrix.copy()
        return matrix, frames

    def forward(self, joints):
        return self.frames(joints)[0]

    def clearance(self, frames):
        distances = []
        for link, samples, radius in self.arm_bounds:
            pose = frames[link]
            points = samples @ pose[:3, :3].T + pose[:3, 3]
            for hull in self.obstacles:
                separation = np.max(points @ hull[:, :3].T + hull[:, 3], axis=1) - radius
                distances.append(float(separation.min()))
        return np.asarray(distances)

    def path(self, initial, goal, orientation, limits):
        from scipy.optimize import least_squares

        measured = self.forward(initial)
        lower, upper = limits[:, 0].copy(), limits[:, 1].copy()
        lower[list(CONTINUOUS)] = initial[list(CONTINUOUS)] - math.pi
        upper[list(CONTINUOUS)] = initial[list(CONTINUOUS)] + math.pi
        lower += 0.008
        upper -= 0.008
        previous = np.clip(initial, lower + 1e-6, upper - 1e-6)
        count = max(1, math.ceil(np.linalg.norm(goal - measured[:3, 3]) / 0.01))
        result = []
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            position = measured[:3, 3] * (1 - fraction) + goal * fraction

            def residual(joints):
                pose, frames = self.frames(joints)
                angle = self.rotation.from_matrix(orientation.T @ pose[:3, :3]).as_rotvec()
                return np.r_[
                    pose[:3, 3] - position,
                    0.2 * angle,
                    0.002 * (joints - previous),
                    3.0 * np.minimum(0.0, self.clearance(frames) - 0.02),
                ]

            solved = least_squares(
                residual,
                previous,
                bounds=(lower, upper),
                max_nfev=120,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            pose = self.forward(solved.x)
            if (
                np.linalg.norm(pose[:3, 3] - position) > 0.004
                or np.linalg.norm(
                    self.rotation.from_matrix(orientation.T @ pose[:3, :3]).as_rotvec()
                )
                > 0.025
                or np.min(self.clearance(self.frames(solved.x)[1])) < 0.015
            ):
                raise ArmReturnError("Arm return withdraw: no joint-limit-feasible clearance path")
            previous = solved.x.copy()
            result.append(previous)
        return result


def execute_return(
    io,
    check=None,
    *,
    payload_radius=0.0,
    payload_height=0.0,
    withdraw=False,
    minimum_tcp_height=None,
):
    """Advance only after each measured intermediate posture has settled."""
    hz = float(io.base.control_freq)
    if not math.isfinite(hz) or hz < 1:
        raise ValueError("Expected a positive control frequency")
    indices = list(ARM_INDICES)

    def q():
        return array(io.robot.robot.get_qpos())[0, indices].astype(float, copy=True)

    def velocity():
        return array(io.robot.robot.get_qvel())[0, indices]

    start_base = io.pose().copy()
    if (
        not np.isfinite([payload_radius, payload_height]).all()
        or min(payload_radius, payload_height) < 0
    ):
        raise ValueError("Payload dimensions must be finite and nonnegative")
    start_step = io.step_id
    maximum_error = 0.0
    receipts = []
    io.hold_measured_arm()

    def check_state(stage):
        if check is not None:
            check()
        measured = q()
        speed = velocity()
        if not np.isfinite(measured).all() or not np.isfinite(speed).all():
            raise ArmReturnError(f"Arm return {stage}: invalid joint feedback")
        base = io.pose()
        if not np.isfinite(base).all():
            raise ArmReturnError(f"Arm return {stage}: invalid base feedback")
        drift = float(np.linalg.norm(base[:2] - start_base[:2]))
        yaw = abs(math.atan2(math.sin(base[2] - start_base[2]), math.cos(base[2] - start_base[2])))
        if drift > 0.02 or yaw > 0.03:
            raise ArmReturnError(f"Arm return {stage}: base moved before stow completion")
        return measured, speed

    try:
        with SelfContactGuard(io.base) as guard:
            if withdraw:
                guard.stage = "withdraw"
                io.arm_return_stage = "withdraw"
                matrix = array(io.robot.tcp_pose.to_transformation_matrix())[0].copy()
                position = matrix[:3, 3]
                relative = position[:2] - start_base[:2]
                reach = float(np.linalg.norm(relative))
                minimum_reach = max(
                    0.42, 0.30 + math.hypot(payload_radius, payload_height / 2) + 0.025
                )
                if reach > minimum_reach + 0.02:
                    goal = position.copy()
                    floor = (
                        position[2] - 0.025
                        if minimum_tcp_height is None
                        else float(minimum_tcp_height)
                    )
                    if not math.isfinite(floor) or position[2] < floor:
                        raise ArmReturnError(
                            "Arm return withdraw: insufficient observed support clearance"
                        )
                    if minimum_tcp_height is not None:
                        goal[2] = min(position[2], floor + 0.04)
                    goal[:2] = start_base[:2] + relative * minimum_reach / reach
                    model = ArmKinematics(io.robot)
                    if np.linalg.norm(model.forward(q()) - matrix) > 0.005:
                        raise ArmReturnError(
                            "Arm return withdraw: robot kinematics disagree with measured pose"
                        )
                    path = model.path(
                        q(), goal, matrix[:3, :3], array(io.robot.robot.get_qlimits())[0, indices]
                    )
                    stage_start = io.step_id
                    for waypoint in path:
                        origin = q()
                        delta = joint_error(waypoint, origin)
                        distance = float(np.max(np.abs(delta)))
                        duration = max(
                            0.25,
                            1.875 * distance / MAX_SPEED,
                            math.sqrt(5.774 * distance / MAX_ACCELERATION),
                        )
                        steps = math.ceil(duration * hz)
                        for tick in range(1, steps + 1):
                            t = tick / steps
                            io.arm = origin + delta * (10 * t**3 - 15 * t**4 + 6 * t**5)
                            io.command()
                            actual, _ = check_state("withdraw")
                            current = array(io.robot.tcp_pose.to_transformation_matrix())[0]
                            if current[2, 3] < floor:
                                raise ArmReturnError(
                                    "Arm return withdraw: lost vertical clearance above the support"
                                )
                            if np.max(np.abs(joint_error(io.arm, actual))) > 0.12:
                                raise ArmReturnError("Arm return withdraw: joint tracking error")
                    settled_controls = 0
                    for _ in range(math.ceil(5 * hz)):
                        io.arm = path[-1].copy()
                        io.command()
                        _, speed = check_state("withdraw")
                        current = array(io.robot.tcp_pose.to_transformation_matrix())[0]
                        residual = float(np.linalg.norm(current[:3, 3] - goal))
                        orientation_error = math.acos(
                            float(
                                np.clip(
                                    (np.trace(matrix[:3, :3].T @ current[:3, :3]) - 1) / 2, -1, 1
                                )
                            )
                        )
                        if current[2, 3] < floor:
                            raise ArmReturnError(
                                "Arm return withdraw: lost vertical clearance above the support"
                            )
                        if (
                            residual < 0.015
                            and orientation_error < 0.06
                            and np.max(np.abs(speed)) < 0.12
                        ):
                            settled_controls += 1
                        else:
                            settled_controls = 0
                        if settled_controls >= math.ceil(0.3 * hz):
                            break
                    else:
                        raise ArmReturnError("Arm return withdraw: clearance pose did not settle")
                    io.hold_measured_arm()
                    receipts.append(
                        dict(
                            stage="withdraw",
                            start_step=stage_start,
                            end_step=io.step_id,
                            target_tcp_world_m=goal.tolist(),
                            measured_tcp_world_m=current[:3, 3].tolist(),
                            minimum_tcp_height_m=floor,
                            measured_joint_positions_rad=q().tolist(),
                            settled=True,
                        )
                    )
            stages = return_stages(
                q(), float(io.grasp_seed_arm[0]), array(io.robot.robot.get_qlimits())[0, indices]
            )
            for stage in stages:
                guard.stage = stage.name
                io.arm_return_stage = stage.name
                measured, _ = check_state(stage.name)
                target = measured + joint_error(stage.joints, measured)
                delta = target - measured
                distance = float(np.max(np.abs(delta)))
                duration = max(
                    1.0,
                    1.875 * distance / MAX_SPEED,
                    math.sqrt(5.774 * distance / MAX_ACCELERATION),
                )
                steps = max(1, math.ceil(duration * hz))
                stage_start = io.step_id
                for tick in range(1, steps + 1):
                    t = tick / steps
                    io.arm = measured + delta * (10 * t**3 - 15 * t**4 + 6 * t**5)
                    io.command()
                    actual, _ = check_state(stage.name)
                    error = float(np.max(np.abs(joint_error(io.arm, actual))))
                    maximum_error = max(maximum_error, error)
                    if error > 0.12:
                        raise ArmReturnError(
                            f"Arm return {stage.name}: tracking error {error:.4f} rad"
                        )
                final = stage.name == "stow"
                required = max(2, math.ceil((1.0 if final else 0.3) * hz))
                window = []
                history = []
                for _ in range(math.ceil(5 * hz)):
                    io.arm = target.copy()
                    io.command()
                    actual, speed = check_state(stage.name)
                    error = float(np.max(np.abs(joint_error(target, actual))))
                    history.append(actual.copy())
                    history = history[-max(2, math.ceil(hz)) :]
                    if error < 0.025 and np.max(np.abs(speed)) < (0.08 if final else 0.12):
                        window.append(actual.copy())
                        window = window[-required:]
                    else:
                        window.clear()
                    spread = (
                        float(np.max(np.ptp(np.asarray(window), axis=0))) if window else math.inf
                    )
                    if len(window) == required and spread < (0.002 if final else 0.005):
                        break
                    if (
                        len(history) >= math.ceil(hz)
                        and error > 0.04
                        and np.max(np.ptp(history, axis=0)) < 0.003
                    ):
                        raise ArmReturnError(
                            f"Arm return {stage.name}: stalled before the intermediate posture"
                        )
                else:
                    raise ArmReturnError(
                        f"Arm return {stage.name}: intermediate posture did not settle"
                    )
                receipts.append(
                    dict(
                        stage=stage.name,
                        start_step=stage_start,
                        end_step=io.step_id,
                        target_joint_positions_rad=target.tolist(),
                        measured_joint_positions_rad=actual.tolist(),
                        settled_joint_range_rad=spread,
                        settled=True,
                    )
                )
            return dict(
                posture_id=POSTURE_ID,
                target_joint_positions_rad=target.tolist(),
                measured_joint_positions_rad=actual.tolist(),
                maximum_tracking_error_rad=maximum_error,
                planned_duration_s=(io.step_id - start_step) / hz,
                maximum_reference_speed_rad_s=MAX_SPEED,
                settled_joint_range_rad=spread,
                settle_window_s=1.0,
                native_endpoint_velocity_rad_s=velocity().tolist(),
                settled=True,
                control_frequency_hz=hz,
                stages=receipts,
                arm_return_protocol="feedback_gated",
                maximum_self_contact_force_n=guard.maximum_force,
                all_intermediate_postures_settled=True,
            )
    finally:
        io.hold_measured_arm()
        io.arm_return_stage = None
