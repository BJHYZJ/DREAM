"""Feedback-gated Fetch arm retraction with stationary-base and self-contact checks.

The sequence first shortens the arm at its current shoulder bearing. Only then
may the shoulder turn toward the transport posture. All targets refer to the
robot's joints; no task-actor pose is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from instruction_arm_motion import JointMotionSampler

ARM_INDICES = (5, 7, 8, 9, 10, 11, 12)
CONTINUOUS = (2, 4, 6)
TRANSPORT = np.array([-1.253857, 1.35, 0.4, 1.9, -0.013637, 1.200051, 0.000009])
POSTURE_ID = "recovery_v2_tighter_belly_v1"
MAX_SPEED = 1.1
MAX_ACCELERATION = 2.0
LOADED_SPEED = 0.85
LOADED_ACCELERATION = 1.5


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


def continuous_joint_reference(
    initial, waypoints, frequency, *, max_speed=MAX_SPEED, max_acceleration=MAX_ACCELERATION,
    joint_limits=None
):
    """Time a continuous IK curve with local joint speed/acceleration bounds.

    Slow down locally at bends rather than imposing the sharpest bend's speed
    on the entire withdrawal. Half the acceleration budget covers curvature;
    the other half covers changes in speed along the curve.
    """
    from scipy.interpolate import CubicSpline, PchipInterpolator

    if (not np.isfinite([max_speed, max_acceleration]).all()
            or min(max_speed, max_acceleration) <= 0):
        raise ValueError("Expected positive finite motion limits")
    points = [np.asarray(initial, dtype=float)]
    for waypoint in waypoints:
        points.append(points[-1] + joint_error(waypoint, points[-1]))
    points = np.asarray(points)
    if points.shape[0] < 2 or points.shape[1:] != (7,) or not np.isfinite(points).all():
        raise ValueError("Expected an initial arm pose and finite IK waypoints")
    if not math.isfinite(frequency) or frequency < 1:
        raise ValueError("Expected a positive control frequency")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    if len(points) == 1:
        return np.repeat(points, 2, axis=0)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    curve = CubicSpline(distance / distance[-1], points, axis=0, bc_type="natural")
    parameter = np.linspace(0., 1., max(4097, len(points) * 128))
    if joint_limits is not None:
        bounds = np.asarray(joint_limits, dtype=float)
        if bounds.shape != (7, 2) or np.isnan(bounds).any():
            raise ValueError("Expected seven joint limits")
        bounded = [i for i in range(7) if i not in CONTINUOUS]
        samples = curve(parameter)[:, bounded]
        if (np.any(samples < bounds[bounded, 0] + .008)
                or np.any(samples > bounds[bounded, 1] - .008)):
            # Cubic interpolation can overshoot safe IK knots near a limit.
            # Preserve coordinate extrema, then apply the same timing and full
            # geometry checks. Never clip controls or reduce the safety margin.
            curve = PchipInterpolator(distance / distance[-1], points, axis=0)
    ds = np.diff(parameter)
    tangent = np.max(np.abs(curve(parameter, 1)), axis=1)
    curvature = np.max(np.abs(curve(parameter, 2)), axis=1)
    speed = np.minimum(max_speed / np.maximum(tangent, 1e-12),
                       np.sqrt(.5 * max_acceleration / np.maximum(curvature, 1e-12)))
    acceleration = .5 * max_acceleration / np.maximum(tangent, 1e-12)
    interval_limit = np.minimum(acceleration[:-1], acceleration[1:])
    speed[0] = speed[-1] = 0.
    for i in range(len(ds)):
        speed[i + 1] = min(speed[i + 1], math.sqrt(speed[i]**2 + 2 * interval_limit[i] * ds[i]))
    for i in range(len(ds) - 1, -1, -1):
        speed[i] = min(speed[i], math.sqrt(speed[i + 1]**2 + 2 * interval_limit[i] * ds[i]))
    times = np.r_[0., np.cumsum(2 * ds / (speed[:-1] + speed[1:]))]
    interval_acceleration = (speed[1:]**2 - speed[:-1]**2) / (2 * ds)
    steps = max(2, math.ceil(1.05 * times[-1] * frequency))
    while True:
        sample_times = np.linspace(0., times[-1], steps + 1)
        index = np.clip(np.searchsorted(times, sample_times, side="right") - 1, 0, len(ds) - 1)
        elapsed = sample_times - times[index]
        at = parameter[index] + speed[index] * elapsed + .5 * interval_acceleration[index] * elapsed**2
        positions = curve(np.clip(at, 0., 1.))
        positions[0], positions[-1] = points[0], points[-1]
        # Include stationary endpoints to check transitions into/out of motion.
        velocity = np.diff(np.vstack([positions[0], positions, positions[-1]]), axis=0) * frequency
        actual_acceleration = np.diff(velocity, axis=0) * frequency
        ratio = max(float(np.max(np.abs(velocity))) / max_speed,
                    math.sqrt(float(np.max(np.abs(actual_acceleration))) / max_acceleration))
        if ratio <= 1.:
            return positions[1:]
        steps = max(steps + 1, math.ceil(steps * ratio * 1.05))


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


def confirm_initial_stow(io):
    """Confirm an empty, already compact startup arm without a return sequence.

    Start the one-second settling window after the ramp, preserving the
    original staged controller's 40-control startup before camera scanning.
    Every later retraction goes through execute_return and its intermediate states.
    """
    indices = list(ARM_INDICES)
    initial = array(io.robot.robot.get_qpos())[0, indices].copy()
    if (io.step_id != 0 or io.grip <= 0 or not np.isfinite(initial).all()
            or np.max(np.abs(joint_error(TRANSPORT, initial))) >= .025):
        return False
    hz = float(io.base.control_freq)
    if not math.isfinite(hz) or hz < 1:
        raise ValueError("Expected a positive control frequency")
    target = TRANSPORT.copy()
    # Retain scalar arithmetic at startup: rounding changes the observations
    # when initialization is followed by deterministic scan/motion schedules.
    for index in CONTINUOUS:
        delta = target[index] - initial[index]
        target[index] = initial[index] + math.atan2(math.sin(delta), math.cos(delta))
    io.robot.set_control_mode("pd_joint_pos")
    io.robot.controller.reset()
    io.arm = initial.copy()
    start_base = io.pose().copy()
    maximum_error = 0.0

    def feedback():
        nonlocal maximum_error
        measured = array(io.robot.robot.get_qpos())[0, indices]
        speed = array(io.robot.robot.get_qvel())[0, indices]
        base = io.pose()
        yaw = math.atan2(math.sin(base[2] - start_base[2]), math.cos(base[2] - start_base[2]))
        if (not np.isfinite(measured).all() or not np.isfinite(speed).all()
                or not np.isfinite(base).all()
                or np.linalg.norm(base[:2] - start_base[:2]) > .02 or abs(yaw) > .03):
            raise ArmReturnError("Initial stow: invalid or moving robot feedback")
        error = float(np.max(np.abs(joint_error(io.arm, measured))))
        maximum_error = max(maximum_error, error)
        if error > .12:
            raise ArmReturnError("Initial stow: joint tracking error")
        return measured, speed

    try:
        with SelfContactGuard(io.base) as guard:
            steps = math.ceil(hz)
            for tick in range(1, steps + 1):
                t = tick / steps
                blend = 10 * t**3 - 15 * t**4 + 6 * t**5
                io.arm = initial + (target - initial) * blend
                io.command()
                feedback()
            # Keep the original observation start time: twenty ramp controls
            # followed by twenty complete settling controls at 20 Hz.
            io.command()
            window = []
            required = max(2, math.ceil(hz))
            for _ in range(math.ceil(5 * hz)):
                measured, speed = feedback()
                if (np.max(np.abs(joint_error(target, measured))) < .025
                        and np.max(np.abs(speed)) < .08):
                    window.append(measured.copy())
                    window = window[-required:]
                else:
                    window.clear()
                if (len(window) == required
                        and np.max(np.ptp(np.asarray(window), axis=0)) < .002):
                    spread = float(np.max(np.ptp(np.asarray(window), axis=0)))
                    return dict(
                        posture_id=POSTURE_ID,
                        target_joint_positions_rad=target.tolist(),
                        measured_joint_positions_rad=measured.tolist(),
                        maximum_tracking_error_rad=maximum_error,
                        planned_duration_s=io.step_id / hz,
                        maximum_reference_speed_rad_s=MAX_SPEED,
                        settled_joint_range_rad=spread,
                        settle_window_s=1.0,
                        native_endpoint_velocity_rad_s=speed.tolist(),
                        settled=True,
                        control_frequency_hz=hz,
                        stages=[dict(stage="stow", start_step=0, end_step=io.step_id,
                                     target_joint_positions_rad=target.tolist(),
                                     measured_joint_positions_rad=measured.tolist(),
                                     settled_joint_range_rad=spread, settled=True)],
                        arm_return_protocol="feedback_gated",
                        maximum_self_contact_force_n=guard.maximum_force,
                        all_intermediate_postures_settled=True,
                    )
                io.command()
            raise ArmReturnError("Initial stow: compact posture did not settle")
    finally:
        io.hold_measured_arm()


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
        self.head_meshes = []
        self.arm_meshes = []
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
                    hull = ConvexHull(points)
                    self.obstacles.append(hull.equations)
                    self.head_meshes.append((points, hull.equations[:, :3]))
                else:
                    hull = ConvexHull(points)
                    self.arm_meshes.append((name, points, hull.equations[:, :3]))
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

    def mesh_clearance(self, frames):
        """Certify a lower distance bound using actual convex collision meshes.

        Capsule bounds can overlap the head even when the meshes are separate,
        particularly at a raised grasp pose. A positive gap between projections
        on any unit axis certifies separation; an inconclusive projection still
        rejects the path. No physics state is assigned or modified here.
        """
        distances = []
        for name, vertices, normals in self.arm_meshes:
            pose = frames[name]
            arm = vertices @ pose[:3, :3].T + pose[:3, 3]
            arm_normals = normals @ pose[:3, :3].T
            for head, head_normals in self.head_meshes:
                axes = np.vstack([arm_normals, head_normals])
                axes /= np.linalg.norm(axes, axis=1)[:, None]
                arm_projection, head_projection = arm @ axes.T, head @ axes.T
                gaps = np.maximum(head_projection.min(0) - arm_projection.max(0),
                                  arm_projection.min(0) - head_projection.max(0))
                distances.append(float(gaps.max()))
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
    loaded = payload_radius > 0 or payload_height > 0 or getattr(io, "grip", 1.) < 0
    speed_limit = LOADED_SPEED if loaded else MAX_SPEED
    acceleration_limit = LOADED_ACCELERATION if loaded else MAX_ACCELERATION
    start_step = io.step_id
    maximum_error = 0.0
    receipts = []
    io.hold_measured_arm()

    motion = None

    def check_state(stage):
        if motion is not None and io.step_id in motion.records:
            io.trace[-1]["arm_joint_motion"] = motion.records[io.step_id]
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
        with SelfContactGuard(io.base) as guard, JointMotionSampler(io) as motion:
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
                    limits = array(io.robot.robot.get_qlimits())[0, indices]
                    # A low inward target may exceed the elbow's feasible
                    # range. Try the same withdrawal at the measured height
                    # before reporting an unreachable path. Plan every candidate
                    # fully before motion; no simulator state is changed by IK.
                    candidates = [goal.copy()]
                    if abs(goal[2] - position[2]) > 0.001:
                        level_goal = goal.copy()
                        level_goal[2] = position[2]
                        candidates.append(level_goal)
                    error = None
                    for candidate in candidates:
                        try:
                            path = model.path(q(), candidate, matrix[:3, :3], limits)
                            reference = continuous_joint_reference(
                                q(), path, hz, max_speed=speed_limit,
                                max_acceleration=acceleration_limit, joint_limits=limits,
                            )
                            for waypoint in reference:
                                pose, frames = model.frames(waypoint)
                                bounded = [i for i in range(7) if i not in CONTINUOUS]
                                if (np.any(waypoint[bounded] < limits[bounded, 0] + .008)
                                        or np.any(waypoint[bounded] > limits[bounded, 1] - .008)
                                        or pose[2, 3] < floor + .01
                                        or (np.min(model.clearance(frames)) < .015
                                            and np.min(model.mesh_clearance(frames)) < .015)):
                                    raise ArmReturnError(
                                        "Arm return withdraw: continuous path violates clearance"
                                    )
                        except ArmReturnError as failure:
                            error = failure
                            continue
                        goal = candidate
                        break
                    else:
                        raise error
                    stage_start = io.step_id
                    for waypoint in reference:
                        io.arm = waypoint.copy()
                        io.command()
                        actual, _ = check_state("withdraw")
                        current = array(io.robot.tcp_pose.to_transformation_matrix())[0]
                        if current[2, 3] < floor:
                            raise ArmReturnError(
                                "Arm return withdraw: lost vertical clearance above the support"
                            )
                        if np.max(np.abs(joint_error(io.arm, actual))) > .12:
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
                            reference_interpolation="cubic_with_shape_preserving_limit_fallback",
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
                    0.2,
                    1.875 * distance / speed_limit,
                    math.sqrt(5.774 * distance / acceleration_limit),
                )
                # A state already reached still gets its complete feedback
                # window, without spending another second commanding no motion.
                steps = 0 if distance < 1e-6 else max(1, math.ceil(duration * hz))
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
                motion_receipt = {}
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
                    # Preserve the native-velocity path and its timing. Only
                    # after its complete timeout, verify a stationary final
                    # pose directly at every physics step (TGS split impulse).
                    motion_receipt = motion.settling_window(io.step_id, target, hz) if final else None
                    if motion_receipt is None:
                        raise ArmReturnError(
                            f"Arm return {stage.name}: intermediate posture did not settle"
                        )
                    spread = motion_receipt["settled_joint_range_rad"]
                receipts.append(
                    dict(
                        stage=stage.name,
                        start_step=stage_start,
                        end_step=io.step_id,
                        target_joint_positions_rad=target.tolist(),
                        measured_joint_positions_rad=actual.tolist(),
                        settled_joint_range_rad=spread,
                        settled=True,
                        **{key: value for key, value in motion_receipt.items()
                           if key != "settled_joint_range_rad"},
                    )
                )
            return dict(
                posture_id=POSTURE_ID,
                target_joint_positions_rad=target.tolist(),
                measured_joint_positions_rad=actual.tolist(),
                maximum_tracking_error_rad=maximum_error,
                planned_duration_s=(io.step_id - start_step) / hz,
                maximum_reference_speed_rad_s=speed_limit,
                maximum_reference_acceleration_rad_s2=acceleration_limit,
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
