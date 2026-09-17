"""Check recorded arm returns and replay contacts for the diverse-object study."""

import math

REST = [-1.4, 1.3, 0.7, 1.8, 0.0, 1.2, 0.0]
POSTURES = {
    "recovery_v2_tighter_belly_v1": [-1.253857, 1.35, 0.4, 1.9, -0.013637, 1.200051, 0.000009],
    "legacy_compact": REST,
    "fetch_in_base_v1": [0.6, 1.48, -0.7, 2.23, 0.0, 1.2, 0.0],
    # User-selected recovery-v2 video, measured control100 +0.005rad clearance.
    "recovery_v2_belly_clearance_v1": [
        -1.228857,
        1.195709,
        0.151547,
        1.6977,
        -0.013637,
        1.200051,
        0.000009,
    ],
}
FOLD_PHASES = {
    "Fold arm for transport",
    "Return arm to rest",
    "Withdraw above the placement surface",
    "Move clear of the table",
    "Lift arm clear before folding",
    "Fold arm with base stationary",
    "withdraw",
    "raise",
    "shorten",
    "turn",
    "lower",
    "approach_stow",
    "align",
    "stow",
}


def review_native_motion(stage, by_step, hz, replay):
    """Independently check every native position and its fresh re-execution."""
    if not isinstance(replay, dict) or replay.get("passed") is not True:
        return False
    end = stage["end_step"]
    count = max(2, math.ceil(hz))
    target = stage["target_joint_positions_rad"]
    previous = by_step.get(end - count, {}).get("arm_joint_positions_rad", [])
    if len(previous) != 7 or not all(map(math.isfinite, previous)):
        return False
    offsets, maximum_speed, total_samples, duration = [], 0., 0, 0.
    dt_first = None
    for step in range(end - count + 1, end + 1):
        if step not in replay.get("control_steps", []):
            return False
        action = by_step.get(step, {})
        record = action.get("arm_joint_motion", {})
        dt = record.get("physics_dt_s", 0)
        if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            return False
        if dt_first is None:
            dt_first = dt
        if dt != dt_first:
            return False
        initial = record.get("initial_joint_positions_rad", [])
        samples = record.get("joint_positions_rad", [])
        if (len(initial) != 7 or not all(map(math.isfinite, initial))
                or max(abs(a - b) for a, b in zip(initial, previous)) > 1e-7
                or len(samples) < 1 or not math.isclose(len(samples) * dt * hz, 1., abs_tol=1e-5)):
            return False
        for joints in [initial, *samples]:
            if len(joints) != 7 or not all(map(math.isfinite, joints)):
                return False
            offset = []
            for i, (actual, goal, before) in enumerate(zip(joints, target, previous)):
                error, change = actual - goal, actual - before
                if i in (2, 4, 6):
                    error = math.atan2(math.sin(error), math.cos(error))
                    change = math.atan2(math.sin(change), math.cos(change))
                maximum_speed = max(maximum_speed, abs(change) / dt)
                if abs(error) >= .025 or maximum_speed >= .08:
                    return False
                offset.append(error)
            offsets.append(offset)
            previous = joints
        recorded = action.get("arm_joint_positions_rad", [])
        if len(recorded) != 7 or max(abs(a - b) for a, b in zip(previous, recorded)) > 1e-7:
            return False
        total_samples += len(samples)
        duration += len(samples) * dt
    spread = max(max(values) - min(values) for values in zip(*offsets))
    return (spread < .002 and duration >= 1. - 1e-5
            and stage.get("native_physics_samples") == total_samples
            and math.isclose(stage.get("measured_window_s", -1), duration, abs_tol=1e-7)
            and math.isclose(stage.get("settled_joint_range_rad", -1), spread, abs_tol=1e-7)
            and math.isclose(stage.get("measured_maximum_speed_rad_s", -1), maximum_speed, abs_tol=1e-7))


def review_return_stages(event, by_step, motion_reexecution=None):
    """Bind every intermediate return receipt to measured control records."""
    if event.get("arm_return_protocol") != "feedback_gated":
        return False
    hz = event.get("control_frequency_hz", 0)
    if not isinstance(hz, (float, int)) or not math.isfinite(hz) or hz < 1:
        return False
    stages = event.get("stages", [])
    names = [stage.get("stage") for stage in stages]
    expected = ["raise", "shorten", "turn", "lower", "approach_stow", "align", "stow"]
    if names and names[0] == "withdraw":
        names = names[1:]
    if names not in (["stow"], expected):
        return False
    previous = None
    for stage in stages:
        start, end = stage.get("start_step"), stage.get("end_step")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            return False
        if previous is not None and start != previous:
            return False
        if end > event["step"] or stage.get("settled") is not True:
            return False
        measured = stage.get("measured_joint_positions_rad", [])
        recorded = by_step.get(end, {}).get("arm_joint_positions_rad", [])
        if len(measured) != 7 or len(recorded) != 7:
            return False
        if not all(math.isfinite(value) for value in measured + recorded):
            return False
        if max(abs(a - b) for a, b in zip(measured, recorded)) >= 1e-5:
            return False
        if stage["stage"] == "withdraw":
            target = stage.get("target_tcp_world_m", [])
            tcp = by_step[end].get("tcp_xyz", [])
            if len(target) != 3 or len(tcp) != 3 or not all(map(math.isfinite, target + tcp)):
                return False
            if math.dist(target, tcp) >= 0.015:
                return False
        else:
            target = stage.get("target_joint_positions_rad", [])
            if len(target) != 7 or not all(map(math.isfinite, target)):
                return False
            errors = [
                abs(math.atan2(math.sin(a - b), math.cos(a - b))) if i in (2, 4, 6) else abs(a - b)
                for i, (a, b) in enumerate(zip(measured, target))
            ]
            if (
                max(errors) >= 0.025
                or not 0 <= stage.get("settled_joint_range_rad", math.inf) < 0.005
            ):
                return False
        final = stage["stage"] == "stow"
        count = max(2, math.ceil(hz * (1.0 if final else 0.3)))
        if end - start < count:
            return False
        measurement = stage.get("settling_measurement", "native_joint_velocity")
        measured_motion = measurement == "joint_positions_at_physics_steps"
        if measurement not in ("native_joint_velocity", "joint_positions_at_physics_steps"):
            return False
        if measured_motion:
            if not final or not review_native_motion(stage, by_step, hz, motion_reexecution):
                return False
        window = []
        for step in range(end - count + 1, end + 1):
            action = by_step.get(step, {})
            joints = action.get("arm_joint_positions_rad", [])
            speeds = action.get("arm_joint_velocities_rad_s", [])
            if len(joints) != 7 or len(speeds) != 7 or not all(map(math.isfinite, joints + speeds)):
                return False
            if not measured_motion and max(map(abs, speeds)) >= (0.08 if final else 0.12):
                return False
            if stage["stage"] == "withdraw":
                tcp = action.get("tcp_xyz", [])
                if (
                    len(tcp) != 3
                    or not all(map(math.isfinite, tcp))
                    or math.dist(target, tcp) >= 0.015
                ):
                    return False
            else:
                for i, (actual, goal) in enumerate(zip(joints, target)):
                    error = actual - goal
                    if i in (2, 4, 6):
                        error = math.atan2(math.sin(error), math.cos(error))
                    if abs(error) >= 0.025:
                        return False
            window.append(joints)
        if stage["stage"] != "withdraw":
            if max(max(values) - min(values) for values in zip(*window)) >= (
                0.002 if final else 0.005
            ):
                return False
        previous = end
    return previous == event["step"]


def review_fold(events, actions, contact, motion_reexecution=None):
    endpoints = {}
    by_step = {row["step"]: row for row in actions}
    for name in ("carried_arm_folded", "idle_arm_folded"):
        matching = [row for row in events if row["event"] == name]
        valid = bool(matching)
        for row in matching:
            if "arm_return_protocol" in row:
                valid &= review_return_stages(row, by_step, motion_reexecution)
            target = POSTURES.get(row.get("posture_id", "legacy_compact"))
            if target is None:
                valid = False
                continue
            measured = row.get("measured_joint_positions_rad", [])
            recorded = by_step.get(row["step"], {}).get("arm_joint_positions_rad", [])
            valid &= len(measured) == len(recorded) == 7
            if len(measured) == len(recorded) == 7:
                valid &= all(math.isfinite(value) for value in measured + recorded)
                valid &= max(abs(a - b) for a, b in zip(measured, recorded)) < 1e-5
                errors = [
                    abs(math.atan2(math.sin(a - b), math.cos(a - b)))
                    if i in (2, 4, 6)
                    else abs(a - b)
                    for i, (a, b) in enumerate(zip(measured, target))
                ]
                valid &= max(errors) < 0.035
            valid &= row.get("settled") is True
            valid &= row.get("settle_window_s", 0) >= 1
            valid &= 0 <= row.get("settled_joint_range_rad", math.inf) < 0.002
        endpoints[name] = bool(valid)
    self_recorded = "robot_self_contact_rows" in contact
    self_rows = contact.get("robot_self_contact_rows", [])
    fixture_pairs = [
        row
        for row in contact.get("pairs", [])
        if row["kind"] == "task_fixture"
        and row["body_kind"] == "robot"
        and row["phase"] in FOLD_PHASES
        and row["substeps"] > 0
    ]
    checks = dict(
        **endpoints,
        self_contact_record_present=self_recorded,
        no_robot_self_contact=self_recorded and not self_rows,
        no_robot_fixture_contact_during_return=not fixture_pairs,
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        robot_self_contact_substeps=len(self_rows),
        return_fixture_contacts=fixture_pairs,
    )
