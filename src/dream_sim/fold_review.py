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
}


def review_fold(events, actions, contact):
    endpoints = {}
    by_step = {row["step"]: row for row in actions}
    for name in ("carried_arm_folded", "idle_arm_folded"):
        matching = [row for row in events if row["event"] == name]
        valid = bool(matching)
        for row in matching:
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
