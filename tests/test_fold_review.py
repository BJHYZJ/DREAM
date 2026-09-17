from copy import deepcopy

import pytest

from dream_sim.fold_review import POSTURES, REST, review_fold


def recording():
    events = [
        dict(
            event=name,
            step=i,
            measured_joint_positions_rad=REST.copy(),
            settled=True,
            settle_window_s=1.0,
            settled_joint_range_rad=0.0001,
        )
        for i, name in enumerate(("carried_arm_folded", "idle_arm_folded"), 1)
    ]
    actions = [dict(step=i, arm_joint_positions_rad=REST.copy()) for i in (1, 2)]
    return events, actions, dict(robot_self_contact_rows=[], pairs=[])


def test_return_requires_both_observed_endpoints():
    events, actions, contacts = recording()
    assert review_fold(events, actions, contacts)["passed"]
    assert not review_fold(events[:1], actions, contacts)["passed"]
    altered = deepcopy(actions)
    altered[0]["arm_joint_positions_rad"][0] += 0.1
    assert not review_fold(events, altered, contacts)["passed"]


def test_return_rejects_self_and_table_contacts_but_allows_placed_object_support():
    events, actions, contacts = recording()
    pair = dict(kind="task_fixture", body_kind="payload", phase="Return arm to rest", substeps=100)
    contacts["pairs"] = [pair]
    assert review_fold(events, actions, contacts)["passed"]
    pair["body_kind"] = "robot"
    assert not review_fold(events, actions, contacts)["passed"]
    contacts["pairs"] = []
    contacts["robot_self_contact_rows"] = [dict(force_n=2.0)]
    assert not review_fold(events, actions, contacts)["passed"]
    del contacts["robot_self_contact_rows"]
    assert not review_fold(events, actions, contacts)["passed"]


@pytest.mark.parametrize(
    "phase",
    ["Lift arm clear before folding", "Fold arm with base stationary", "withdraw", "shorten"],
)
def test_stationary_fold_still_rejects_robot_support_contacts(phase):
    events, actions, contacts = recording()
    contacts["pairs"] = [dict(kind="task_fixture", body_kind="robot", phase=phase, substeps=1)]
    assert not review_fold(events, actions, contacts)["passed"]
    contacts["pairs"][0]["body_kind"] = "payload"
    assert review_fold(events, actions, contacts)["passed"]


@pytest.mark.parametrize(
    "posture_id",
    ["fetch_in_base_v1", "recovery_v2_belly_clearance_v1", "recovery_v2_tighter_belly_v1"],
)
def test_new_tuck_is_versioned_without_changing_historical_pose_checks(posture_id):
    events, actions, contacts = recording()
    assert review_fold(events, actions, contacts)["passed"]
    for event, action in zip(events, actions):
        event["posture_id"] = posture_id
        event["measured_joint_positions_rad"] = POSTURES[posture_id].copy()
        action["arm_joint_positions_rad"] = POSTURES[posture_id].copy()
    assert review_fold(events, actions, contacts)["passed"]
    events[0]["posture_id"] = "legacy_compact"
    assert not review_fold(events, actions, contacts)["passed"]
    events[0]["posture_id"] = "unknown_posture"
    assert not review_fold(events, actions, contacts)["passed"]


def test_feedback_gated_return_requires_recorded_intermediate_arrivals():
    from dream_sim.fold_review import review_return_stages

    arm = POSTURES["recovery_v2_tighter_belly_v1"].copy()
    names = ["raise", "shorten", "turn", "lower", "approach_stow", "align", "stow"]
    stages = [
        dict(
            stage=name,
            start_step=i * 20,
            end_step=(i + 1) * 20,
            settled=True,
            target_joint_positions_rad=arm.copy(),
            measured_joint_positions_rad=arm.copy(),
            settled_joint_range_rad=0.001,
        )
        for i, name in enumerate(names)
    ]
    event = dict(
        arm_return_protocol="feedback_gated", control_frequency_hz=20, step=140, stages=stages
    )
    actions = {
        step: dict(arm_joint_positions_rad=arm.copy(), arm_joint_velocities_rad_s=[0.0] * 7)
        for step in range(1, 141)
    }
    assert review_return_stages(event, actions)
    changed = deepcopy(event)
    changed["stages"][2]["settled"] = False
    assert not review_return_stages(changed, actions)
    changed = deepcopy(actions)
    changed[60]["arm_joint_positions_rad"][0] += 0.1
    assert not review_return_stages(event, changed)
    changed = deepcopy(event)
    del changed["stages"][2]
    assert not review_return_stages(changed, actions)

    changed = deepcopy(actions)
    changed[139]["arm_joint_velocities_rad_s"][0] = 0.2
    assert not review_return_stages(event, changed)
    changed = deepcopy(actions)
    del changed[139]
    assert not review_return_stages(event, changed)
