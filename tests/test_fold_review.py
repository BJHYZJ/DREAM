from copy import deepcopy

from dream_sim.fold_review import REST, review_fold


def recording():
    events = [dict(event=name, step=i, measured_joint_positions_rad=REST.copy(),
                   settled=True, settle_window_s=1., settled_joint_range_rad=.0001)
              for i, name in enumerate(('carried_arm_folded', 'idle_arm_folded'), 1)]
    actions = [dict(step=i, arm_joint_positions_rad=REST.copy()) for i in (1, 2)]
    return events, actions, dict(robot_self_contact_rows=[], pairs=[])


def test_return_requires_both_observed_endpoints():
    events, actions, contacts = recording()
    assert review_fold(events, actions, contacts)['passed']
    assert not review_fold(events[:1], actions, contacts)['passed']
    altered = deepcopy(actions)
    altered[0]['arm_joint_positions_rad'][0] += .1
    assert not review_fold(events, altered, contacts)['passed']


def test_return_rejects_self_and_table_contacts_but_allows_placed_object_support():
    events, actions, contacts = recording()
    pair = dict(kind='task_fixture', body_kind='payload', phase='Return arm to rest', substeps=100)
    contacts['pairs'] = [pair]
    assert review_fold(events, actions, contacts)['passed']
    pair['body_kind'] = 'robot'
    assert not review_fold(events, actions, contacts)['passed']
    contacts['pairs'] = []
    contacts['robot_self_contact_rows'] = [dict(force_n=2.)]
    assert not review_fold(events, actions, contacts)['passed']
    del contacts['robot_self_contact_rows']
    assert not review_fold(events, actions, contacts)['passed']
