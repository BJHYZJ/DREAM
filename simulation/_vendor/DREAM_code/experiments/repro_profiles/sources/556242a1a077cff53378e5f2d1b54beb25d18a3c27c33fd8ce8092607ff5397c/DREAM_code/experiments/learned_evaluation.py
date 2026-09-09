"""Offline, evaluator-only physical checks. Never imported by the policy."""
from __future__ import annotations

import numpy as np


def longest_true_run(values):
    best=current=0
    for value in values:
        current=current+1 if value else 0
        best=max(best,current)
    return best


def evaluate_episode(trajectory,actions,events,bin_xy,*,hz=20,require_delivery=True):
    if not trajectory:
        return dict(evaluator_task_success=False,evaluator_reason="no_physics_trace")
    xyz=np.asarray([r["target_xyz"] for r in trajectory])
    contact=np.array([r.get("bilateral_contact",False) for r in trajectory])
    phases=np.array([r.get("phase","") for r in trajectory])
    pre=(phases=="Build memory")
    reference=float(np.median(xyz[pre,2])) if pre.any() else float(xyz[0,2])
    post=np.isin(phases,["Lift","Carry","Place","Check placement"])
    held_lift=contact & post & (xyz[:,2]>=reference+.06)
    sustained=longest_true_run(held_lift)/hz
    held_indices=np.flatnonzero(held_lift)
    held_travel=0.
    if len(held_indices)>1:
        consecutive=np.diff(held_indices)==1
        held_travel=float(np.linalg.norm(np.diff(xyz[held_indices,:2],axis=0),axis=1)[consecutive].sum())
    tail=trajectory[-hz:]
    placement_stable=len(tail)>=hz and all(
        r.get("phase")=="Check placement" and not r.get("bilateral_contact",False)
        and np.max(np.abs(np.asarray(r["target_xyz"][:2])-bin_xy))<.205
        and .69<r["target_xyz"][2]<.81
        and np.linalg.norm(r.get("target_velocity",[float("inf")]))<.05 for r in tail)
    base=np.array([r["base_xyyaw"][:2] for r in actions])
    base_distance=float(np.linalg.norm(np.diff(base,axis=0),axis=1).sum()) if len(base)>1 else 0.
    approach=any(r["event"]=="visual_approach_complete" for r in events)
    grasp_ok=sustained>=1.
    success=approach and grasp_ok and (not require_delivery or (held_travel>=.5 and placement_stable))
    return dict(evaluator_task_success=bool(success),evaluator_sustained_lift_s=sustained,
        evaluator_peak_lift_m=float(np.max(xyz[post,2])-reference) if post.any() else 0.,
        evaluator_held_transport_m=held_travel,evaluator_stable_placement=bool(placement_stable),
        evaluator_base_travel_m=base_distance,
        evaluator_success_definition="fresh visual approach AND bilateral hold >= 1 s at >= 6 cm lift; delivery additionally requires >= 0.5 m held transport and stable released bin placement >= 1 s")


def evaluate_dynamic_protocol(trajectory,events,*,disturbance_step,query_step,endpoint):
    """Separate valid dynamic-task construction from manipulation success."""
    by_step={r["step"]:r for r in trajectory}
    seen=False
    for event in events:
        if event["event"]!="observation" or event["step"]>=disturbance_step:
            continue
        row=by_step.get(event["step"])
        if row is not None:
            seen |= any(np.linalg.norm(np.asarray(d["point_world"])-row["target_xyz"])<.15
                for d in event.get("detections",[]))
    at_query=by_step.get(query_step)
    if at_query is None:
        return dict(evaluator_initial_target_observed=bool(seen),evaluator_disturbance_valid=False)
    support=np.asarray(at_query["support_xyz"])
    target=np.asarray(at_query["target_xyz"])
    reached=np.linalg.norm(support[:2]-endpoint)<.08
    supported=np.linalg.norm(support[:2]-target[:2])<.16 and .90<target[2]<1.05
    return dict(evaluator_initial_target_observed=bool(seen),
        evaluator_disturbance_valid=bool(reached and supported),
        evaluator_support_endpoint_error_at_query_m=float(np.linalg.norm(support[:2]-endpoint)),
        evaluator_target_on_support_at_query=bool(supported))
