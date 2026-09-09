"""Independent new-protocol evaluator; never imported by the search policy."""
import itertools

import numpy as np
from learned_evaluation import longest_true_run


def room_at(xy,room_map,core=False):
    row=int(round((room_map["maximum_xy"][1]-xy[1])/float(room_map["resolution"])))
    col=int(round((xy[0]-room_map["minimum_xy"][0])/float(room_map["resolution"])))
    field=room_map["room_cores" if core else "room_labels"]
    return int(field[row,col]) if 0<=row<field.shape[0] and 0<=col<field.shape[1] else 0


def placement_geometry(target,destination,pickup,receptacle,relation):
    from maniskill_learned_probe import array
    corners=np.array(list(itertools.product(*zip(pickup.lower,pickup.upper))))
    transform=array(target.pose.to_transformation_matrix())[0]
    world=corners@transform[:3,:3].T+transform[:3,3]
    origin=array(destination.pose.p)[0]
    center=origin[:2]+(receptacle.lower[:2]+receptacle.upper[:2])/2
    radius=float(np.min(receptacle.upper[:2]-receptacle.lower[:2])/2)*(.82 if relation=="in" else .94)
    inside=bool(np.max(np.linalg.norm(world[:,:2]-center,axis=1))<radius)
    # Contact is checked independently. This height interval additionally rules
    # out an object below a visually overlapping but unsupported receptacle.
    above=bool(np.max(world[:,2])>origin[2]+receptacle.lower[2]+.005)
    below_rim=(relation!="in" or float(world[:,2].mean())<origin[2]+receptacle.upper[2])
    return bool(inside and above and below_rim)


def chronological_reacquisition(events,checks,start,initial):
    """Separate following a moving detection from actually losing/refinding it.

    Revision 2 is stricter than the development scorer. A live moved detection
    in the same observation as invalidation is not a search reacquisition.
    Require a recorded missing detection with observed old-location invalidation,
    then an independently correct moved-target detection in a later observation.
    Record review separately rechecks the invalidation's saved free-depth rays.
    This evaluator output is never supplied to the acting policy.
    """
    moved=sorted((r for r in checks if start is not None and r["step"]>start
        and r["evaluator_correct_identity"] and np.linalg.norm(
            np.asarray(r["detection"]["point_world"][:2])-np.asarray(initial)[:2])>.50),key=lambda r:r["step"])
    losses=sorted((e for e in events if start is not None and e["step"]>start
        and e["event"]=="observed_memory_update" and e.get("task_stage")=="pickup_search"
        and "new_point" in e and e["new_point"] is None),key=lambda e:e["step"])
    loss=next((e for e in losses if any(r["step"]>e["step"] for r in moved)),None)
    reacquired=next((r for r in moved if loss is not None and r["step"]>loss["step"]),None)
    return dict(definition_version=2,first_moved_detection_step=moved[0]["step"] if moved else None,
        observed_loss_step=loss["step"] if loss else None,
        observed_loss_observation_id=loss.get("evidence_observation_id") if loss else None,
        reacquisition_step=reacquired["step"] if reacquired else None,
        reacquisition_observation_id=reacquired.get("observation_id") if reacquired else None,
        strict_loss_then_reacquisition=reacquired is not None,
        boundary="Continuous visual tracking alone does not satisfy lost-and-found. Saved depth/identity and physical replay must also pass.")


def retained_after_lift(rows,reference,hz):
    """Contact transport after a witnessed lift, without a fixed carry height.

    A 1 s lift >6 cm above the initial origin is still required. Once witnessed
    in a continuous bilateral-contact segment, lowering the held arm does not
    make contact disappear. A contact break resets qualification; empty travel
    or an unqualified later grip cannot contribute. This measures retained-grip
    transport, not a guarantee of absence of every environment contact.
    """
    retained=[];lift_run=0;qualified=False
    for row in rows:
        contact=bool(row["bilateral_contact"])
        if not contact:
            lift_run=0;qualified=False
        else:
            lift_run=lift_run+1 if row["target_xyz"][2]>reference+.06 else 0
            qualified=qualified or lift_run>=hz
        retained.append(contact and qualified)
    return np.asarray(retained,dtype=bool)


def stable_endpoint_room(rows,room_map,hz,*,at_start=False):
    """Require 1 s in one wall-partition region, not its eroded clearance core.

    The >=0.60 m-clearance cores construct the room partition and safe starts;
    a legitimate table-side base pose may be outside a core while inside the
    same wall-bounded room. Labels are geometry-derived, not semantic room IDs.
    A one-frame doorway-label crossing alone cannot pass this endpoint test.
    """
    if len(rows)<hz:return 0
    sample=rows[:hz] if at_start else rows[-hz:]
    labels={room_at(r["base_xyyaw"][:2],room_map) for r in sample}
    return next(iter(labels)) if len(labels)==1 and 0 not in labels else 0


def evaluate_instruction(rows,events,actions,move,room_map,*,hz=20):
    if not rows:
        return dict(evaluator_protocol_success=False,reason="no_physics_trace")
    reference=float(np.median([r["target_xyz"][2] for r in rows[:min(100,len(rows))]]))
    held=np.array([r["bilateral_contact"] and r["target_xyz"][2]>reference+.06 for r in rows])
    sustained=longest_true_run(held)/hz
    stable=[r["task_stage"] in ("place","finished") and not r["bilateral_contact"]
            and r["receptacle_contact_force_n"]>.02 and r["inside_receptacle"]
            and np.linalg.norm(r["target_velocity"])<.05 for r in rows]
    stable_seconds=longest_true_run(stable)/hz
    checks=move.observation_checks
    start=move.gate.started_step
    discoveries=[r for r in checks if r["evaluator_correct_identity"] and start is not None and r["step"]<=start]
    temporal=chronological_reacquisition(events,checks,start,move.initial)
    initial_room=room_at(actions[0]["base_xyyaw"][:2],room_map,True) if actions else 0
    grasp_rows=[r for r in rows if r["task_stage"]=="grasp"]
    place_rows=[r for r in rows if r["task_stage"]=="place"]
    grasp_core=room_at(grasp_rows[0]["base_xyyaw"][:2],room_map,True) if grasp_rows else 0
    place_core=room_at(place_rows[-1]["base_xyyaw"][:2],room_map,True) if place_rows else 0
    grasp_room=stable_endpoint_room(grasp_rows,room_map,hz,at_start=True)
    place_room=stable_endpoint_room(place_rows,room_map,hz)
    lost=any(e["event"]=="observed_memory_update" and e.get("task_stage")=="pickup_search"
             and start is not None and e["step"]>start for e in events)
    second_search=any(e["event"]=="placement_search_started" for e in events)
    moving_rows=[r for r in rows if start is not None and start<r["step"]<=
                 (move.done_step if move.done_step is not None else rows[-1]["step"])]
    move_retained_support=bool(moving_rows and all(r["target_xyz"][2]>reference-.10 for r in moving_rows))
    steps=np.array([r["step"] for r in rows])
    xy=np.array([r["base_xyyaw"][:2] for r in rows])
    distances=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    legacy_held_distance=float(distances[held[1:]&held[:-1]].sum())
    retained=retained_after_lift(rows,reference,hz)
    held_distance=float(distances[retained[1:]&retained[:-1]].sum())
    criteria=dict(instruction_before_initial_scan=bool(events and events[0]["event"]=="instruction_received"
                        and events[0]["step"]==0),
        correct_discovery_before_relocation=bool(discoveries),
        relocation_triggered_during_translation=bool(move.trace and move.trace[0]["robot_translating"]),
        external_relocation_finished=move.done_step is not None,
        external_move_preserved_payload_support=move_retained_support,
        observed_stale_check=lost,correct_visual_reacquisition=temporal["strict_loss_then_reacquisition"],
        sustained_physical_lift=sustained>=1.,held_base_transport=held_distance>=.5,
        second_language_search=second_search,stable_released_receptacle_placement=bool(
            len(stable)>=2*hz and all(stable[-2*hz:])),
        cross_room_search=bool(initial_room and grasp_room and initial_room!=grasp_room),
        cross_room_delivery=bool(grasp_room and place_room and grasp_room!=place_room))
    # Continuous correct tracking is legitimate task completion, but cannot
    # fill the user's requested lost-and-found video set. Separate endpoints
    # before the paired study so good tracking is not penalized as a failure.
    task_criteria={key:value for key,value in criteria.items()
                   if key not in ("correct_visual_reacquisition","observed_stale_check")}
    task_criteria["correct_visual_moved_target_observation"]=temporal["first_moved_detection_step"] is not None
    return dict(evaluator_protocol_success=all(criteria.values()),criteria=criteria,
        evaluator_task_success=all(task_criteria.values()),task_criteria=task_criteria,
        sustained_lift_s=sustained,held_base_transport_m=held_distance,stable_placement_s=stable_seconds,
        legacy_height_gated_transport_m=legacy_held_distance,
        legacy_grasp_room_core=grasp_core,legacy_placement_room_core=place_core,
        transport_definition="Bilateral contact after >=1 s witnessed >6 cm lift; reset on contact loss; no constant carry-height requirement",
        room_definition="Wall-only geometry partition propagated from >=0.60 m-clearance cores; stable 1 s grasp/place base-region endpoints, not semantic annotations",
        base_distance_m=float(distances.sum()),initial_room=initial_room,grasp_room=grasp_room,placement_room=place_room,
        relocation_start_step=start,relocation_done_step=move.done_step,
        reacquisition_step=temporal["reacquisition_step"],temporal_evidence=temporal,evaluation_definition_version=4,
        independent_control_contact_audit_pending=True,release_ready=False,
        boundary="New instruction-protocol scoring; native-contact/control re-execution audit is still required")
