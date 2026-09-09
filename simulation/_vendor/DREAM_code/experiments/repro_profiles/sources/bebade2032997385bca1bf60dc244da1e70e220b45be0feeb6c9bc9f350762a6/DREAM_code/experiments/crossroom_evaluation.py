"""Offline wall-bounded room transition scoring, never policy input."""
import numpy as np


def evaluate_room_transitions(actions,events,task,room_map,query_step):
    labels=room_map["room_labels"]
    cores=room_map["room_cores"]
    minimum=room_map["minimum_xy"]
    maximum=room_map["maximum_xy"]
    resolution=float(room_map["resolution"])
    def region(xy,field):
        row=int(round((maximum[1]-xy[1])/resolution))
        col=int(round((xy[0]-minimum[0])/resolution))
        if row<0 or col<0 or row>=labels.shape[0] or col>=labels.shape[1]:
            return 0
        return int(field[row,col])
    approach=next((event["step"] for event in events if event["event"]=="visual_approach_complete"),None)
    initial=int(task["initial_room"])
    target=int(task["destination_room"])
    rows=[]
    previous=None
    reached_core=False
    returned=False
    for action in actions:
        if action["step"]<query_step:
            continue
        room=region(action["base_xyyaw"][:2],labels)
        core=region(action["base_xyyaw"][:2],cores)
        if approach is not None and action["step"]<=approach and core==target:
            reached_core=True
        if approach is not None and action["step"]>approach and core==initial:
            returned=True
        if room!=previous:
            rows.append(dict(step=action["step"],room=room,base_xy=action["base_xyyaw"][:2]))
            previous=room
    approach_action=next((a for a in actions if a["step"]==approach),None)
    approached_room=region(approach_action["base_xyyaw"][:2],labels) if approach_action else 0
    passed=bool(initial!=target and rows and rows[0]["room"]==initial and
                approached_room==target and reached_core and returned)
    return dict(evaluator_cross_room_success=passed,evaluator_initial_room=initial,
                evaluator_destination_room=target,evaluator_approach_room=approached_room,
                evaluator_destination_room_core_entered=reached_core,
                evaluator_returned_to_initial_room_core=returned,
                evaluator_room_transitions=rows,
                evaluator_room_definition="Wall-only stage partition with doorway-separated >= 0.60 m-clearance room cores; not furniture detours or policy room labels. Crossing also requires returning to the initial room core after approach.")
