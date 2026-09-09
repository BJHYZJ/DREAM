#!/usr/bin/env python3
"""Compare circular and heading-dependent pose feasibility on SAVED observed maps.

The heading union is optimistic: it does not certify turns, continuous paths or
robot task success. A learned memory detection supplies the query, never an
environment endpoint or evaluator actor position.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from dream_fetch_navigation import FetchObservedMap
from dream_fetch_footprint import orientation_masks,pose_clear
from houseexpo_cross_room import weighted_distances
from maniskill_learned_probe import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    footprint_path=args.replay/"robot_footprint.json"
    footprint=json.loads(footprint_path.read_text())
    if not footprint["physical_reexecution_passed"] or not footprint["complete_controls_replayed"]:
        raise ValueError("Footprint must follow exact complete physical replay")
    source=Path(footprint["source_run"])
    events=[json.loads(line) for line in (source/"events.jsonl").read_text().splitlines()]
    candidates=[e for e in events if e["event"]=="memory_retrieval" and e.get("detection")]
    if not candidates:raise ValueError("No learned memory query in this recording")
    query_event=candidates[-1];query=np.asarray(query_event["detection"]["point_world"])
    with np.load(source/"observed_occupancy.npz") as data:
        observed=FetchObservedMap(origin=data["origin"],size=len(data["known"]),
                                 resolution=float(data["resolution"]),radius=float(data["radius"]))
        observed.known=data["known"].copy()
    # The older map NPZ omits stall-inferred cells. Reconstruct these only from
    # logged robot controls and controller-stall goals, using the same helper.
    controls={row["step"]:row for row in json.loads((source/"actions.json").read_text())}
    stalls=[e for e in events if e["event"]=="controller_stall"]
    for event in stalls:
        pose=np.asarray(controls[event["step"]]["base_xyyaw"])
        delta=np.asarray(event["goal"])-pose[:2]
        observed.infer_stall_obstacle(pose[:2],float(np.arctan2(delta[1],delta[0])))
    base=np.asarray(footprint["base_xyyaw"]);start=tuple(observed.cells(base[:2]))
    vertices=np.asarray(footprint["vertices_base_xy"]);padding=footprint["padding_m"]
    masks=orientation_masks(observed.known,observed.resolution,vertices,padding=padding,
                            inferred=observed.inferred_blocked)
    union=masks.any(axis=0)
    def summarize(free):
        distances,_=weighted_distances(free,start)
        cells=np.argwhere(free&np.isfinite(distances))
        ranges=np.linalg.norm(observed.world(cells)-query[:2],axis=1)
        return dict(start_valid=bool(free[start]),reachable_cells=len(cells),
                    docking_cells=int(((ranges>=.58)&(ranges<=.64)).sum()),
                    nearest_query_range_m=float(ranges.min()) if len(ranges) else None)
    report=dict(source_run=str(source.resolve()),query_from_actual_memory_event=query_event,
        robot_footprint_sha256=sha(footprint_path),observed_map_sha256=sha(source/"observed_occupancy.npz"),
        events_sha256=sha(source/"events.jsonl"),circle_radius_m=observed.radius,
        controls_sha256=sha(source/"actions.json"),reconstructed_stall_events=len(stalls),
        robot_max_radius_m=footprint["maximum_robot_radius_m"],
        combined_max_radius_m=footprint["maximum_combined_radius_m"],
        footprint_padding_m=padding,original_circle=summarize(observed.traversable()),
        optimistic_heading_union=summarize(union),
        measured_current_pose_clear=pose_clear(observed.known,observed.origin,observed.resolution,
                                               vertices,base[:2],base[2],padding=padding,
                                               inferred=observed.inferred_blocked),
        orientation_counts=[int(m.sum()) for m in masks],
        motion_or_task_success_verified=False,
        boundary="Recorded observed map + exact replayed robot geometry + learned memory query only. Heading-union connectivity is optimistic, not a valid turning/swept path or successful task.")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"diagnostic.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(args.output/"diagnostic_maps.npz",known=observed.known,
        circular=observed.traversable(),heading_masks=masks,inferred_blocked=observed.inferred_blocked,
        origin=observed.origin,resolution=observed.resolution)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
