#!/usr/bin/env python3
"""List every development attempt; never equate script completion with success."""
import argparse
import json
from pathlib import Path


def summarize(run):
    config=json.loads((run/"configuration.json").read_text())
    result=json.loads((run/"result.json").read_text()) if (run/"result.json").exists() else {}
    events=[json.loads(s) for s in (run/"events.jsonl").read_text().splitlines()] if (run/"events.jsonl").exists() else []
    scheduled=[]
    if (run.parent/"attempts.jsonl").exists():
        scheduled=[json.loads(line) for line in (run.parent/"attempts.jsonl").read_text().splitlines()]
    execution=next((r["status"] for r in scheduled if r["name"]==run.name),None)
    row=dict(run=str(run.resolve()),name=run.name,scene=config["scene"],
             status=execution or ("completed" if result else ("execution_failure" if (run/"failure.json").exists() else "incomplete")),
             cross_room_success=bool(result.get("evaluator_cross_room_success",False)),
             protocol_success=bool(result.get("evaluator_protocol_success",False)),
             entered_destination_core=bool(result.get("evaluator_destination_room_core_entered",False)),
             returned_home_core=bool(result.get("evaluator_returned_to_initial_room_core",False)),
             fresh_visual_approach=bool(result.get("visual_approach_complete",False)),
             sustained_lift_s=result.get("evaluator_sustained_lift_s",0.),
             held_transport_m=result.get("evaluator_held_transport_m",0.),
             stable_placement=bool(result.get("evaluator_stable_placement",False)),
             base_travel_m=result.get("evaluator_base_travel_m",0.),
             last_event=events[-1]["event"] if events else None,
             last_step=events[-1]["step"] if events else 0,
             generation=run.parent.name,
             source_snapshot=str((run/"source_snapshot").resolve()))
    row["eligible_success"]=all(row[k] for k in ("cross_room_success","protocol_success",
        "entered_destination_core","returned_home_core","fresh_visual_approach","stable_placement"))
    row["eligible_success"] &= row["sustained_lift_s"]>=1 and row["held_transport_m"]>=.5
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=Path(__file__).with_name("results"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    candidates=list(args.runs.glob("learned_crossroom_v3_*"))+list(args.runs.glob("crossroom_frozen_v3_*/*"))
    rows=[summarize(p) for p in sorted(candidates) if (p/"configuration.json").exists()]
    selected={}
    for row in rows:
        if row["eligible_success"] and row["scene"] not in selected:
            selected[row["scene"]]=row
    report=dict(boundary="Development ledger. Any selected successes are curated demonstrations, not an unbiased success rate. Incomplete entries need process-status checks.",
                attempts=rows,eligible_distinct_houses=len(selected),selected=list(selected.values()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(attempts=len(rows),eligible_distinct_houses=len(selected),
                         outcomes=[dict(name=r["name"],success=r["eligible_success"],last=r["last_event"]) for r in rows]),indent=2))


if __name__=="__main__":
    main()
