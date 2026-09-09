#!/usr/bin/env python3
"""Generate only measured TeX numbers/rows and copy recorded figures."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery",type=Path,required=True)
    p.add_argument("--figures",type=Path,required=True)
    p.add_argument("--visual-review",type=Path,required=True)
    p.add_argument("--manuscript",type=Path,required=True)
    args=p.parse_args()
    report=json.loads((args.gallery/"audited_successes.json").read_text())
    cases=report["cases"]
    visual=json.loads((args.visual_review/"video_review_measurements.json").read_text())
    carry={r["source_run"]:r for r in visual["cases"]}
    for case in cases:
        if not carry[case["source_run"]]["full_video_decode_passed"]:
            raise ValueError("Failed complete-video verification")
    travel=[r["metrics"]["evaluator_base_travel_m"] for r in cases]
    numbers={"CrossroomVideoCount":str(len(cases)),"CrossroomTravelMin":f"{min(travel):.2f}",
             "CrossroomTravelMax":f"{max(travel):.2f}"}
    text="% Generated from audited_successes.json; do not hand-edit measured numbers.\n"
    text+="\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in numbers.items())+"\n"
    (args.manuscript/"crossroom_metrics.tex").write_text(text)
    lines=[r"\begin{table}[H]",r"\centering",
        r"\caption{\latestchange{Curated successful cross-room demonstrations, not an unbiased success-rate sample. Travel is total base path length; carry is base translation with bilateral contact during return, excluding arm-only object motion. Time includes scan, disturbance and manipulation in simulated seconds.}}",
        r"\label{tab:crossroom_cases}",r"\begingroup\color{red}\scriptsize",
        r"\begin{tabular}{clrrrr}",r"\toprule",
        r"Video & House & Travel (m) & Time (s) & Carry (m) & Lift hold (s) \\",r"\midrule"]
    for i,r in enumerate(cases,1):
        m=r["metrics"]
        lines.append(f"{i} & {r['scene']} & {m['evaluator_base_travel_m']:.2f} & {r['video']['duration_s']:.1f} & "
                     f"{carry[r['source_run']]['measured_bilateral_carry_base_distance_m']:.2f} & {m['evaluator_sustained_lift_s']:.1f} "+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\endgroup",r"\end{table}"])
    (args.manuscript/"crossroom_rows.tex").write_text("\n".join(lines)+"\n")
    for name in ("crossroom_layouts.pdf","crossroom_observation_sequence.pdf","crossroom_contact_timeline.pdf"):
        shutil.copy2(args.figures/name,args.manuscript/"figs"/name)
    shutil.copy2(args.figures/"figure_provenance.json",args.manuscript/"crossroom_figure_provenance.json")
    shutil.copy2(args.visual_review/"video_review_measurements.json",args.manuscript/"crossroom_video_review.json")
    print(json.dumps(dict(cases=len(cases),partial_preview=report["partial_preview"],manuscript=str(args.manuscript))))


if __name__=="__main__":
    main()
