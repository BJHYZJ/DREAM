#!/usr/bin/env python3
"""Construct ten native-house fixtures; this does not run or freeze a study."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ASSIGNMENTS=(
    ("ProcTHOR-Train-283","tomato_bowl_white_table"),
    ("ProcTHOR-Val-632","tomato_bowl_white_table"),
    ("ProcTHOR-Train-7819","bread_plate_white_table"),
    ("ArchitecTHOR-Test-02","bread_plate_white_table"),
    ("ProcTHOR-Train-8361","mug_plate_wooden_table"),
    ("ProcTHOR-Train-5318","mug_plate_wooden_table"),
    # 1579's alternative room core is only 1.728 m²: the actual wooden table
    # cannot fit. Replace it based on geometry, before running these tasks.
    ("ProcTHOR-Train-9031","egg_plate_wooden_table"),
    ("ProcTHOR-Train-5080","egg_plate_wooden_table"),
    ("ProcTHOR-Test-244","egg_bowl_white_table"),
    ("ArchitecTHOR-Val-01","egg_bowl_white_table"),
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--recipes-json",type=Path,default=Path(__file__).parent/"configs/instruction_pickplace_v3/recipes.json")
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    configs=Path(__file__).parent/"configs"

    def one(item):
        index,(scene,recipe)=item
        source_dir=configs/("procthor_crossroom_v3_geometry02" if scene.startswith("ProcTHOR-") else "learned_crossroom_v3_geometry02")
        matches=list(source_dir.glob("*"+scene+"_layout01.json"))
        if len(matches)!=1:return dict(scene=scene,recipe=recipe,status="missing_unique_geometry_reference")
        output=args.output/f"{index+1:02d}_{scene}"
        command=[sys.executable,str(Path(__file__).with_name("prepare_instruction_task.py")),
            "--reference-task",str(matches[0]),"--recipe",recipe,"--recipes-json",str(args.recipes_json),
            "--seed","100","--fit-furniture","--output",str(output)]
        with (args.output/f"{index+1:02d}_{scene}.log").open("x") as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        record=dict(scene=scene,recipe=recipe,returncode=result.returncode,
            status="geometry_constructed_not_executed" if result.returncode==0 else "construction_failure")
        if result.returncode==0:record["task"]=str((output/"task.json").relative_to(args.output))
        print(json.dumps(record),flush=True);return record

    with ThreadPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(one,enumerate(ASSIGNMENTS)))
    report=dict(scope="Development fixture construction only. Houses draw on previously explored native scene geometry; not a held-out benchmark split. No outcomes or robot paths reused.",
        frozen_study_started=False,tasks=rows)
    (args.output/"construction_report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
