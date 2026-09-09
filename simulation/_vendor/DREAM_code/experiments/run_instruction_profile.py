#!/usr/bin/env python3
"""Validate and reproduce a versioned instruction profile; dry run by default.

Historical video profiles and the one-controller paired study are different
workflows. This launches learned-policy tasks, not prerecorded robot paths.
Use --execute only after installing and verifying the declared assets/models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child(root, relative):
    path=(root/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Profile path escapes its catalog")
    return path


def load_catalog(path):
    path=Path(path).resolve();data=json.loads(path.read_text());root=path.parent
    if data.get("schema_version")!=1 or data.get("scope")!="versioned_instruction_reproduction_profiles":
        raise ValueError("Not a versioned instruction-profile catalog")
    cases=data["cases"]
    if len(cases)!=10 or len({c["id"] for c in cases})!=10 or len({c["scene"] for c in cases})!=10:
        raise ValueError("Expected ten uniquely identified, distinct-house profiles")
    if len({c["recipe"] for c in cases})!=5:
        raise ValueError("Expected five declared object/receptacle recipes")
    for entry in data["sources"].values():
        manifest=child(root,entry["manifest"])
        if sha(manifest)!=entry["manifest_sha256"]:
            raise ValueError("Source manifest changed")
    for key in ("model_lock","asset_lock"):
        entry=data[key]
        if sha(child(root,entry["file"]))!=entry["sha256"]:
            raise ValueError(f"Catalog {key} changed")
    study=data["study"]
    if sha(child(root,study["recorded_protocol"]))!=study["recorded_protocol_sha256"]:
        raise ValueError("Recorded study protocol changed")
    return root,data


def verify_source(root, entry):
    source=child(root,entry["repository"]);hashes=json.loads(child(root,entry["manifest"]).read_text())
    actual={p.relative_to(source).as_posix() for p in [*(source/"experiments").glob("*.py"),
        *(source/"src/dream").rglob("*.py")]}
    if not hashes or set(hashes)!=actual:
        raise ValueError("Source snapshot is missing files or has additional Python modules")
    for name,digest in hashes.items():
        path=child(source,name)
        if any(p.is_symlink() for p in (source/name, *(source/name).parents) if p.is_relative_to(source)) or sha(path)!=digest:
            raise ValueError(f"Source snapshot changed: {name}")
    if not (source/"experiments/run_instruction_task.py").is_file():
        raise ValueError("Missing historical task entrypoint")
    return source


def verify_case(root, case):
    task_path=child(root,case["task"]);task=json.loads(task_path.read_text())
    if sha(task_path)!=case["task_sha256"] or sha(child(task_path.parent,task["room_map_file"]))!=case["room_map_sha256"]:
        raise ValueError("Task or room map changed")
    if (task["scene"],task["seed"],task["recipe"]["id"],task["instruction"])!=(case["scene"],case["seed"],case["recipe"],case["instruction"]):
        raise ValueError("Case identity differs from saved task")
    config=case["configuration"]
    if config.get("threshold")!=.15 or config.get("navigation_budget")!=110 or config.get("variant")!="dynamic" or config.get("video") is not True:
        raise ValueError("Profile parameters cannot be silently replaced by launcher defaults")
    return task_path


def build_command(catalog, *, case_id=None, study=False, output, gpus=None,
                  asset_dir=None, model_cache=None, python=None, launcher=None):
    root,data=load_catalog(catalog)
    if study == (case_id is not None):
        raise ValueError("Choose exactly one historical case or the frozen study")
    output=Path(output).resolve()
    if output.exists():raise FileExistsError("Use a new output path; never overwrite an earlier attempt")
    if study:
        entry=data["study"]
        by_id={c["id"]:c for c in data["cases"]}
        if any(key not in by_id for key in entry["case_order"]):
            raise ValueError("Unknown case in frozen study order")
        cases=[by_id[key] for key in entry["case_order"]]
        if len(cases)!=10 or len({c["id"] for c in cases})!=10:
            raise ValueError("Frozen study must include all ten cases exactly once")
        if entry["seeds"]!=[100,101,102] or entry["variants"]!=["dynamic","static"]:
            raise ValueError("Study seed/variant pairing differs from the declared comparison")
        source_id=entry["source_id"];heading=entry["heading_navigation"]
        if heading is not True:raise ValueError("Study requires the frozen heading adaptation")
        workers=gpus or entry["gpu_workers"]
    else:
        matches=[c for c in data["cases"] if c["id"]==case_id]
        if len(matches)!=1:raise ValueError(f"Unknown profile: {case_id}")
        cases=matches;source_id=cases[0]["source_id"]
        heading=bool(cases[0]["configuration"].get("heading_navigation",False));workers=gpus or ["0"]
    source=verify_source(root,data["sources"][source_id])
    tasks=[verify_case(root,case) for case in cases]
    command=[str(python or sys.executable),str(launcher or Path(__file__).with_name("run_instruction_frozen_batch.py")),
        "--source-repo",str(source),"--tasks",*map(str,tasks),"--output",str(output),
        "--gpus",*workers,"--wall-timeout","14400"]
    if heading:command.append("--heading-navigation")
    if study:command.extend(["--benchmark","--seeds","100","101","102","--variants","dynamic","static"])
    else:command.extend(["--variants","dynamic"])
    if asset_dir:command.extend(["--asset-dir",str(Path(asset_dir).resolve())])
    if model_cache:command.extend(["--model-cache",str(Path(model_cache).resolve())])
    report=dict(mode="frozen_paired_study_reproduction" if study else "historical_video_profile_reproduction",
        case_ids=[c["id"] for c in cases],source_id=source_id,source_files_verified=True,
        tasks_verified=True,command=command,policy_executed=False,asset_contents_verified_by_this_launcher=False,
        boundary="A validated command, not a simulation result. Historical profiles use different source snapshots; the paired study uses one. Outcomes on other hardware are not guaranteed identical. All new attempts must be retained.")
    return command,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog",type=Path,default=Path(__file__).with_name("repro_profiles")/"profiles.json")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case",help="Two-digit gallery ID, e.g. 01")
    mode.add_argument("--study",action="store_true",help="All sixty attempts with the one frozen study controller")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpus",nargs="+")
    parser.add_argument("--asset-dir",type=Path)
    parser.add_argument("--model-cache",type=Path)
    parser.add_argument("--execute",action="store_true",help="Actually run; otherwise print a validated dry-run command")
    args=parser.parse_args()
    command,report=build_command(args.catalog,case_id=args.case,study=args.study,output=args.output,
        gpus=args.gpus,asset_dir=args.asset_dir,model_cache=args.model_cache)
    print(json.dumps(report,indent=2),flush=True);print(shlex.join(command),flush=True)
    if args.execute:
        root,catalog=load_catalog(args.catalog)
        repository=Path(__file__).resolve().parents[1]
        cache=(args.model_cache or repository.parent/".dream_model_cache").resolve()
        expected=json.loads(child(root,catalog["model_lock"]["file"]).read_text())
        actual=json.loads((cache/"dream_models.lock.json").read_text())
        if actual!=expected:raise ValueError("Prepared model revisions differ from the recorded production lock")
        subprocess.run(command,cwd=repository,check=True)


if __name__=="__main__":main()
