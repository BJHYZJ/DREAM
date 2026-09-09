#!/usr/bin/env python3
"""Export exact historical video profiles and one frozen-study source snapshot.

No simulator task is run. Originals, failed scores and prior audit bytes stay
unchanged. The export is a local author preview, not a completed-study release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from run_instruction_profile import child, sha, load_catalog, verify_source, verify_case


def source_id(hashes):
    return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def copy_checked(source,destination,expected=None):
    content=source.read_bytes()
    if source.suffix in (".py",".json",".md",".txt") and re.search(rb"(?:sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,})",content):
        raise ValueError(f"Possible credential in {source.name}; do not export before reviewing it")
    digest=hashlib.sha256(content).hexdigest()
    if expected is not None and digest!=expected:raise ValueError(f"Source changed: {source}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("xb") as stream:stream.write(content)
    if sha(destination)!=digest:raise ValueError(f"Copy verification failed: {destination}")
    return digest


def export(workspace,gallery,study,output):
    workspace=Path(workspace).resolve();gallery=Path(gallery).resolve();study=Path(study).resolve();output=Path(output).resolve()
    selected=json.loads((gallery/"manifest.json").read_text())
    if selected.get("selection")!="development_selected_successful_examples":raise ValueError("Wrong gallery protocol")
    cases=selected["cases"]
    if len(cases)!=10 or len({c["scene"] for c in cases})!=10 or len({c["recipe"] for c in cases})!=5:
        raise ValueError("Expected ten checked distinct-house cases and five recipes")
    protocol=json.loads((study/"protocol.json").read_text())
    if protocol.get("mode")!="prespecified_60_attempt_paired_study" or protocol.get("planned_attempts")!=60:
        raise ValueError("Missing declared paired-study protocol")
    if sha(study/"protocol.json")!=(study/"protocol.sha256").read_text().strip():raise ValueError("Changed study protocol")
    if (len(protocol["attempts"])!=60 or protocol["seeds"]!=[100,101,102]
            or protocol["variants"]!=["dynamic","static"] or protocol["heading_navigation"] is not True
            or protocol["threshold"]!=.15 or protocol["navigation_budget"]!=110):
        raise ValueError("Study parameters differ from the declared comparison")
    model_lock=workspace/".dream_model_cache/dream_models.lock.json"
    if sha(model_lock)!=protocol["model_lock_sha256"]:
        raise ValueError("Current model lock differs from the frozen study")
    asset_lock=workspace/"clips/instruction_pickplace_development/instruction_asset_lock03.json"
    assets=json.loads(asset_lock.read_text())
    if not assets.get("complete") or assets.get("failures") or not assets.get("artifacts"):
        raise ValueError("Asset reference lock is incomplete")
    if not {c["scene"] for c in cases}<={s["scene"] for s in assets["scenes"]}:
        raise ValueError("Asset reference lock does not cover the selected houses")
    output.mkdir(parents=True,exist_ok=False);sources={};profiles=[]

    def add_source(repository,hashes):
        identifier=source_id(hashes)
        if identifier not in sources:
            destination=output/"sources"/identifier/"DREAM_code"
            for name,digest in sorted(hashes.items()):
                target=child(destination,name)
                copy_checked(child(repository,name),target,digest);target.chmod(0o444)
            manifest=destination.parent/"source_hashes.json"
            manifest.write_text(json.dumps(hashes,sort_keys=True,indent=2)+"\n")
            sources[identifier]=dict(repository=destination.relative_to(output).as_posix(),
                manifest=manifest.relative_to(output).as_posix(),manifest_sha256=sha(manifest),source_files=len(hashes))
        return identifier

    for case in cases:
        key=f"{case['index']:02d}";run=child(workspace,case["source_run"])
        before=json.loads((run/"source_hashes_before.json").read_text())
        if before!=json.loads((run/"source_hashes_after.json").read_text()):raise ValueError("Original source changed during task")
        result=json.loads((run/"result.json").read_text())
        if result.get("source_files_unchanged") is not True:raise ValueError("Unverified original source")
        config=json.loads((run/"configuration.json").read_text());task=json.loads((run/"environment_task.json").read_text())
        if (task["scene"],task["seed"],task["instruction"])!=(case["scene"],case["seed"],case["instruction"]):raise ValueError("Gallery/task identity mismatch")
        evidence={Path(ref).name:child(gallery,ref) for ref in case["evidence"]}
        physical=json.loads(evidence["audit.json"].read_text());record=json.loads(evidence["record_review.json"].read_text())
        if not (physical["physical_reexecution_passed"] is True and physical["evaluation"]["evaluator_protocol_success"] is True
                and record["record_review_passed"] is True and physical["contact_audit"]["native_environment_contact_control_steps"]==0):
            raise ValueError("Selected evidence did not pass")
        if Path(physical["source_run"]).resolve()!=run or Path(record["source_run"]).resolve()!=run:
            raise ValueError("Selected audits refer to another run")
        if sha(child(gallery,case["video"]))!=case["video_sha256"]:raise ValueError("Selected video changed")
        identifier=add_source(run/"source_snapshot",before)
        folder=output/"tasks"/f"{key}_{case['scene']}"
        task_sha=copy_checked(run/"environment_task.json",folder/"task.json")
        room_sha=copy_checked(child(run,task["room_map_file"]),child(folder,task["room_map_file"]))
        record_folder=output/"records"/key;refs={}
        for filename in ("result.json","configuration.json","initial_conditions.json","source_hashes_before.json","source_hashes_after.json"):
            target=record_folder/filename;copy_checked(run/filename,target);refs[filename]=target.relative_to(output).as_posix()
        for filename,path in evidence.items():
            target=record_folder/filename;copy_checked(path,target);refs[filename]=target.relative_to(output).as_posix()
        profiles.append(dict(id=key,scene=case["scene"],seed=case["seed"],recipe=case["recipe"],instruction=case["instruction"],
            source_id=identifier,task=(folder/"task.json").relative_to(output).as_posix(),task_sha256=task_sha,
            room_map_sha256=room_sha,configuration=config,original_source_run=case["source_run"],
            original_evaluation_version=result.get("evaluation_definition_version",1),
            original_strict_success=result["evaluator_protocol_success"],
            reviewed_evaluation_version=physical["evaluation"].get("evaluation_definition_version",1),
            reviewed_strict_success=True,scoring_reassessment=case["scoring_reassessment"],
            same_controls_spectator_rerender=case["same_controls_spectator_rerender"],
            video_sha256=case["video_sha256"],video_filename=Path(case["video"]).name,records=refs))
    common_id=add_source(study/"frozen_workspace/DREAM_code",protocol["source_sha256"])
    common_order=[]
    for job in protocol["attempts"]:
        key=next((c["id"] for c in profiles if c["scene"]==job["scene"]),None)
        if key is None:raise ValueError("Study has a house outside the gallery configuration set")
        original_path=child(study,job["task"])
        original=json.loads(original_path.read_text())
        if (sha(original_path)!=job["task_sha256"]
                or sha(child(original_path.parent,original["room_map_file"]))!=job["room_map_sha256"]
                or (original["scene"],original["seed"],original["recipe"]["id"])!=(job["scene"],job["seed"],job["recipe"])):
            raise ValueError("Frozen study task input changed")
        profile=next(c for c in profiles if c["id"]==key)
        exported=json.loads(child(output,profile["task"]).read_text())
        fixed=lambda t:{k:v for k,v in t.items() if k not in ("seed","initial_sample_preview")}
        if fixed(original)!=fixed(exported) or job["room_map_sha256"]!=profile["room_map_sha256"]:
            raise ValueError("Study fixture differs from profile; do not silently substitute it")
        if key not in common_order:common_order.append(key)
    locks={}
    for key,path in (("model_lock",model_lock),("asset_lock",asset_lock)):
        target=output/"locks"/path.name;copy_checked(path,target)
        locks[key]=dict(file=target.relative_to(output).as_posix(),sha256=sha(target))
    copy_checked(study/"protocol.json",output/"study/recorded_protocol.json")
    catalog=dict(schema_version=1,scope="versioned_instruction_reproduction_profiles",cases=profiles,sources=sources,**locks,
        study=dict(source_id=common_id,case_order=common_order,seeds=protocol["seeds"],variants=protocol["variants"],
            gpu_workers=protocol["gpu_workers"],heading_navigation=protocol["heading_navigation"],
            recorded_protocol="study/recorded_protocol.json",recorded_protocol_sha256=sha(study/"protocol.json"),
            result_status="not_included_in_this_profile_export"),
        boundary="Reproduction configurations and exact source snapshots, not new tasks or a completed result release. Original failures/reassessments and spectator-only re-renders are explicit. Third-party assets and model weights are downloaded separately.",
        release_ready=False)
    (output/"profiles.json").write_text(json.dumps(catalog,indent=2)+"\n")
    root,checked=load_catalog(output/"profiles.json")
    for entry in checked["sources"].values():verify_source(root,entry)
    for case in checked["cases"]:verify_case(root,case)
    report=dict(profiles=10,distinct_houses=10,recipes=5,unique_source_snapshots=len(sources),
        exported_source_files=sum(s["source_files"] for s in sources.values()),all_profile_checks_passed=True,
        policy_executed=False,release_ready=False,catalog_sha256=sha(output/"profiles.json"))
    (output/"export_check.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--gallery",type=Path,required=True)
    parser.add_argument("--study",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();print(json.dumps(export(args.workspace,args.gallery,args.study,args.output),indent=2),flush=True)


if __name__=="__main__":main()
