#!/usr/bin/env python3
"""Download pinned native houses and instruction-recipe assets, with hashes.

Downloads original upstream ZIP members; does not redistribute third-party
meshes. Existing conflicting files are never overwritten. A reference lock
supports exact verification/reuse, including on a later clean installation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path,PurePosixPath
import tempfile
import threading

from remotezip import RemoteZip
from extract_architecthor_subset import _safe_member,_sha256

REVISION="1a173d5de042aaad8f1af09d4d2bc2ce4004b28a"


def resolve(remote,tasks,recipes):
    available=set(remote.namelist())
    members={"ai2thor/ai2thor-hab/configs/object_semantic_id_mapping.json"}
    scenes=[]
    for scene in sorted({task["scene"] for task in tasks}):
        matches=[name for name in available if name.startswith("ai2thor/ai2thor-hab/configs/scenes/")
                 and PurePosixPath(name).name==scene+".scene_instance.json"]
        if len(matches)!=1:raise ValueError(f"Expected exactly one official scene: {scene}")
        name=matches[0];config=json.loads(remote.read(name));members.add(name)
        members.add("ai2thor/ai2thor-hab/assets/"+config["stage_instance"]["template_name"]+".glb")
        for template in {obj["template_name"] for obj in config["object_instances"]}:
            members.add(f"ai2thor/ai2thorhab-uncompressed/assets/{template}.glb")
            members.add(f"ai2thor/ai2thorhab-uncompressed/configs/{template}.object_config.json")
        scenes.append(dict(scene=scene,scene_config=name))
    assets=set()
    for recipe in [*(t["recipe"] for t in tasks),*recipes]:assets.update(recipe["environment_assets"].values())
    assets.update(t["placement_table_asset"] for t in tasks)
    for asset in assets:
        if "/" in asset or ".." in asset:raise ValueError("Invalid household asset name")
        members.add(f"ai2thor/ai2thorhab-uncompressed/assets/objects/{asset}.glb")
        members.add(f"ai2thor/ai2thorhab-uncompressed/configs/objects/{asset}.object_config.json")
    missing=members-available
    if missing:raise FileNotFoundError(f"Missing official members: {sorted(missing)}")
    return sorted(members),scenes,sorted(assets)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks",type=Path,nargs="+")
    p.add_argument("--recipes-json",type=Path)
    inputs=p.add_mutually_exclusive_group()
    inputs.add_argument("--reference-lock",type=Path)
    inputs.add_argument("--resume-manifest",type=Path,help="Resume a partial download, preserving its completed member hashes")
    p.add_argument("--output-parent",type=Path,required=True)
    p.add_argument("--output-manifest",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--verify-only",action="store_true",help="No network/download; requires a reference lock")
    args=p.parse_args()
    if args.output_manifest.exists():raise FileExistsError(args.output_manifest)
    if args.workers<1:raise ValueError("Positive workers required")
    reference_path=args.reference_lock or args.resume_manifest
    reference=json.loads(reference_path.read_text()) if reference_path else None
    if args.reference_lock and not reference.get("complete",False):
        raise ValueError("An incomplete download is not a valid lock; use --resume-manifest to finish it")
    if args.verify_only and reference is None:raise ValueError("Verification requires a reference lock")
    revision=reference["source_revision"] if reference else REVISION
    if revision!=REVISION:raise ValueError("Unsupported asset revision; do not silently migrate the experiment")
    url=f"https://huggingface.co/datasets/haosulab/AI2THOR/resolve/{revision}/ai2thor.zip"
    if reference:
        members=sorted(set(reference["artifacts"])|{r["member"] for r in reference.get("failures",[])})
        scenes=reference["scenes"];assets=reference["recipe_assets"]
    else:
        if not args.tasks:raise ValueError("Task configs are required when creating a new lock")
        tasks=[json.loads(path.read_text()) for path in args.tasks]
        recipes=json.loads(args.recipes_json.read_text())["recipes"] if args.recipes_json else []
        with RemoteZip(url) as remote:members,scenes,assets=resolve(remote,tasks,recipes)
    for name in members:_safe_member(name)
    local=threading.local();opened=[];lock=threading.Lock()

    def one(name):
        target=args.output_parent/name
        if target.exists():
            checksum=_sha256(target)
            if not reference:
                raise FileExistsError("Creating a lock requires a fresh destination; use --reference-lock to reuse files")
            expected=reference["artifacts"].get(name)
            if expected is None:raise ValueError(f"Existing file has no completed reference hash: {target}")
            if checksum!=expected["sha256"] or target.stat().st_size!=expected["bytes"]:
                raise ValueError(f"Existing asset mismatch; refusing overwrite: {target}")
            return name,dict(bytes=target.stat().st_size,sha256=checksum,reused_existing=True)
        if args.verify_only:raise FileNotFoundError(target)
        if not hasattr(local,"remote"):
            local.remote=RemoteZip(url)
            with lock:opened.append(local.remote)
        payload=local.remote.read(name);checksum=hashlib.sha256(payload).hexdigest()
        if reference and name in reference["artifacts"]:
            expected=reference["artifacts"][name]
            if checksum!=expected["sha256"] or len(payload)!=expected["bytes"]:
                raise ValueError(f"Downloaded member differs from reference: {name}")
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent,prefix=target.name+".",suffix=".part",delete=False) as stream:
            stream.write(payload);partial=Path(stream.name)
        # Exclusive target creation avoids overwriting another worker/process.
        target.hardlink_to(partial)
        partial.unlink()
        return name,dict(bytes=len(payload),sha256=checksum,reused_existing=False)

    artifacts={};errors=[]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(one,name):name for name in members}
            for future in as_completed(futures):
                try:
                    name,row=future.result();artifacts[name]=row
                except Exception as error:errors.append(dict(member=futures[future],error=repr(error)))
                if (len(artifacts)+len(errors))%20==0:
                    print(json.dumps(dict(completed=len(artifacts),failed=len(errors),total=len(members))),flush=True)
    finally:
        for remote in opened:remote.close()
    report=dict(source_revision=revision,source_url=url,scenes=scenes,recipe_assets=assets,
        reference_lock_sha256=_sha256(args.reference_lock) if args.reference_lock else None,
        resumed_manifest_sha256=_sha256(args.resume_manifest) if args.resume_manifest else None,
        verify_only=args.verify_only,complete=not errors and len(artifacts)==len(members),
        artifacts=artifacts,failures=errors,third_party_assets_redistributed=False)
    args.output_manifest.parent.mkdir(parents=True,exist_ok=True)
    args.output_manifest.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(complete=report["complete"],files=len(artifacts),failures=len(errors),manifest=str(args.output_manifest))),flush=True)
    if not report["complete"]:raise SystemExit(1)


if __name__=="__main__":main()
