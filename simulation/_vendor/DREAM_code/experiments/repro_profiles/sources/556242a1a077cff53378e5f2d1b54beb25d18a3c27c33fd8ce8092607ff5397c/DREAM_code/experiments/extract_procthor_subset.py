#!/usr/bin/env python3
"""Download selected official ProcTHOR configs/stages, then their assets.

The scene list is recorded before any learned attempt. Candidate suitability
is assessed from geometry only; success-only footage is not a benchmark split.
The immutable revision is pinned. A reference manifest verifies exact SHA256
content, including reused files. Conflicting existing assets are not overwritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading

from remotezip import RemoteZip
from extract_architecthor_subset import _safe_member,_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--configs-only",action="store_true")
    parser.add_argument("--count",type=int,default=24)
    parser.add_argument("--selected-manifest",type=Path)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--revision",default="1a173d5de042aaad8f1af09d4d2bc2ce4004b28a")
    parser.add_argument("--reference-manifest",type=Path)
    args=parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    reference=json.loads(args.reference_manifest.read_text()) if args.reference_manifest else None
    revision=reference["source_revision"] if reference else args.revision
    url=f"https://huggingface.co/datasets/haosulab/AI2THOR/resolve/{revision}/ai2thor.zip"
    if args.selected_manifest:
        selections=json.loads(args.selected_manifest.read_text())["selected_configs"]
    else:
        import importlib.util
        package=Path(importlib.util.find_spec("mani_skill").origin).parent
        metadata=json.loads((package/"utils/scene_builder/ai2thor/metadata/ProcTHOR.json").read_text())["scenes"]
        selections=[metadata[i] for i in range(0,len(metadata),max(1,len(metadata)//args.count))][:args.count]
    members={"ai2thor/ai2thor-hab/configs/object_semantic_id_mapping.json"}
    with RemoteZip(url) as remote:
        available=set(remote.namelist())
        configs=[]
        for selection in selections:
            name="ai2thor/ai2thor-hab/configs/scenes/ProcTHOR/"+str(Path(selection))
            config=json.loads(remote.read(name))
            members.add(name)
            members.add("ai2thor/ai2thor-hab/assets/"+config["stage_instance"]["template_name"]+".glb")
            if not args.configs_only:
                templates={obj["template_name"] for obj in config["object_instances"]}|{"objects/Apple_1"}
                for template in templates:
                    members.add(f"ai2thor/ai2thorhab-uncompressed/assets/{template}.glb")
                    members.add(f"ai2thor/ai2thorhab-uncompressed/configs/{template}.object_config.json")
            configs.append(dict(config_file=selection,scene=Path(selection).name.split(".")[0],
                                stage=config["stage_instance"]["template_name"],objects=len(config["object_instances"])))
        if members-available:
            raise RuntimeError(f"Unresolved official members: {sorted(members-available)[:10]}")
        sizes={name:remote.getinfo(name).file_size for name in members}
        print(json.dumps(dict(configs=configs,files=len(members),bytes=sum(sizes.values()),revision=revision)),flush=True)
    local=threading.local()
    opened=[]
    lock=threading.Lock()
    def download(name):
        _safe_member(name)
        target=args.output_parent/name
        if target.exists() and target.stat().st_size==sizes[name]:
            checksum=_sha256(target)
            if reference and checksum!=reference["artifacts"][name]["sha256"]:
                raise RuntimeError(f"Existing asset hash differs; refusing overwrite: {target}")
            return name,dict(bytes=sizes[name],sha256=checksum,reused_existing=True)
        if target.exists():
            raise RuntimeError(f"Existing asset differs; refusing overwrite: {target}")
        if not hasattr(local,"remote"):
            local.remote=RemoteZip(url)
            with lock:
                opened.append(local.remote)
        payload=local.remote.read(name)
        if reference:
            import hashlib
            if hashlib.sha256(payload).hexdigest()!=reference["artifacts"][name]["sha256"]:
                raise RuntimeError(f"Downloaded asset differs from recorded reference: {name}")
        target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_suffix(target.suffix+".download-part")
        partial.write_bytes(payload)
        partial.replace(target)
        return name,dict(bytes=len(payload),sha256=_sha256(target),reused_existing=False)
    artifacts={}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(download,name) for name in sorted(members)]):
                name,record=future.result()
                artifacts[name]=record
                print(json.dumps(dict(done=len(artifacts),total=len(members),member=name)),flush=True)
    finally:
        for remote in opened:
            remote.close()
    result=dict(source_url=url,source_revision=revision,configs_only=args.configs_only,
                selected_configs=selections,configs=configs,artifacts=artifacts)
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
