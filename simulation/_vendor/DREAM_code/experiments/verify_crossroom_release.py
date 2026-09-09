#!/usr/bin/env python3
"""Check packaged evidence and compile/test extracted files in a new directory.

Reuses the explicitly supplied Python environment and optional asset directory;
this is archive portability verification, not a new dependency installation.
Raw evidence is streamed for hashing to avoid requiring another full-size copy.
"""
import argparse
from contextlib import ExitStack
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import fitz


def digest(stream):
    value=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b""):
        value.update(block)
    return value.hexdigest()


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths=[]

    def handle_starttag(self,tag,attrs):
        self.paths.extend(value for key,value in attrs if key in ("src","href","poster") and value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--tectonic",type=Path,required=True)
    p.add_argument("--asset-dir",type=Path,help="Optional existing official assets for one fresh physical re-execution")
    args=p.parse_args()
    args.release=args.release.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    scratch=Path(tempfile.mkdtemp(prefix="dream-crossroom-release-"))
    report=dict(scratch=str(scratch),scope="Packaged-source/data verification using the existing Python environment; not a fresh dependency install or a new learned-policy trial.")
    (args.output/"started.json").write_text(json.dumps(report,indent=2)+"\n")
    release=json.loads((args.release/"RELEASE_VERIFICATION.json").read_text())
    selection=json.loads((args.release/"selection_manifest.json").read_text())["cases"]
    assert len(selection)==len({c["scene"] for c in selection})==10
    with ExitStack() as stack:
        archives={}
        members={}
        for package in release["packages"]:
            file=args.release/package["file"]
            with file.open("rb") as stream:
                assert digest(stream)==package["sha256"],file.name
            archive=stack.enter_context(zipfile.ZipFile(file))
            archives[file.name]=archive
            for name in archive.namelist():
                assert not Path(name).is_absolute() and ".." not in Path(name).parts,name
                if file.name!="OVERLEAF_DREAM_CROSSROOM_R2.zip":
                    assert name not in members,("Duplicate evidence member",name)
                    members[name]=archive
        checked=[]
        for case in selection:
            prefix=case["source_run"]
            manifest=json.loads(members[prefix+"/sha256.json"].read(prefix+"/sha256.json"))
            for name,expected in manifest.items():
                relative="source_snapshot/"+name[7:] if name.startswith("source:") else name
                key=prefix+"/"+relative
                assert key in members,("Missing recorded evidence",key)
                with members[key].open(key) as stream:
                    assert digest(stream)==expected,("Evidence hash mismatch",key)
            checked.append(dict(scene=case["scene"],verified_files=len(manifest)))
            print(json.dumps(checked[-1]),flush=True)
        batch=str(Path(selection[0]["source_run"]).parent)
        protocol=json.loads(members[batch+"/protocol.json"].read(batch+"/protocol.json"))
        for name,expected in protocol["source_sha256"].items():
            key=batch+"/frozen_source/"+name
            with members[key].open(key) as stream:
                assert digest(stream)==expected,key
        for index,task in enumerate(protocol["tasks"],1):
            key=f"{batch}/frozen_tasks/{index:02d}/{Path(task['file']).name}"
            values=json.loads(members[key].read(key))
            assert hashlib.sha256(members[key].read(key)).hexdigest()==task["sha256"]
            room=str(Path(key).parent/values["room_map_file"])
            assert hashlib.sha256(members[room].read(room)).hexdigest()==task["room_map_sha256"]
        page=release["gallery"]+"/index.html"
        links=Links()
        links.feed(members[page].read(page).decode())
        for link in links.paths:
            if ":" not in link and not link.startswith("#"):
                normalized=os.path.normpath(str(Path(page).parent/link))
                assert normalized in members,("Broken gallery link",normalized)
        evidence=scratch/"evidence"
        code=archives["DREAM_CROSSROOM_CODE_AND_LOGS.zip"]
        for name in code.namelist():
            path=Path(name)
            if name.startswith(("DREAM_code/src/","DREAM_code/tests/","DREAM_code/experiments/configs/",batch+"/")) or path.parent==Path("DREAM_code/experiments"):
                code.extract(name,evidence)
        manuscript=scratch/"overleaf"
        archives["OVERLEAF_DREAM_CROSSROOM_R2.zip"].extractall(manuscript)
        report.update(selected_evidence=checked,frozen_sources_verified=len(protocol["source_sha256"]),
                      frozen_tasks_verified=len(protocol["tasks"]),gallery_links_verified=len(links.paths),
                      package_hashes_verified=len(archives))
    env=os.environ.copy()
    env.pop("PYTHONPATH",None)
    env["PYTHONNOUSERSITE"]="1"
    with (args.output/"extracted_source_tests.log").open("w") as stream:
        subprocess.run([sys.executable,"-m","pytest","tests","-q"],cwd=evidence/"DREAM_code",env=env,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    report["extracted_source_tests_passed"]=True
    pdfs=[]
    for name in ("DREAM","DREAM_Clean","DREAM_R2_Tracked_Changes","Response_R2"):
        with (args.output/(name+"_compile.log")).open("w") as stream:
            subprocess.run([str(args.tectonic.resolve()),"-X","compile",name+".tex","--keep-logs"],
                           cwd=manuscript,stdout=stream,stderr=subprocess.STDOUT,check=True)
        document=fitz.open(manuscript/(name+".pdf"))
        text=" ".join(" ".join(page.get_text() for page in document).split())
        assert "10 curated successful" in text and "Working draft" not in text
        assert name=="Response_R2" or len(document)<=11
        pdfs.append(dict(file=name+".pdf",pages=len(document)))
    report["independently_compiled_pdfs"]=pdfs
    if args.asset_dir:
        env["MS_ASSET_DIR"]=str(args.asset_dir.resolve())
        run=evidence/selection[-1]["source_run"]
        output=args.output.resolve()/"extracted_source_physical_reexecution.json"
        with (args.output/"extracted_source_physical_reexecution.log").open("w") as stream:
            subprocess.run([sys.executable,"experiments/replay_learned_actions.py","--source-run",str(run),
                            "--output",str(output),"--contact-audit"],cwd=evidence/"DREAM_code",env=env,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        physical=json.loads(output.read_text())
        assert physical["passed"] and physical["contact_audit"]["native_environment_contact_control_steps"]==0
        report["extracted_source_physical_reexecution"]=dict(scene=selection[-1]["scene"],
            maximum_error=physical["maximum_error"],steps=physical["steps"],passed=True,
            dependency_boundary="Reused installed Python packages and explicitly supplied official assets.")
    report["passed"]=True
    (args.output/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
