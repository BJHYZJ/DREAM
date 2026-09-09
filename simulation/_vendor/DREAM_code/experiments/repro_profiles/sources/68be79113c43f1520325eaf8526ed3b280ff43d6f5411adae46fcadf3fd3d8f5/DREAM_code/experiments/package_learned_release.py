#!/usr/bin/env python3
"""Package audited learned evidence; never publish a still-running suite."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import fitz

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/"DREAM_code"


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def add(zipper,path,name=None):
    name=str(name or path.relative_to(ROOT))
    if path.suffix in (".py",".yaml",".yml",".json",".md",".txt"):
        if re.search(rb"(?:sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,})",path.read_bytes()):
            raise RuntimeError(f"Possible embedded credential; review before release: {path.name}")
    compression=zipfile.ZIP_STORED if path.suffix in (".mp4",".npz",".png",".jpg",".pdf") else zipfile.ZIP_DEFLATED
    zipper.write(path,name,compress_type=compression)


def screenshots(manuscript,output):
    rows=[]
    for name in ("DREAM.pdf","DREAM_Clean.pdf","DREAM_R2_Tracked_Changes.pdf","Response_R2.pdf"):
        document=fitz.open(manuscript/name)
        if name!="Response_R2.pdf" and len(document)>11:
            raise RuntimeError(f"Manuscript exceeds the decision's 11-page limit: {name}")
        for page in document:
            content=page.get_text().lower()
            if any(term in content for term in ("learned-perception simulation", "all ten preselected", "initial scan/relocation", "actual observations from one successful")):
                target=output/f"{Path(name).stem}_page_{page.number+1:02d}.png"
                page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(target)
        rows.append(dict(file=name,pages=len(document),sha256=digest(manuscript/name)))
    (output/"pdf_verification.json").write_text(json.dumps(rows,indent=2)+"\n")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--gallery",type=Path,required=True)
    parser.add_argument("--manuscript",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--clean-python",type=Path)
    args=parser.parse_args()
    for field in ("suite","gallery","manuscript","output"):
        setattr(args,field,getattr(args,field).resolve())
    suite=json.loads((args.suite/"suite_result.json").read_text())
    audit=json.loads((args.gallery/"audit_and_results.json").read_text())
    if len(suite["attempts"])!=20 or len(audit["runs"])!=20:
        raise RuntimeError("Expected all twenty scheduled attempts")
    for row in audit["runs"]:
        if row["hash_mismatches"] or not row["hash_manifest_present"] or not row["trajectory_aligned"] or not row["action_steps_contiguous"]:
            raise RuntimeError(f"Evidence audit failed: {row['name']}")
        if row["variant"]=="dynamic" and not row["video"]["continuous"]:
            raise RuntimeError("Missing/noncontinuous dynamic recording")
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(ROOT/"clips/LEARNED_RELEASE_README.md",args.output/"README.md")
    shutil.copy2(args.gallery/"learned_scene_contact_sheet.png",args.output/"TEN_SCENES.png")
    shutil.copy2(ROOT/"clips/learned_integration_development/illustrated_val02_v2/recorded_observation_sequence.png",
        args.output/"OBSERVATION_SEQUENCE.png")
    pdfs=screenshots(args.manuscript,args.output)
    if args.clean_python:
        dependencies=subprocess.run([str(args.clean_python),"-m","pip","freeze"],check=True,capture_output=True,text=True)
        (args.output/"clean_install_dependency_inventory.txt").write_text(dependencies.stdout)
    (args.output/"released_worktree.patch").write_text(subprocess.run(
        ["git","diff","--","src/dream","README.md"],cwd=REPO,check=True,capture_output=True,text=True).stdout)

    # Main/clean/red/response share one source. Exclude obsolete latexdiff
    # versions, sample IEEE papers, auxiliary files and historical simulator figs.
    tex_names=("DREAM.tex","DREAM_Clean.tex","DREAM_R2_Tracked_Changes.tex","Response_R2.tex",
        "learned_maniskill_section.tex","learned_maniskill_response.tex","learned_suite_summary.tex","learned_suite_rows.tex")
    text="\n".join((args.manuscript/name).read_text() for name in tex_names)
    images=set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",text))
    with zipfile.ZipFile(args.output/"OVERLEAF_DREAM_LEARNED_R2.zip","x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipper:
        for name in (*tex_names,"reference.bib","IEEEtran.cls","READ_BEFORE_SUBMISSION.md",*(r["file"] for r in pdfs),*sorted(images)):
            add(zipper,args.manuscript/name,name)

    suites=[REPO/"experiments/results/learned_ten_scene_suite_v1",args.suite]
    component=REPO/"experiments/results/learned_matched_replay_v1"
    component_source=Path(json.loads((component/"result.json").read_text())["source_run"])
    code_files=set()
    for directory in (REPO/"src/dream",REPO/"tests",REPO/"experiments/configs"):
        code_files.update(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts and
            (p.suffix in (".py",".json",".yaml",".yml",".md",".txt",".png") or p.name=="LICENSE"))
    code_files.update(p for p in (REPO/"experiments").iterdir() if p.is_file() and p.suffix in (".py",".md",".txt"))
    code_files.update((REPO/"README.md",REPO/"LICENSE",REPO/"experiments/results/REPORT.md",ROOT/"clips/README.md",ROOT/".dream_model_cache/dream_models.lock.json",
        ROOT/".maniskill_assets/data/scene_datasets/ai2thor/ARCHITECTHOR_SUBSET_MANIFEST.json"))
    for directory in (REPO/"experiments/results").iterdir():
        if directory.is_dir() and (directory.name.startswith("learned_") or directory.name in (
            "rmp","rmp_scaling","rmp_sensitivity","rmp_timing","combined_sensitivity","exploration","exploration_sensitivity","houseexpo_cross_room","video_timeline")):
            code_files.update(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts and
                p.suffix in (".py",".json",".jsonl",".log",".csv",".md",".pdf",".txt"))
    code_files.update(p for p in component.iterdir() if p.is_file())
    for gallery in (ROOT/"clips/learned_maniskill_suite_v1",args.gallery):
        code_files.update(p for p in gallery.iterdir() if p.is_file() and p.suffix!=".mp4")
    development=ROOT/"clips/learned_integration_development"
    code_files.update(p for p in development.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix in (".json",".log",".md",".pdf"))
    code_files.update(p for p in args.output.iterdir() if p.is_file() and p.suffix!=".zip")
    with zipfile.ZipFile(args.output/"DREAM_LEARNED_CODE_AND_LOGS.zip","x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zipper:
        for path in sorted(code_files):
            add(zipper,path)

    # Keep all archives on the same workspace-relative tree. Extract together
    # for a full source/hash audit; the compact code archive alone is runnable.
    gallery_text={p:p.read_text() for p in args.gallery.iterdir() if p.suffix in (".html",".md")}
    videos=sorted(p for p in args.gallery.iterdir() if p.suffix==".mp4")
    if len(videos)!=10:
        raise RuntimeError("The primary video supplement must contain ten videos")
    for video in videos:
        relative="../../"+str(video.resolve().relative_to(ROOT))
        gallery_text={p:value.replace(video.name,relative) for p,value in gallery_text.items()}
    with zipfile.ZipFile(args.output/"DREAM_LEARNED_10_VIDEOS.zip","x",compression=zipfile.ZIP_DEFLATED) as zipper:
        for video in videos:
            add(zipper,video.resolve())
        for path in args.gallery.iterdir():
            if path.suffix==".mp4":
                continue
            if path in gallery_text:
                zipper.writestr(str(path.relative_to(ROOT)),gallery_text[path])
            else:
                add(zipper,path)
        zipper.writestr("OPEN_VIDEOS.html",f'<!doctype html><meta http-equiv="refresh" content="0;url={args.gallery.relative_to(ROOT)}/index.html"><a href="{args.gallery.relative_to(ROOT)}/index.html">Open ten-scene gallery</a>')

    raw=set()
    for directory in (*suites,component_source):
        raw.update(p for p in directory.rglob("*") if p.is_file() and p.suffix in (".npz",".mp4",".png"))
    for directory in {video.resolve().parent for video in videos}:
        raw.update(p for p in directory.iterdir() if p.is_file() and p.suffix in (".mp4",".png"))
    raw.difference_update(video.resolve() for video in videos)
    with zipfile.ZipFile(args.output/"DREAM_LEARNED_RAW_EVIDENCE_OPTIONAL.zip","x",compression=zipfile.ZIP_STORED) as zipper:
        for path in sorted(raw):
            add(zipper,path)

    packages=[]
    for path in sorted(args.output.glob("*.zip")):
        with zipfile.ZipFile(path) as zipper:
            corrupt=zipper.testzip()
            if corrupt:
                raise RuntimeError(f"Corrupt package member: {corrupt}")
            members=len(zipper.infolist())
        packages.append(dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path),members=members,crc_verified=True))
    report=dict(suite=str(args.suite.relative_to(ROOT)),counts=audit["counts"],pdfs=pdfs,packages=packages,
        evidence_scope="Two separate development suites, never pooled; all v2 dynamic videos included whether successful or not. Compact code/log package omits bulky raw observations/videos; extract all evidence ZIPs together for the full hash audit. No assets or model weights redistributed.")
    (args.output/"RELEASE_VERIFICATION.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
