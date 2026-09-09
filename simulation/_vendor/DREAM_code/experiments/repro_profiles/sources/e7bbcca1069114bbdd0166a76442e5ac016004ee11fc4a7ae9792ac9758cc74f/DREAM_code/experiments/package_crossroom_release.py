#!/usr/bin/env python3
"""Package ten audited distinct-house successes without hiding development.

All writes are new local archives; this does not submit/upload/push anything.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import fitz
from package_learned_release import add,digest

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/"DREAM_code"


def required_tex_files(manuscript):
    files=set()
    def visit(name):
        path=manuscript/name
        if path in files:
            return
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)
        content=path.read_text()
        for child in re.findall(r"\\input\{([^}]+)\}",content):
            visit(child if Path(child).suffix else child+".tex")
        for child in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",content):
            image=manuscript/child
            if not image.is_file():
                image=next((manuscript/(child+suffix) for suffix in (".pdf",".png",".jpg")
                            if (manuscript/(child+suffix)).is_file()),None)
            if image is None:
                raise FileNotFoundError(child)
            files.add(image)
    for name in ("DREAM.tex","DREAM_Clean.tex","DREAM_R2_Tracked_Changes.tex","Response_R2.tex"):
        visit(name)
    files.update(manuscript/name for name in ("IEEEtran.cls","reference.bib","READ_BEFORE_SUBMISSION.md",
                                             "crossroom_figure_provenance.json","crossroom_video_review.json"))
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery",type=Path,required=True)
    p.add_argument("--visual-review",type=Path,required=True)
    p.add_argument("--manuscript",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    for key in ("gallery","visual_review","manuscript","output"):
        setattr(args,key,getattr(args,key).resolve())
    gallery=json.loads((args.gallery/"audited_successes.json").read_text())
    cases=gallery["cases"]
    if gallery["partial_preview"] or len(cases)!=10 or len({r["scene"] for r in cases})!=10:
        raise ValueError("Final release needs ten distinct audited houses, not a partial preview")
    visual=json.loads((args.visual_review/"video_review_measurements.json").read_text())
    by_source={r["source_run"]:r for r in visual["cases"]}
    for case in cases:
        if not all(case.get("independent_physical_checks",{}).values()) or not case.get("independent_physical_checks"):
            raise ValueError("Independent physical checks missing/failed")
        if not case.get("batch_provenance",{}).get("passed",False):
            raise ValueError("Checked pre/post frozen-source and task provenance missing/failed")
        check=by_source[case["source_run"]]
        if not check["full_video_decode_passed"] or check["measured_bilateral_carry_base_distance_m"]<.5:
            raise ValueError("Video decode / physical base-carry check failed")
    args.output.mkdir(parents=True,exist_ok=False)
    pdfs=[]
    for name in ("DREAM.pdf","DREAM_Clean.pdf","DREAM_R2_Tracked_Changes.pdf","Response_R2.pdf"):
        doc=fitz.open(args.manuscript/name)
        text="\n".join(page.get_text() for page in doc)
        if name!="Response_R2.pdf" and len(doc)>11:
            raise ValueError(f"Manuscript exceeds eleven pages: {name}")
        if "Working draft" in text or "requested ten-house" in text:
            raise ValueError(f"Partial-draft warning still rendered: {name}")
        if not re.search(r"\b10\s+curated successful", " ".join(text.split())):
            raise ValueError(f"PDF does not contain the current ten-case result: {name}")
        for page in doc:
            if any(term in page.get_text() for term in ("Cross-room learned", "Curated successful cross-room", "Recorded search and carry", "Recorded robot observations")):
                page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(args.output/f"{Path(name).stem}_page_{page.number+1:02d}.png")
        pdfs.append(dict(file=name,pages=len(doc),sha256=digest(args.manuscript/name)))
        shutil.copy2(args.manuscript/name,args.output/name)
    (args.output/"pdf_verification.json").write_text(json.dumps(pdfs,indent=2)+"\n")
    shutil.copy2(args.gallery/"audited_successes.json",args.output/"audited_successes.json")
    shutil.copy2(args.gallery/"development_attempt_ledger.json",args.output/"development_attempt_ledger.json")
    shutil.copy2(args.visual_review/"video_review_measurements.json",args.output/"video_review_measurements.json")
    tasks=args.output/"selected_tasks"
    tasks.mkdir()
    selection=[]
    for index,case in enumerate(cases,1):
        run=Path(case["source_run"])
        task=json.loads((run/"environment_task.json").read_text())
        name=f"{index:02d}_{case['scene']}.json"
        shutil.copy2(run/"environment_task.json",tasks/name)
        shutil.copy2(run/"evaluator_room_map.npz",tasks/task["room_map_file"])
        selection.append(dict(scene=case["scene"],task=f"selected_tasks/{name}",task_sha256=digest(tasks/name),
                              source_run=str(run.relative_to(ROOT))))
    (args.output/"selection_manifest.json").write_text(json.dumps(dict(
        boundary="Curated after development; rerunning these configurations is not a held-out benchmark.",cases=selection),indent=2)+"\n")
    inventory=subprocess.run([sys.executable,"-m","pip","freeze"],check=True,capture_output=True,text=True).stdout
    (args.output/"tested_dependency_inventory.txt").write_text(inventory)
    (args.output/"README.md").write_text(f'''# DREAM 跨房间：10 条精选成功视频及作者审阅包

[观看十条完整视频](../{args.gallery.name}/index.html)。每条来自不同房屋，均通过动态任务、跨房间、物理抓取/返回放置、独立控制重执行与接触检查。
它们是开发后精选成功案例，**不是无偏的 100% 成功率，不证明优于其他方法**。失败和排除记录未删除。

## Overleaf 上传哪个文件

只把 `OVERLEAF_DREAM_CROSSROOM_R2.zip` 导入一个新 Overleaf 项目。使用 pdfLaTeX / BibTeX：

- `DREAM.tex` 或 `DREAM_R2_Tracked_Changes.tex`：本轮新增内容为红字；
- `DREAM_Clean.tex`：同一份正文的无修订颜色版本；
- `Response_R2.tex`：单独的逐条回复。

正文（含简介）已检查不超过 11 页。视频和代码 ZIP 不要上传进 Overleaf。
旧的 `OVERLEAF_DREAM_LEARNED_R2.zip` 属于历史 V2 包，不是这次的跨房间十成功展示。

## 复现和审计

将 `DREAM_CROSSROOM_CODE_AND_LOGS.zip`、`DREAM_CROSSROOM_10_VIDEOS.zip` 及需要时的
`DREAM_CROSSROOM_SELECTED_RAW.zip` 解压到同一目录。视频 ZIP 的 `OPEN_CROSSROOM_VIDEOS.html` 是观看入口。
仅观看不需要 raw ZIP；完整逐文件哈希审计和 RGB-D 重分析需要三个证据 ZIP 全部解压。
代码、选定任务、房间评估图、原始控制/事件、全部开发尝试和逐条审计均保留原目录结构。
按 `DREAM_code/experiments/README_CROSSROOM_MANISKILL.md` 安装依赖、下载官方场景与固定模型版本；模型/场景资产不在 ZIP 中。

从解压后的 `DREAM_code` 运行：

```bash
../.venv-maniskill-learned/bin/python experiments/run_crossroom_frozen_batch.py \\
  --tasks ../{args.output.relative_to(ROOT)}/selected_tasks/[0-9][0-9]_*.json \\
  --output experiments/results/my_crossroom_rerun --gpus 0 --timeout 7200
```

输出目录必须是新目录。保存控制的重执行不需要重新推理，也不是额外学习策略试验；
新学习策略执行则需下载模型并运行完整管线。不能保证跨硬件/驱动的逐位一致。
完整视频按 1 倍仿真时间连续记录，推理墙钟时间另计；章节按钮只跳转原视频。

## 证据边界与作者检查

使用真实 DREAM 特征/检测/体素记忆/动态清除/检索/探索选择器及 A*，但 Fetch 控制、
观测深度栅格与圆形物体抓放是适配层。使用仿真里程计，未验证真机 SLAM/RMP 和 mLLM；
目标类别有限，投递箱位置已知，不能宣称室外、全房覆盖或任意物体抓取能力。
物体可能只在目标房间内移动，跨房间要求指机器人寻找并返回；挪动由独立环境力控制执行。
接触检查含机器人和目标载荷对原生环境的接触，0.5 N 阈值不构成未来无碰撞保证。
早期成对试验未显示动态版本的完整任务优势，已在回复与历史日志中如实保留。

请作者最终核对真实实验原始记录、署名/单位/资助、AI 使用声明、期刊页数与门户截止日期。
此过程只生成本地文件，未上传 Overleaf、提交期刊或推送 GitHub。
''')
    with zipfile.ZipFile(args.output/"OVERLEAF_DREAM_CROSSROOM_R2.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(required_tex_files(args.manuscript)|{args.manuscript/r["file"] for r in pdfs}):
            add(archive,file,file.relative_to(args.manuscript))

    code=set()
    for directory in (REPO/"src/dream",REPO/"tests",REPO/"experiments/configs"):
        code.update(f for f in directory.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    code.update(f for f in (REPO/"experiments").iterdir() if f.is_file() and f.suffix in (".py",".md",".txt"))
    code.update((REPO/"README.md",REPO/"LICENSE",ROOT/"clips/README.md",ROOT/".dream_model_cache/dream_models.lock.json"))
    small={".py",".json",".jsonl",".csv",".log",".md",".txt",".yaml",".yml",".pdf"}
    for directory in (REPO/"experiments/results").iterdir():
        if directory.is_dir():
            code.update(f for f in directory.rglob("*") if f.is_file() and "__pycache__" not in f.parts and
                        (f.suffix in small or ("frozen_tasks" in f.parts and f.suffix==".npz")))
    for directory in (ROOT/"clips/crossroom_v3_development",args.output,args.visual_review):
        code.update(f for f in directory.rglob("*") if f.is_file() and not f.is_symlink() and
                    f.suffix in small|{".npz"} and ".zip" not in f.suffixes)
    with zipfile.ZipFile(args.output/"DREAM_CROSSROOM_CODE_AND_LOGS.zip","x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for file in sorted(code):
            add(archive,file)
    with zipfile.ZipFile(args.output/"DREAM_CROSSROOM_10_VIDEOS.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        replacements={}
        for index,case in enumerate(cases,1):
            run=Path(case["source_run"])
            add(archive,run/"reviewer_view.mp4")
            replacements[f"{index:02d}_{case['scene']}/reviewer_view.mp4"]="../../"+str((run/"reviewer_view.mp4").relative_to(ROOT))
        for file in args.gallery.rglob("*"):
            if not file.is_file() or file.suffix==".mp4":
                continue
            if file.suffix in (".html",".md"):
                content=file.read_text()
                for original,replacement in replacements.items():
                    content=content.replace(original,replacement)
                archive.writestr(str(file.relative_to(ROOT)),content)
            else:
                add(archive,file)
        page=str((args.gallery/"index.html").relative_to(ROOT))
        archive.writestr("OPEN_CROSSROOM_VIDEOS.html",f'<!doctype html><meta http-equiv="refresh" content="0;url={page}"><a href="{page}">Open cross-room videos</a>')
    with zipfile.ZipFile(args.output/"DREAM_CROSSROOM_SELECTED_RAW.zip","x",compression=zipfile.ZIP_STORED) as archive:
        for case in cases:
            run=Path(case["source_run"])
            for file in sorted(run.rglob("*")):
                if file.is_file() and "__pycache__" not in file.parts and file.suffix in (".npz",".png",".jpg",".mp4") and file.name!="reviewer_view.mp4":
                    add(archive,file)
    packages=[]
    for file in sorted(args.output.glob("*.zip")):
        with zipfile.ZipFile(file) as archive:
            bad=archive.testzip()
            if bad:
                raise RuntimeError(f"Archive CRC failure: {file.name}: {bad}")
            members=len(archive.infolist())
        packages.append(dict(file=file.name,bytes=file.stat().st_size,sha256=digest(file),members=members,crc_verified=True))
        print(json.dumps(packages[-1]),flush=True)
    report=dict(distinct_successful_houses=10,pdfs=pdfs,packages=packages,
        gallery=str(args.gallery.relative_to(ROOT)),boundary="Curated development successes, not an unbiased 100% success-rate benchmark.")
    (args.output/"RELEASE_VERIFICATION.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
