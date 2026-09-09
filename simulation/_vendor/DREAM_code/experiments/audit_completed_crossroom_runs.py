#!/usr/bin/env python3
"""Audit completed successes while explicitly named local batches finish.

Never retries or changes a failed task. Independent audit subprocesses only
read source runs and write separate reports. --watch has a finite host deadline.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--existing-audits",type=Path,nargs="*",default=[])
    p.add_argument("--watch",action="store_true")
    p.add_argument("--timeout",type=float,default=7200.)
    p.add_argument("--workers",type=int,default=2)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    known={}
    for root in args.existing_audits:
        for file in root.glob("*/audit.json"):
            report=json.loads(file.read_text())
            known[report["source_run"]]=dict(report=str(file.resolve()),passed=report["passed"],existing=True)
    pending={}
    def check(run,destination):
        command=[sys.executable,str(Path(__file__).with_name("audit_crossroom_success.py")),
                 "--run",str(run.resolve()),"--output",str(destination.resolve())]
        with destination.with_suffix(".stdout.log").open("w") as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        file=destination/"audit.json"
        return dict(report=str(file.resolve()),passed=json.loads(file.read_text())["passed"] if file.is_file() else False,
                    returncode=result.returncode,existing=False)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            for batch in args.batches:
                for file in sorted(batch.glob("*/result.json")):
                    run=file.parent
                    key=str(run.resolve())
                    if key in known or key in pending or not (run/"sha256.json").is_file():
                        continue
                    result=json.loads(file.read_text())
                    if not result.get("evaluator_protocol_success",False):
                        continue
                    destination=args.output/f"{batch.name}__{run.name}"
                    pending[key]=pool.submit(check,run,destination)
                    print(json.dumps(dict(event="audit_started",run=key)),flush=True)
            for key,future in list(pending.items()):
                if future.done():
                    known[key]=future.result()
                    del pending[key]
                    print(json.dumps(dict(event="audit_finished",run=key,**known[key])),flush=True)
            finished=all((batch/"batch_result.json").is_file() for batch in args.batches)
            (args.output/"audit_status.json").write_text(json.dumps(dict(
                boundary="Independent control re-executions are evidence checks, not extra learned trials.",
                reports=known,pending=list(pending),batches_finished=finished),indent=2)+"\n")
            if (finished or not args.watch or time.monotonic()-started>args.timeout) and not pending:
                break
            time.sleep(15)


if __name__=="__main__":
    main()
