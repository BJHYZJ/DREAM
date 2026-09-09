#!/usr/bin/env python3
"""Bind the response's selected-example table to existing checked recordings.

This verifies transcription and evidence references, not a new physical run,
video-quality judgment, success-rate denominator or publication approval.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class Articles(HTMLParser):
    def __init__(self):
        super().__init__();self.articles=[];self.current=None

    def handle_starttag(self,tag,attrs):
        if tag=="article":self.current=[]
        if tag=="a" and self.current is not None:
            href=dict(attrs).get("href")
            if href:self.current.append(href)

    def handle_endtag(self,tag):
        if tag=="article" and self.current is not None:
            self.articles.append(self.current);self.current=None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_rows(text):
    pattern=r"((?:ProcTHOR|ArchitecTHOR)-(?:Train|Val|Test)-\d+)\s*&\s*(\d+)\s*&\s*(\w+)\s*\$\\rightarrow\$\s*(\w+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)"
    return [(house,int(seed),pickup.lower(),place.lower(),float(distance),float(stable))
            for house,seed,pickup,place,distance,stable in re.findall(pattern,text)]


def check(page,response):
    parsed=Articles();parsed.feed(page.read_text())
    rows=response_rows(response.read_text());expected={row[:4]:row[4:] for row in rows}
    cases=[];seen=[]
    for refs in parsed.articles:
        physical=[(page.parent/ref).resolve() for ref in refs if ref.endswith("/audit.json")]
        recorded=[(page.parent/ref).resolve() for ref in refs if ref.endswith("/record_review.json")]
        if len(physical)!=1 or len(recorded)!=1:
            cases.append(dict(passed=False,error="Each video needs exactly one physics and recording audit"));continue
        audit=json.loads(physical[0].read_text());record=json.loads(recorded[0].read_text())
        source=Path(audit["source_run"]);task_path=source/"environment_task.json"
        task=json.loads(task_path.read_text());assets=task["recipe"]["environment_assets"]
        key=(task["scene"],int(task["seed"]),assets["pickup"].rsplit("_",1)[0].lower(),
             assets["placement"].rsplit("_",1)[0].lower())
        seen.append(key);values=expected.get(key)
        evaluation=audit["evaluation"]
        actual=(evaluation["held_base_transport_m"],evaluation["stable_placement_s"])
        checks=dict(table_row_present=values is not None,
            rounded_values_match=values is not None and abs(values[0]-actual[0])<.00051 and abs(values[1]-actual[1])<.051,
            physics_passed=bool(audit.get("physical_reexecution_passed")),
            protocol_passed=bool(evaluation.get("evaluator_protocol_success")),
            recording_passed=bool(record.get("record_review_passed")),
            same_source_run=Path(record["source_run"]).resolve()==source.resolve(),
            no_above_threshold_native_contact=audit["contact_audit"]["native_environment_contact_control_steps"]==0)
        cases.append(dict(key=key,reported_values=values,replayed_values=actual,checks=checks,
            passed=all(checks.values()),input_sha256={str(p):sha(p) for p in [physical[0],recorded[0],task_path]}))
    unique=len(rows)==len(expected)==len(seen)==len(set(seen)) and set(seen)==set(expected)
    return dict(table_rows=len(rows),distinct_houses=len({r[0] for r in rows}),cases=cases,
        every_table_row_matches_one_recording=unique,
        response_table_audit_passed=bool(cases) and unique and all(c["passed"] for c in cases),
        response_sha256=sha(response),website_page_sha256=sha(page),release_ready=False,
        boundary="Selected development-example transcription audit only. Independent reports are inputs; no new experiment, unbiased success rate or publication approval is inferred.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page",type=Path,required=True)
    parser.add_argument("--response-tex",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();report=check(args.page,args.response_tex)
    with args.output.open("x") as stream:json.dump(report,stream,indent=2);stream.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k!="cases"},indent=2))
    if not report["response_table_audit_passed"]:raise SystemExit(1)


if __name__=="__main__":main()
