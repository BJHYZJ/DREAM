#!/usr/bin/env python3
"""Verify installed native scene assets against the recorded download hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--asset-parent",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    failures=[]
    for name,record in manifest["artifacts"].items():
        path=args.asset_parent/name
        if not path.exists():
            failures.append(dict(file=name,error="missing"))
            continue
        checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum!=record["sha256"] or path.stat().st_size!=record["bytes"]:
            failures.append(dict(file=name,error="content_mismatch",actual_sha256=checksum))
    result=dict(checked=len(manifest["artifacts"]),passed=not failures,failures=failures,
        reference_revision=manifest["source_revision"],manifest=str(args.manifest.resolve()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if failures:
        raise SystemExit(1)


if __name__=="__main__":
    main()
