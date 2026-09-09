"""Evaluate recorded episodes using physics replay and versioned record reviewers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from dream_sim.run import CATALOG, audit, digest, load_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case", nargs="+", help="Subset of already completed case IDs")
    args = parser.parse_args()
    root, output = args.execution.resolve(), args.output.resolve()
    if args.workers < 1:
        parser.error("Positive worker count required")
    _, catalog = load_catalog(CATALOG)
    cases = [case for case in catalog["cases"] if args.case is None or case["id"] in args.case]
    if args.case and {case["id"] for case in cases} != set(args.case):
        parser.error("Unknown case ID")
    for case in cases:
        if not (root / ("case_" + case["id"]) / "execution" / "batch_result.json").is_file():
            raise ValueError(f"Case {case['id']} has not completed its policy attempt")
    output.mkdir(parents=True, exist_ok=False)

    def one(case: dict) -> dict:
        previous = root / ("case_" + case["id"])
        batch = previous / "execution"
        original_hash = digest(previous / "acceptance.json") if (previous / "acceptance.json").is_file() else None
        attempts = json.loads((batch / "batch_result.json").read_text())["attempts"]
        if len(attempts) != 1:
            raise ValueError("Expected exactly one preserved policy attempt")
        attempt = attempts[0]
        run = batch / attempt["name"]
        protocol = json.loads((batch / "protocol.json").read_text())
        env = os.environ.copy()
        env.update(MS_ASSET_DIR=protocol["asset_root"], HF_HUB_CACHE=protocol["model_cache"],
                   HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   PYTHONDONTWRITEBYTECODE="1")
        env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
        inputs = {str(path.relative_to(previous)): digest(path) for path in run.glob("*.json")}
        result = audit(case, batch, run, output / ("case_" + case["id"]), env)
        unchanged = all(digest(previous / name) == value for name, value in inputs.items())
        row = {"case": case["id"], "scene": case["scene"], "seed": case["seed"],
               "policy_execution": str(previous), "original_policy_protocol_success": attempt["protocol_success"],
               "audit": result, "original_acceptance_sha256": original_hash,
               "original_record_json_sha256": inputs, "original_record_json_unchanged": unchanged,
               "new_policy_execution": False, "passed": result["passed"] and unchanged}
        (output / ("case_" + case["id"]) / "acceptance.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps({"case": case["id"], "passed": row["passed"]}), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(one, cases))
    report = {"cases": rows, "all_passed": all(row["passed"] for row in rows),
              "new_policy_execution": False, "original_attempts_retained": True,
              "boundary": "Physics replay and record validation for previously recorded episodes."}
    (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
