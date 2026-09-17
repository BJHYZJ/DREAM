"""Verify published experiment archives, outcomes, and controller source offline."""

import argparse
import hashlib
import json
import zipfile

from dream_sim.sources import EVIDENCE


def verify_files(root, files):
    for name, expected in files.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe evidence path")
        payload = path.read_bytes()
        if (
            len(payload) != expected["bytes"]
            or hashlib.sha256(payload).hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"Evidence file changed: {name}")
        if expected.get("kind") == "zip" or "members" in expected:
            with zipfile.ZipFile(path) as archive:
                if set(archive.namelist()) != set(expected["members"]):
                    raise ValueError("Archive member set changed")
                for member, row in expected["members"].items():
                    payload = archive.read(member)
                    if (
                        len(payload) != row["bytes"]
                        or hashlib.sha256(payload).hexdigest() != row["sha256"]
                    ):
                        raise ValueError(f"Archived record changed: {member}")


def verify_comparison(root):
    """Verify the complete records for a separately versioned controller study."""
    manifest = json.loads((root / "manifest.json").read_text())
    attempts = [
        name for name in manifest["files"] if name.startswith("attempts/") and name.endswith(".zip")
    ]
    if manifest.get("attempt_archives") != 60 or len(attempts) != 60:
        raise ValueError("Incomplete controller comparison archive set")
    for flag in (
        "all_60_outcomes_bound",
        "all_reported_successes_audited",
        "all_counted_successes_passed_audits",
        "all_archive_members_verified",
    ):
        if manifest.get(flag) is not True:
            raise ValueError(f"Unverified controller comparison: {flag}")
    verify_files(root, manifest["files"])
    return len(attempts)


def verify_residential(root):
    """Verify every residential outcome and its compact recording archive."""
    from dream_sim.sources import safe_member

    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        path = root / safe_member(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe residential evidence path")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Residential evidence changed: {name}")
    report = json.loads((root / "analysis/study_analysis.json").read_text())
    rows = report["attempts"]
    if (
        len(rows) != 50
        or len({r["scene"] for r in rows}) != 50
        or len({r["name"] for r in rows}) != 50
        or any(r["seed"] != 42 or r["variant"] != "dynamic" for r in rows)
        or report.get("all_planned_outcomes_bound") is not True
        or report.get("all_counted_successes_passed_audits") is not True
    ):
        raise ValueError("Incomplete residential cohort")
    diverse = any("pickup_model" in row for row in rows)
    if diverse and (
        any(not row.get("pickup_model") for row in rows)
        or len({row["pickup_model"] for row in rows}) != 50
    ):
        raise ValueError("Diverse residential cohort requires 50 distinct pickup models")
    successes = 0
    for row in rows:
        path = root / "attempts" / (safe_member(row["name"]).as_posix() + ".zip")
        if path.relative_to(root).as_posix() not in checksums:
            raise ValueError("Unbound residential recording")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("Corrupt residential recording")
            data = archive.read("result.json")
            if hashlib.sha256(data).hexdigest() != row["result_sha256"]:
                raise ValueError("Residential outcome differs from recording")
            result = json.loads(data)
            if result["evaluator_task_success"] != row["task_success"]:
                raise ValueError("Inconsistent residential task score")
            if diverse:
                task = json.loads(archive.read("environment_task.json"))
                if task["recipe"]["environment_assets"]["pickup"] != row["pickup_model"]:
                    raise ValueError("Pickup model differs from archived task")
            if row["qualified_task_success"]:
                if diverse:
                    from dream_sim.fold_review import review_fold

                    physical = json.loads(archive.read("audit/physical/audit.json"))
                    events = [
                        json.loads(line)
                        for line in archive.read("events.jsonl").decode().splitlines()
                    ]
                    fold = review_fold(
                        events,
                        json.loads(archive.read("actions.json")),
                        physical.get("contact_audit", {}),
                    )
                    if not fold["passed"]:
                        raise ValueError(
                            "Counted residential success lacks verified compact-arm returns"
                        )
                review = json.loads(archive.read("audit/record/record_review.json"))
                if (
                    not row["task_success"]
                    or review.get("primary_task_record_review_passed") is not True
                ):
                    raise ValueError("Counted residential success lacks a passing audit")
                successes += 1
    if report["statistics"]["variants"]["dynamic"] != dict(
        attempts=50, successes=successes, success_rate=successes / 50
    ):
        raise ValueError("Residential counts differ from complete records")
    return dict(attempts=50, successes=successes, compact_records_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="Also check the earlier complete study cohorts retained for comparison",
    )
    args = parser.parse_args()
    from dream_sim.evaluate import verify_controller, verify_historical_results, verify_results

    current = verify_results()
    verify_controller(current.pop("source_sha256"))
    components = json.loads((EVIDENCE / "components/manifest.json").read_text())
    verify_files(
        EVIDENCE / "components", {row["file"]: row for row in components["groups"].values()}
    )
    comparison = verify_comparison(EVIDENCE / "recovery-v2-study")
    long_search = EVIDENCE / "long-search"
    verify_files(long_search, json.loads((long_search / "manifest.json").read_text())["files"])
    historical = {}
    if args.include_historical:
        study = json.loads((EVIDENCE / "study/manifest.json").read_text())
        verify_files(EVIDENCE / "study", study["files"])
        historical["study_attempts"] = study["attempt_archives"]
        historical["recovery_study_attempts"] = verify_comparison(EVIDENCE / "recovery-study")
        historical["residential50_seed42"] = verify_residential(EVIDENCE / "residential50-seed42")
        compact = verify_historical_results()
        verify_controller(compact.pop("source_sha256"), "compact_v1")
        historical["compact_controller"] = compact
    print(
        json.dumps(
            {
                "current_residential": current,
                "current_controller_matches_evaluated_source": True,
                "paired_comparison_attempts": comparison,
                "component_groups": len(components["groups"]),
                "long_search_files_verified": True,
                "historical": historical,
                "raw_sensor_video_audit_performed": False,
                "new_policy_execution": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
