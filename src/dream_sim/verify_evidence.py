"""Verify experiment archive checksums and video-to-case mappings offline."""

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


def validate_gallery_manifest(manifest, catalog):
    """Numeric public labels must still identify the exact frozen case."""
    profiles = {row["id"]: row for row in catalog["cases"]}
    identifiers = [f"{index:02d}" for index in range(1, 11)]
    if [row.get("profile_id") for row in manifest["cases"]] != identifiers:
        raise ValueError("Gallery must map each public video to exactly one ordered profile")
    for row in manifest["cases"]:
        identifier = row["profile_id"]
        profile = profiles[identifier]
        if row["video"] != identifier + ".mp4" or row["index"] != int(identifier):
            raise ValueError("Public video numbering does not match its profile")
        for key in ("scene", "seed", "instruction", "source_id", "task_sha256"):
            if row[key] != profile[key]:
                raise ValueError(f"Gallery changes original profile {identifier}: {key}")
        if row["original_video_sha256"] != profile["video_sha256"]:
            raise ValueError("Gallery changes original recording identity")
        if row["source_run"] != profile["original_source_run"]:
            raise ValueError("Gallery changes original source-run provenance")


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
    evidence = EVIDENCE
    study = json.loads((evidence / "study" / "manifest.json").read_text())
    if study["attempt_archives"] != 60:
        raise ValueError("Incomplete attempt archive set")
    verify_files(evidence / "study", study["files"])
    components = json.loads((evidence / "components" / "manifest.json").read_text())
    verify_files(
        evidence / "components", {row["file"]: row for row in components["groups"].values()}
    )
    reproduction = json.loads((evidence / "reproduction" / "manifest.json").read_text())
    if reproduction["new_policy_attempts"] != 10 or len(reproduction["cases"]) != 10:
        raise ValueError("Incomplete ten-case reproduction records")
    verify_files(evidence / "reproduction", reproduction["files"])
    gallery_root = evidence / "gallery"
    gallery = json.loads((gallery_root / "manifest.json").read_text())
    catalog = json.loads((gallery_root / gallery["profile_catalog"]).read_text())
    validate_gallery_manifest(gallery, catalog)
    verify_files(gallery_root, gallery["files"])
    for row in gallery["cases"]:
        encoding = json.loads((gallery_root / row["encoding_report"]).read_text())
        if encoding["output_sha256"] != row["video_sha256"]:
            raise ValueError("Gallery video identity differs from its encoding record")
    packaging = json.loads((evidence / "packaging" / "manifest.json").read_text())
    if packaging["new_policy_attempts"] != 1 or packaging["case_ids"] != ["01"]:
        raise ValueError("Unexpected packaging smoke-test scope")
    verify_files(evidence / "packaging", packaging["files"])
    comparisons = {}
    for name in ("recovery-study", "recovery-v2-study"):
        if (evidence / name).exists():
            comparisons[name] = verify_comparison(evidence / name)
    residential_root = evidence / "residential50-seed42"
    residential = verify_residential(residential_root) if residential_root.exists() else None
    diverse_root = evidence / "residential50-diverse-seed42"
    diverse = verify_residential(diverse_root) if diverse_root.exists() else None
    print(
        json.dumps(
            {
                "diverse_residential_study": diverse,
                "residential_study": residential,
                "compact_evidence_verified": True,
                "attempts_retained": 60,
                "controller_comparison_attempts_retained": comparisons,
                "reproduction_cases_retained": 10,
                "component_groups": len(components["groups"]),
                "separate_packaging_smoke_cases_retained": 1,
                "public_video_profile_mappings_verified": 10,
                "raw_sensor_video_audit_performed": False,
                "new_policy_execution": False,
            }
        )
    )


if __name__ == "__main__":
    main()
