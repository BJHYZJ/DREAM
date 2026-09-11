"""Verify experiment archive checksums and video-to-case mappings offline."""
import hashlib
import json
from pathlib import Path
import zipfile
from dream_sim.sources import EVIDENCE


def verify_files(root, files):
    for name, expected in files.items():
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe evidence path")
        payload = path.read_bytes()
        if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
            raise ValueError(f"Evidence file changed: {name}")
        if expected.get("kind") == "zip" or "members" in expected:
            with zipfile.ZipFile(path) as archive:
                if set(archive.namelist()) != set(expected["members"]):
                    raise ValueError("Archive member set changed")
                for member, row in expected["members"].items():
                    payload = archive.read(member)
                    if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
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
    attempts = [name for name in manifest["files"] if name.startswith("attempts/") and name.endswith(".zip")]
    if manifest.get("attempt_archives") != 60 or len(attempts) != 60:
        raise ValueError("Incomplete controller comparison archive set")
    for flag in ("all_60_outcomes_bound", "all_reported_successes_audited",
                 "all_counted_successes_passed_audits", "all_archive_members_verified"):
        if manifest.get(flag) is not True:
            raise ValueError(f"Unverified controller comparison: {flag}")
    verify_files(root, manifest["files"])
    return len(attempts)


def main():
    evidence = EVIDENCE
    study = json.loads((evidence / "study" / "manifest.json").read_text())
    if study["attempt_archives"] != 60:
        raise ValueError("Incomplete attempt archive set")
    verify_files(evidence / "study", study["files"])
    components = json.loads((evidence / "components" / "manifest.json").read_text())
    verify_files(evidence / "components", {row["file"]: row for row in components["groups"].values()})
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
    print(json.dumps({"compact_evidence_verified": True, "attempts_retained": 60,
                      "controller_comparison_attempts_retained": comparisons,
                      "reproduction_cases_retained": 10, "component_groups": len(components["groups"]),
                      "separate_packaging_smoke_cases_retained": 1,
                      "public_video_profile_mappings_verified": 10,
                      "raw_sensor_video_audit_performed": False, "new_policy_execution": False}))


if __name__ == "__main__":
    main()
