"""Check the published residential outcomes and their evaluated controller source.

This command reads saved evidence. It does not launch a simulator or estimate a
new success rate for a changed controller.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from dream_sim.io import digest
from dream_sim.sources import PROJECT, engine_root
from dream_sim.study import controller_overrides, materialize_controller
from dream_sim.verify_evidence import verify_files

EVALUATION = PROJECT / "reproducibility/evidence/residential-evaluation"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_results(directory: Path = EVALUATION) -> dict:
    """Verify every task, including failures, against the original result hashes."""
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_files(directory, manifest["files"])
    final = json.loads((directory / "results.json").read_text())
    with zipfile.ZipFile(directory / "records.zip") as archive:

        def record(name: str, expected: str | None = None) -> dict:
            payload = archive.read(name)
            if expected is not None:
                require(hashlib.sha256(payload).hexdigest() == expected, f"Changed record: {name}")
            return json.loads(payload)

        protocol = record("run/protocol.json", final["protocol_sha256"])
        validation = record("validation/source_validation.json", final["source_validation_sha256"])
        tasks = {row["name"]: row for row in protocol["attempts"]}
        require(
            len(tasks) == len(final["results"]) == protocol["planned_attempts"] == 50,
            "The cohort must include all 50 attempts",
        )
        require(len({row["scene"] for row in tasks.values()}) == 50, "House identifiers repeat")
        require(
            protocol["seeds"] == [42] and protocol["variants"] == ["dynamic"],
            "The cohort seed or memory variant changed",
        )
        require(
            protocol["simulation_budget_s"] == 900 and protocol["wall_timeout_s"] == 2700,
            "The task budget differs from the reported protocol",
        )
        seen = set()
        strict = raw = 0
        for row in final["results"]:
            name = row["name"]
            require(name not in seen and name in tasks, f"Duplicate or unknown attempt: {name}")
            seen.add(name)
            task = tasks[name]
            task_name = "run/" + task["task"]
            task_data = record(task_name, task["task_sha256"])
            room_name = str(Path(task_name).parent / task_data["room_map_file"])
            require(
                hashlib.sha256(archive.read(room_name)).hexdigest() == task["room_map_sha256"],
                f"Changed room map: {name}",
            )
            result = record(f"run/{name}/result.json", row["task_result_sha256"])
            record(f"run/screen_results/{name}.json", row["screen_result_sha256"])
            require(
                result["evaluator_task_success"] == row["task_success"], f"Outcome mismatch: {name}"
            )
            require(
                0 <= row["robot_action_seconds"] <= 900, f"Task duration exceeds budget: {name}"
            )
            if not row["task_success"]:
                require(not row["strict_pass"], f"Failure incorrectly counted as success: {name}")
                continue
            raw += 1
            completion = record(
                f"physics/{name}.completion.json", row["independent_completion_sha256"]
            )
            review = record(
                f"physics/{name}/screen_physics_review.json", row["independent_review_sha256"]
            )
            require(
                completion["returncode"] == 0 and not completion["wall_timeout"],
                f"Independent replay did not complete: {name}",
            )
            require(
                all(value == 0 for value in completion["oom_kill_delta"].values()),
                f"Independent replay exceeded memory: {name}",
            )
            passed = (
                all(review["checks"].values())
                and review["fold"]["passed"]
                and all(review["fold"]["checks"].values())
            )
            require(
                passed == completion["physical_checks_passed"] == row["strict_pass"],
                f"Independent review mismatch: {name}",
            )
            strict += bool(passed)
        require(
            raw == strict == final["strict_successes"] == final["task_successes"] == 36,
            "Completion counts differ from the recorded result",
        )
    return {
        "attempts": 50,
        "strict_successes": strict,
        "strict_success_rate": strict / 50,
        "source_sha256": validation["source_sha256"],
    }


def verify_controller(expected: dict[str, str]) -> None:
    """Reconstruct the public controller and compare it with the evaluated source."""
    engine = engine_root()
    sys.path.insert(0, str(engine / "experiments"))
    from run_instruction_profile import load_catalog, verify_source

    catalog_root, catalog = load_catalog(engine / "experiments/repro_profiles/profiles.json")
    source_id = catalog["study"]["source_id"]
    source = verify_source(catalog_root, catalog["sources"][source_id])
    overrides = controller_overrides(PROJECT / "controllers/compact_v1", source_id)
    with tempfile.TemporaryDirectory(prefix="dream-controller-check-") as temporary:
        target = Path(temporary) / "DREAM_code"
        materialize_controller(source, target, overrides)
        actual = {str(path.relative_to(target)): digest(path) for path in target.rglob("*.py")}
        require(actual == expected, "The installed controller differs from the evaluated source")


def main() -> None:
    result = verify_results()
    verify_controller(result.pop("source_sha256"))
    print(
        json.dumps(
            {
                **result,
                "controller_matches_evaluated_source": True,
                "independent_review_records_verified": True,
                "raw_sensor_arrays_rechecked": False,
                "new_policy_execution": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
