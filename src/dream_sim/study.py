"""Run a common controller over a declared set of houses, seeds, and memory variants."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from dream_sim.sources import PROJECT, engine_root, safe_member, verify_public_configs


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def controller_overrides(directory: Path, base_source_id: str) -> dict[str, bytes]:
    manifest = json.loads((directory / "controller.json").read_text())
    if manifest["base_study_source_id"] != base_source_id:
        raise ValueError("Controller revision requires a different base implementation")
    files = {}
    for name, digest in manifest["overrides"].items():
        relative = safe_member(name)
        if relative.parts[0] != "experiments" or relative.suffix != ".py":
            raise ValueError("Controller overrides must be experiment Python modules")
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Controller override escapes its directory")
        content = path.read_bytes()
        if sha(content) != digest:
            raise ValueError(f"Controller revision checksum mismatch: {name}")
        files[name] = content
    if not files:
        raise ValueError("Controller revision contains no implementation")
    return files


def materialize_controller(source: Path, destination: Path, overrides: dict[str, bytes]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    paths = [*(source / "experiments").glob("*.py"), *(source / "src/dream").rglob("*.py")]
    names = {path.relative_to(source).as_posix() for path in paths}
    if not set(overrides) <= names:
        raise ValueError("Controller override does not match an existing module")
    for path in paths:
        name = path.relative_to(source).as_posix()
        content = overrides.get(name)
        if content is None:
            content = path.read_bytes()
        output = destination / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        if sha(output.read_bytes()) != sha(content):
            raise OSError(f"Controller copy verification failed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["baseline", "recovery"], default="recovery")
    parser.add_argument("--cases", nargs="+", choices=[f"{i:02d}" for i in range(1, 11)],
                        default=[f"{i:02d}" for i in range(1, 11)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--variants", nargs="+", choices=["dynamic", "static"],
                        default=["dynamic", "static"])
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, default=PROJECT / ".runtime/assets")
    parser.add_argument("--model-cache", type=Path, default=PROJECT / ".runtime/models")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for name in ("cases", "seeds", "variants"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate {name}")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory")

    engine = engine_root()
    sys.path.insert(0, str(engine / "experiments"))
    from run_instruction_profile import load_catalog, verify_case, verify_source

    verify_public_configs()
    root, catalog = load_catalog(engine / "experiments/repro_profiles/profiles.json")
    source_id = catalog["study"]["source_id"]
    source = verify_source(root, catalog["sources"][source_id])
    by_id = {case["id"]: case for case in catalog["cases"]}
    tasks = [verify_case(root, by_id[key]) for key in args.cases]
    overrides = (controller_overrides(PROJECT / "controllers/recovery", source_id)
                 if args.controller == "recovery" else {})
    full_comparison = len(tasks) == 10 and len(args.seeds) == 3 and set(args.variants) == {"dynamic", "static"}
    print(json.dumps({"controller": args.controller, "base_source_id": source_id,
                      "source_overrides": {key: sha(value) for key, value in overrides.items()},
                      "cases": args.cases, "seeds": args.seeds, "variants": args.variants,
                      "planned_attempts": len(tasks) * len(args.seeds) * len(args.variants),
                      "execute": args.execute, "output": str(output)}, indent=2), flush=True)
    if not args.execute:
        return
    cache = args.model_cache.resolve()
    expected = json.loads((root / catalog["model_lock"]["file"]).read_text())
    if json.loads((cache / "dream_models.lock.json").read_text()) != expected:
        raise ValueError("Prepared models differ from the recorded production versions")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dream-controller-", dir=output.parent) as temporary:
        revised = Path(temporary) / "DREAM_code"
        materialize_controller(source, revised, overrides)
        command = [sys.executable, str(engine / "experiments/run_instruction_frozen_batch.py"),
                   "--source-repo", str(revised), "--tasks", *map(str, tasks),
                   "--output", str(output), "--heading-navigation", "--seeds", *map(str, args.seeds),
                   "--variants", *args.variants, "--gpus", *args.gpus, "--wall-timeout", "14400",
                   "--asset-dir", str(args.asset_dir.resolve()), "--model-cache", str(cache)]
        if full_comparison:
            command.append("--benchmark")
        subprocess.run(command, cwd=engine, check=True)


if __name__ == "__main__":
    main()
