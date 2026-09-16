"""Build a render-compatible scene cache while preserving upstream files."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path

from dream_sim.sources import safe_member


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repair_uv_index(payload):
    """Supply UV0 for a primitive exposing only UV1; preserve all binary buffers."""
    if payload[:4] != b"glTF" or struct.unpack_from("<I", payload, 4)[0] != 2:
        raise ValueError("Expected glTF binary version 2")
    length, kind = struct.unpack_from("<II", payload, 12)
    if kind != 0x4E4F534A:
        raise ValueError("Expected the GLB JSON chunk first")
    original = json.loads(payload[20 : 20 + length])
    document = copy.deepcopy(original)
    repairs = []
    for mi, mesh in enumerate(document.get("meshes", [])):
        for pi, primitive in enumerate(mesh.get("primitives", [])):
            attributes = primitive.get("attributes", {})
            indices = sorted(int(key[9:]) for key in attributes if key.startswith("TEXCOORD_"))
            if indices == [1]:
                attributes["TEXCOORD_0"] = attributes["TEXCOORD_1"]
                primitive["attributes"] = dict(sorted(attributes.items()))
                repairs.append(dict(mesh=mi, primitive=pi, added="TEXCOORD_0", alias="TEXCOORD_1"))
            elif indices and indices != list(range(indices[-1] + 1)):
                raise ValueError(f"Unsupported texture-coordinate index gap: {indices}")
    if not repairs:
        return payload, []
    restored = copy.deepcopy(document)
    for row in repairs:
        del restored["meshes"][row["mesh"]]["primitives"][row["primitive"]]["attributes"][
            "TEXCOORD_0"
        ]
    if restored != original:
        raise ValueError("Repair changed unrelated GLB data")
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    remaining = payload[20 + length :]
    total = 20 + len(encoded) + len(remaining)
    result = b"glTF" + struct.pack("<IIII", 2, total, len(encoded), kind) + encoded + remaining
    return result, repairs


def build_cache(source, destination, reference_lock, output_lock):
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists() or output_lock.exists():
        raise FileExistsError("Choose a new cache and output lock")
    lock = json.loads(reference_lock.read_text())
    if not lock.get("complete") or lock.get("failures"):
        raise ValueError("Source asset lock is incomplete")
    destination.mkdir(parents=True)
    updated = copy.deepcopy(lock)
    changes = []
    for name, expected in lock["artifacts"].items():
        relative = safe_member(name)
        src = source / "data/scene_datasets" / relative
        if src.stat().st_size != expected["bytes"] or digest(src) != expected["sha256"]:
            raise ValueError(f"Source asset mismatch: {name}")
        dst = destination / "data/scene_datasets" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        repairs = []
        if src.suffix == ".glb":
            payload, repairs = repair_uv_index(src.read_bytes())
        if repairs:
            dst.write_bytes(payload)
            updated["artifacts"][name].update(bytes=len(payload), sha256=digest(dst))
            changes.append(
                dict(
                    file=name,
                    source_sha256=expected["sha256"],
                    rendered_sha256=digest(dst),
                    binary_buffers_unchanged=True,
                    repairs=repairs,
                )
            )
        else:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        if digest(src) != expected["sha256"]:
            raise ValueError("Upstream file changed during cache preparation")
    updated["upstream_lock_sha256"] = digest(reference_lock)
    updated["render_compatibility"] = dict(
        version=1,
        changes=changes,
        method="Alias missing TEXCOORD_0 to existing TEXCOORD_1; preserve materials, positions, and binary buffers",
    )
    output_lock.parent.mkdir(parents=True, exist_ok=True)
    output_lock.write_text(json.dumps(updated, indent=2) + "\n")
    return dict(
        assets=len(updated["artifacts"]),
        changed_assets=len(changes),
        output_lock_sha256=digest(output_lock),
        upstream_files_preserved=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-lock", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_cache(args.source_dir, args.output_dir, args.reference_lock, args.output_lock),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
