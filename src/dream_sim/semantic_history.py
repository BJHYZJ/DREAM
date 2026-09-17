"""Reconstruct semantic voxel features from a recorded observation stream.

Uses the recording's frozen encoder, feature pooling and depth clearing. The
export is raw text-feature alignment before candidate rejection, with no new
object detections, navigation, or policy decisions. Logged candidates and paths
are overlaid separately when composing the video.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dream_sim.io import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    record = run / args.name
    source = run / "frozen_workspace/DREAM_code"
    protocol = json.loads((run / "protocol.json").read_text())
    if any(digest(source / name) != value for name, value in protocol["source_sha256"].items()):
        raise ValueError("Frozen implementation does not match the recorded protocol")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_CACHE=protocol["model_cache"],
        DREAM_MODEL_LOCK_FILE=str(Path(protocol["model_cache"]) / "dream_models.lock.json"),
    )
    sys.path[:0] = [str(source / "experiments"), str(source / "src")]
    import numpy as np
    import torch
    from dream.perception.encoders.siglip_encoder import MaskSiglipEncoder
    from dream.semantic_retrieval import feature_alignment
    from dream_learned_core import RGBDObservation, SemanticMemory
    from instruction_feature_sampling import sampled_mask_siglip

    torch.set_num_threads(2)
    task = json.loads((record / "environment_task.json").read_text())
    from instruction_task import parse_instruction

    instruction = parse_instruction(task["instruction"])
    queries = [instruction.pickup_query, instruction.placement_query]
    events = [json.loads(line) for line in (record / "events.jsonl").read_text().splitlines()]
    output.mkdir(parents=True, exist_ok=False)
    encoder = MaskSiglipEncoder(device="cuda", version="so400m")
    encoder.model.eval()

    class FeatureEncoder:
        @staticmethod
        def dense_sampled(observation, stride):
            with torch.inference_mode():
                return sampled_mask_siglip(encoder, observation.rgb, stride)

    memory = SemanticMemory(FeatureEncoder(), dynamic=True)
    text_features = [encoder.encode_text(query).detach().cpu() for query in queries]
    records = []
    inputs = {}
    for event in events:
        if event["event"] != "observation":
            continue
        path = record / f"observation_{event['frame_id']:05d}.npz"
        with np.load(path) as saved:
            observation = RGBDObservation(
                **{
                    key: saved[key].item() if saved[key].shape == () else saved[key]
                    for key in saved.files
                }
            )
        inputs[path.name] = digest(path)
        update = memory.integrate(observation)
        expected = event["memory"]
        matches = all(
            update[key] == expected[key]
            for key in ("frame_id", "sim_step", "before", "removed", "after", "input_points")
        )
        if not matches:
            raise ValueError(
                f"Memory reconstruction disagrees with recorded counts: {update}, {expected}"
            )
        # Stored images are not needed for feature accumulation or this display.
        memory.observations[observation.frame_id] = None
        cells = np.floor(memory.cloud.points.numpy()[:, :2] / 0.04).astype(np.int32)
        unique, inverse = np.unique(cells, axis=0, return_inverse=True)
        fields = []
        with torch.inference_mode():
            for text in text_features:
                scores = feature_alignment(text, memory.cloud.features).reshape(-1).numpy()
                field = np.full(len(unique), -np.inf, np.float32)
                np.maximum.at(field, inverse, scores)
                fields.append(field)
        filename = f"memory_{observation.frame_id:05d}.npz"
        np.savez_compressed(
            output / filename, xy=(unique + 0.5) * 0.04, pickup=fields[0], placement=fields[1]
        )
        records.append(
            dict(
                step=event["step"],
                frame_id=observation.frame_id,
                file=filename,
                voxels=len(memory),
                counts_match=True,
                sha256=digest(output / filename),
            )
        )
        if len(records) % 20 == 0:
            print(
                json.dumps(
                    dict(
                        case=args.name,
                        observations=len(records),
                        step=event["step"],
                        voxels=len(memory),
                    )
                ),
                flush=True,
            )
    if any(digest(record / name) != value for name, value in inputs.items()):
        raise ValueError("Original observations changed during semantic reconstruction")
    atomic_json(
        output / "semantic_receipt.json",
        dict(
            case=args.name,
            queries=queries,
            protocol_sha256=digest(run / "protocol.json"),
            source_sha256=protocol["source_sha256"],
            observation_sha256=inputs,
            renderer_sha256=digest(Path(__file__)),
            frames=records,
            all_recorded_memory_counts_match=all(row["counts_match"] for row in records),
            new_policy_execution=False,
            new_detections=False,
            boundary="Recomputed frozen-encoder voxel feature alignment before candidate rejection; logged controller decisions are overlaid separately.",
        ),
    )
    print(json.dumps(dict(case=args.name, completed=True, observations=len(records))), flush=True)


if __name__ == "__main__":
    main()
