# Reproduction results — 2026-09-09

The ten gallery profiles were each executed once in the reference Python environment on the same server. All ten `actions.json` files and all ten `evaluator_trajectory.json` files matched the corresponding source recordings byte for byte. The existing 1,339-file scene-asset lock was verified, and the prepared model cache was reused.

All ten executions passed their matching-version physics and record checks. Cases 07 and 10 also passed spectator-render checks using their recorded renderer versions. Replays and presentation renders are recorded separately from policy attempts.

## Evaluation versions

| Cases | Evaluation notes |
| --- | --- |
| 02, 06 | The initial wrapper selected older record reviewers and reported a rediscovery failure. Rechecking the same episodes with the reviewer hashes recorded in the gallery passed. Both sets of reports are retained. |
| 05 | The original false score and the previously declared evaluator-v4 reassessment are retained. The correction covers held transport while lowering an object and wall-derived room assignment. |
| 07 | Uses its documented v4 physical-contact accounting and retains its original successful score. |
| 07, 10 | Additional spectator-camera renders reuse the saved control sequence. |

The case manifest records the relevant policy, reviewer, and renderer versions. The evaluation correction for case 05 was specified before this reproduction run.

## Files

`manifest.json` describes the executions and their checksums. The per-case ZIPs contain controls, forces, events, memory records, scores, and evaluation reports. Controller source is deduplicated in `../../source_archives/` and linked by the source hashes in each run record.

Large sensor arrays and videos are omitted from the compact ZIPs. Their hashes are listed in the manifests, and the raw data are retained in the authors' archive. Absolute paths in recorded JSON identify the original execution workspace. Use the [run guide](../../../docs/reproduction.md) to create new executions with complete local outputs.

These results concern the selected gallery profiles in the reference environment. The [memory comparison](../study/README.md) contains the separate 60-attempt evaluation, including failures and uncertainty estimates. Numerical behavior may differ on other hardware.
