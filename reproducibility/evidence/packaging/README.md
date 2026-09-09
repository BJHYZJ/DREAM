# Entrypoint validation — 2026-09-09

The `dream_sim.run` entrypoint was checked with one case-01 execution in the reference environment. The task completed and passed its matching-version physics and record checks with one attempt and the recorded seed.

The validation also covered all 2,193 files in the controller source archive, 81 offline tests, and the 1,339 locked scene assets. The controller files matched the preceding source version byte for byte.

`manifest.json` records the case, runtime checks, and action/trajectory comparisons. `case01_checks.zip` contains the policy, protocol, and evaluation JSON/JSONL files with member checksums. Large sensor arrays, videos, and logs are listed by hash in the manifest and retained in the authors' archive. Controller sources are stored under `../../source_archives/`.

This single-case validation is separate from the [ten-case reproduction](../reproduction/README.md) and [60-attempt comparison](../study/README.md). Run `python -m dream_sim.verify_evidence` to check the stored files.
