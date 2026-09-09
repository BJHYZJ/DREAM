# Standalone-branch packaging check — 2026-09-09

After the branch/layout reorganization, the new `dream_sim.run` entrypoint
executed case 01 once in the documented environment, using current learned
perception and closed-loop control. It completed and passed its original-version
independent physics and record audits. There were no retries or altered seeds.
This is one additional packaging smoke case, **not ten new experiments**.

The source archive was independently compared, member by member, with the
previous Git release: all 2,193 original files were identical. The new package
also passed 81 offline tests. The preparation wrapper verified all 1,339 locked
scene files without downloading or changing them. These are integrity checks,
not extra robot trials.

`manifest.json` records the fresh attempt, preflight, action/trajectory checksums
and comparison with the prior ten-profile reproduction. `case01_checks.zip`
retains the new JSON/JSONL policy, protocol and audit records, with checksums for
every member. Large sensor arrays/videos/logs are omitted from this compact ZIP;
the manifest records their hashes and the retained server location. Frozen
source files are deduplicated in `../../source_archives/` and bound by the run's
source manifests. Absolute paths inside records are provenance, not installation
instructions.

The ten published videos and earlier reproduction/comparison evidence were not
replaced. No success-rate or cross-hardware reliability claim follows from this
single packaging check. Run `python -m dream_sim.verify_evidence` to verify the
compact records; use the [tutorial](../../../docs/reproduction.md) for new runs.
