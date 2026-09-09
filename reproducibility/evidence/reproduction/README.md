# Final delivery reproduction check — 2026-09-09

All ten published profiles were executed once in the documented separate Python
environment on the same server. The new `actions.json` and
`evaluator_trajectory.json` match their original runs byte-for-byte in all ten
cases. No policy retries, changed seeds or saved-action inputs were used in
these learned-policy runs. The complete existing asset lock (1,339 files) was
verified; the fixed model cache was reused. This is not an all-assets-fresh
download check or a cross-hardware guarantee.

Read `manifest.json` and the per-case ZIPs together. Each archive retains the
fresh controls, forces, event/memory records, original score, initial audit,
and subsequent matching-version independent physics/record checks. All ten
matching-version audits passed. Cases 07 and 10 also passed display-only
replays using their original spectator renderer versions. These audits and
re-renders are not additional learned-policy trials.

The initial packaging wrapper selected older reviewers for 02 and 06, which
checked the wrong rediscovery event and reported failures. These reports and
the initial wrapper source are preserved, not overwritten. Reviewer selection
was corrected to the script hashes already recorded with the original videos;
the same unchanged episodes were re-audited. Case 05 retains its original false
score and its **pre-declared** evaluator-v4 correction. This is not a new scoring
change chosen after this run failed. Case 07 retains its previously declared
physical-accounting version. The manifest records these distinctions per case.

Large sensor arrays and rendered videos are omitted from these compact ZIPs;
their original bytes/hash identities are listed. Raw data remain in the private
workspace archive. Immutable source copies are deduplicated under
the policy profiles and audit sources inside `../../source_archives/`
(locate expanded paths with `python -m dream_sim.sources`);
the original protocol records bind their hashes. Absolute paths inside original
JSON are historical provenance, not runnable installation instructions. Use
the portable commands in `../../../docs/reproduction.md` to create fresh full records.

These are selected development demonstrations, not a 100% benchmark or evidence
of superior dynamic-memory performance. The separate 60-attempt comparison,
including every failure, remains unchanged under `../study/`.
