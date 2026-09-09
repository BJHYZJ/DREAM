# Code organization and source preservation

## Branches

- `realtime`: original ROS/hardware implementation, plus a link to simulation.
- `simulation`: this standalone Python reproduction project. Hardware meshes,
  drivers and ROS launch/configuration files are not part of this checkout.
- `dream-web` / `master`: presentation only; videos and posters, without copied
  per-case metadata. Detailed provenance belongs in this code repository.

`git switch realtime` and `git switch simulation` select the two implementations
in a normal clone. Finish or commit local edits before switching. For
simultaneous use, use two ordinary clones. Generated caches/results are ignored.
These are alternative project trees, not a feature branch intended to be merged
back wholesale into the hardware tree.

## Public Python package

`src/dream_sim/run.py` coordinates locked-profile validation, one new policy
execution and independent audits. `prepare.py` downloads upstream resources;
`audit.py`, `render.py`, `video.py` handle explicitly separate evidence/display
steps. `sources.py` verifies the source bundle and expands it automatically.
`verify_evidence.py` checks retained evidence and video-number mappings.

Install from a full checkout with `pip install --no-deps -e .` after installing
the pinned requirements. This release supports editable checkout installation,
not a standalone wheel lacking the source archives/configuration/evidence.

## Why preserve source versions?

The selected recordings were produced during development with six different
controller snapshots, not one frozen benchmark controller. The separate
comparison used a seventh version. Review-script versions are independently
pinned, and case 05 needs an additional historical v3 reviewer snapshot.
Replacing them with one rewritten controller would require new experiments;
renaming a folder must not imply that such experiments have happened.

`reproducibility/source_archives/selected_profiles_v1.zip` is an ordinary ZIP of
inspectable Python, tasks, locks and original records. It contains no model
weights or scene meshes, and is not saved actions substituted for a policy.
The adjacent lock binds the archive and every member's bytes/SHA256 to the
pre-refactor release. Existing evidence archives are retained separately.

At first use, the source bundle is checked and expanded beneath ignored
`.runtime/sources/<archive-sha256>/`. Extraction is locked across processes,
rejects unsafe paths and symbolic links, verifies every member, and publishes
the tree only when complete. Later processes verify the expanded tree again;
modified caches are rejected, not silently repaired. Set
`DREAM_SIM_SOURCE_CACHE=/absolute/path` for a different cache location.

```bash
python -m dream_sim.sources --case 01
```

This prints the exact controller source directory. Open its
`experiments/instruction_policy.py`, `experiments/maniskill_crossroom_policy.py`,
and `src/dream/` to inspect the implementation. The original `DREAM_code`
workspace name exists only inside this historical compatibility tree and newly
frozen run records: old code uses it to establish relative cache/source paths.
It is not the public package name or an extra repository to install. No frozen
Python module was renamed, reformatted or algorithmically changed in this split.

## Cases, configurations and evidence

`configs/cases.json` mirrors the ten original catalog entries exactly. Its
historical `task`, `records` and source-run fields are relative to the archived
catalog or identify the original recording; they are not current checkout
download links. Readable `configs/tasks/` and `configs/locks/` are byte-identical
copies checked against that catalog. Execute by case ID through the public
runner; do not edit a locked task to make an integrity check pass.

Controller versions, task bytes, source IDs, original videos, reviewers and the
07/10 spectator renderers are independently bound. Case 05's original false
scorer verdict and pre-declared correction remain visible. Prior wrapper
reviewer-version failures and the comparison's failures have not been removed.

For new algorithm development, create new configurations/source versions and
run new trials with their own evidence. Do not label them as reproductions of
these original videos merely because the public entrypoint is the same.

## Scope of this reorganization

This change moves code, adds packaging/source-integrity checks, and changes
website links/poster locations. It does not change a robot policy, scorer,
seed, recorded action, video frame or paper result. Previously completed
ten-case policy reproduction is documented in `reproducibility/evidence/reproduction/`.
New packaging tests or physics replay are identified separately and must
not be counted as ten additional policy trials.
