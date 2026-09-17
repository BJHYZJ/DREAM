# Experiment records and reproduction

To install and run DREAM, start with the [run guide](../docs/reproduction.md). This directory holds the source archives needed by the loader and the records supporting the reported experiments.

## Experiments used in the paper and response

| Directory | Experiment | Reproduction entry point |
| --- | --- | --- |
| [residential-fast-return](evidence/residential-fast-return/) | Current 50-house evaluation: 38 successes, 12 failures; all 50 website recordings | `scripts/run_residential.sh`; offline check: `python -m dream_sim.evaluate` |
| [recovery-v2-study](evidence/recovery-v2-study/) | Paired dynamic/static comparison, 60 attempts | [Common-controller comparison](../docs/reproduction.md#evaluate-a-common-controller) |
| [components](evidence/components/) | RMP and exploration measurements | Archived measurement scripts and configurations described in its README |
| [long-search](evidence/long-search/) | Four separate extended searches, including the 78.7-minute completion | Recorded settings and outcomes; [video export](../docs/reproduction.md#extended-search-video) requires complete raw recordings |
| [arm-return](evidence/arm-return/) | Slower staged-return cohort underlying the paired arm-return timing comparison | `DREAM_CONTROLLER=staged_return` and the historical budget in the run guide |

The paired comparison also retains its controller-selection records. Keeping all declared attempts and selection inputs is necessary to interpret its reported uncertainty; these are experimental evidence, not installation prerequisites.

## Earlier complete cohorts

These records are kept to document earlier reported outcomes and comparisons. They do not contribute attempts or videos to the current 38/50 cohort.

| Directory | Result / purpose |
| --- | --- |
| [residential-evaluation](evidence/residential-evaluation/) | Compact controller, 36/50; 900 action seconds and a 2700-second server deadline |
| [residential50-seed42](evidence/residential50-seed42/) | Earlier five-recipe cohort, 27/50 |
| [study](evidence/study/) | Baseline paired comparison: dynamic 7/30, static 6/30 |
| [recovery-study](evidence/recovery-study/) | Earlier common-controller comparison used by the subsequent paired analysis |

`python -m dream_sim.verify_evidence` checks the current residential, paired, component, and long-search records. Add `--include-historical` to verify the earlier complete cohorts too.

## What the archives allow

- **Inspect outcomes:** all published attempts, including failures, and their protocol and source identities.
- **Check integrity:** compare file and ZIP-member hashes with the manifests.
- **Recompute supported summaries:** use the archived inputs described in each experiment README.
- **Run a new experiment:** download models and scenes, then execute the task configuration from a fresh output directory.

The compact archives omit the large original sensor arrays and videos. They are not complete replay datasets. Exact replay, semantic reconstruction, and video export require a complete recording with RGB-D arrays, actions, trajectories, and frozen sources. New runs create these files locally. The prepared scene assets are separate from the recorded sensor images.

Absolute paths inside original JSON records identify the machine used for that experiment. They are retained as provenance; do not copy them into commands on another machine. Use the portable paths in the run guide.

## Required source archive

[source_archives](source_archives/) contains the checksum-locked base implementation used by the controllers. It is a runtime dependency, not an experiment-output folder. The loader combines it with `controllers/` and verifies the resulting source. `.runtime/`, `results/`, model weights, scene caches, and internal packaging checks are excluded from Git.
