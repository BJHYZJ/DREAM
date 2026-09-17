# Residential manipulation with faster arm return

The `continuous_return` controller completes **38 of 50 tasks (76%)**. Every counted success passes independent physics replay, observation checks, and checks of both loaded and empty arm return. All 12 unsuccessful tasks remain in the denominator.

Each house has one native-scale pickup/place task, seed 42, dynamic memory, an **1800-second robot-action budget**, and **no fixed server execution deadline**. Queue waiting is excluded from execution clocks. The controller retains its navigation iteration and stagnation stopping rules. The cohort contains 21 pickup models from six categories, selected for graspable geometry. These houses were used during controller development; this is not a held-out generalization test or an isolated memory ablation. Hosted mLLM verification is disabled in this simulation configuration.

| File | Contents |
| --- | --- |
| [results.json](results.json) | All 50 outcomes, action times, source hashes, and review identities |
| [failures.json](failures.json) | All 12 failures, event evidence, measurements, and uncertainty in the causal interpretation |
| [return_times.json](return_times.json) | Loaded and empty return durations for all 32 jointly qualified tasks in the staged/fast comparison |
| [records.zip](records.zip) | Original protocol, task definitions and maps, all task results, frozen source identities, and all 38 physics and observation reviews |
| [videos.json](videos.json) | All 50 current video identities, playback rates, source hashes, and outcomes |
| [video_records.zip](video_records.zip) | All 50 external and head-camera replay checks, semantic reconstruction records, frame times, and executed rendering sources |
| [manifest.json](manifest.json) | File and archive-member sizes and SHA-256 hashes |

The reported endpoint is **audit-qualified task completion**. Original public-review records also retain a separate `strict_success` field for the more restrictive recording protocol; that field is not the endpoint used to calculate 38/50. The portable result's `strict_pass` means that task completion and independent reviews passed. The original fields and records are preserved in the archive.

## Failure analysis

| Recorded outcome or interpretation | Tasks |
| --- | ---: |
| Pickup search or navigation incomplete | 4 |
| Placement search or navigation incomplete | 2 |
| Global navigation iteration limit, inferred from termination and control flow | 2 |
| Placement search reached the 1800-second action budget | 2 |
| Released object did not satisfy stable placement | 1 |
| Grasp alignment failed; empty-arm recovery found no feasible path | 1 |

Cases 05 and 07 are marked as **inferred** iteration-limit terminations because the original logs do not explicitly record the final loop index. Case 26 records only 0.35 seconds of stable placement. Case 46 records an unreachable grasp alignment followed by unsuccessful empty-arm recovery. A search-stage label does not prove correct-object acquisition or physical reachability. These causal explanations use recorded events and evaluator measurements. All 50 recordings, including every failed task, were physically replayed for video export; each replay preserved the original task criteria and outcome. This verifies the recorded behavior without proving a unique cause for every failure.

Long-search supplements for cases 05, 07, 30, and 32 remove the action deadline and global navigation iteration limit. They are separate experiments and do not change this cohort's outcomes. Their trajectories must be compared with the original control and physical-state prefixes before interpreting any later success as an extension.

## Arm-return timing comparison

Among the 32 tasks qualified in both this run and the slower staged-return run, median complete loaded return takes 44.55 → 20.98 seconds, and empty return takes 52.40 → 24.38 seconds. Each duration includes the intermediate postures and final settling. These paired timing comparisons exclude tasks that did not qualify in both runs; success rates retain all 50 tasks.

The [historical compact-controller result](../residential-evaluation/) remains 36/50 at 900 action seconds with a 2700-second server watchdog. The [earlier five-recipe study](../residential50-seed42/) retains its own task definitions and results. Its recordings are separate from the current gallery.

## Current video coverage

The main gallery includes all 38 qualified successes and all 12 failures from this exact cohort. Complete timelines play at 12×. Trials 01 and 11 also include 1× grasp and placement excerpts with both arm returns. Each video combines the external replay, a synchronous head-camera replay, the latest saved semantic observation with its capture time, and a reconstructed semantic heatmap with logged targets and routes. Newly rendered camera images never enter the policy or memory. The heatmap uses the frozen encoder and original observations, with every recorded voxel-count update checked; it shows feature alignment before candidate rejection.

The separately labeled extended-search case 07 belongs to the four-case follow-up and does not replace main trial 07 or change 38/50.

## Verify and reproduce

From the repository root:

```bash
python -m dream_sim.evaluate
```

This verifies all file and ZIP-member hashes, binds the 50 outcomes to their frozen tasks and independent success reviews, and reconstructs the evaluated `continuous_return` source. It does not run the policy again or perform a new physics replay. The original RGB-D arrays are not distributed in this compact archive.

Follow the [run guide](../../../docs/reproduction.md) to configure your machine, prepare the assets and models, run a new 50-task cohort, and independently review it. `scripts/run_residential.sh` selects this controller with 1800 action seconds and no server deadline. Use `python -m dream_sim.evaluate --cohort compact` only for the historical 36/50 compact-controller archive.
