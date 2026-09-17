# Task configuration

The current **38/50 residential evaluation** uses [`residential50-easy-grasp/task_manifest.json`](residential50-easy-grasp/task_manifest.json): 50 houses, seed 42, dynamic memory, and 21 native-scale pickup models across six categories. The main website trial numbers refer to this manifest.

| Directory | Contents |
| --- | --- |
| `residential50-easy-grasp/` | Current 50 tasks, object selection, room maps, and SHA256 identities |
| `residential50/` | Shared 50-house source/rendering asset locks and the earlier five-recipe task set |
| `residential50-diverse/` | Separate historical configuration with 50 distinct pickup models |
| `tasks/` | Ten earlier demonstration / paired-comparison houses and evaluator maps |
| `locks/` | Model revisions and asset lock for those ten houses |

`cases.json` identifies the ten archived profiles used by `dream_sim.profile` and the common-controller comparison. Its `01` is **not** current website trial 01. Use `scripts/run_residential.sh` for the current cohort.

Task and room-map hashes are checked before execution. GPU/CPU allocation is local to the machine: create it with `python -m dream_sim.configure`, as described in the [run guide](../docs/reproduction.md#worker-resources). It is stored in the ignored `.runtime/worker_resource_limits.json` file.
