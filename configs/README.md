# Task configuration

[`residential50/task_manifest.json`](residential50/task_manifest.json) defines the 50-house study: one cross-room task per house, seed 42 throughout, and dynamic memory. Each task and room map is bound to a SHA256 checksum. The accompanying asset lock records the required scene files.

| Directory | Contents |
| --- | --- |
| `residential50/` | Fifty-house task manifest, configurations, room maps, and scene-asset lock |
| `tasks/` | Earlier demonstration task definitions and evaluator room maps |
| `locks/` | Model revisions and earlier scene-asset records |

`cases.json` maps the original ten demonstrations to their recorded profiles. Those profiles retain their controller, task, seed, and evaluation version. Their outcomes are stored separately in the [experiment records](../reproducibility/evidence/).

The runner verifies task and source checksums before execution. See the [run guide](../docs/reproduction.md) for the residential study command and output files.
