# Earlier five-recipe residential evaluation

DREAM completed **27/50 tasks (54%)** across 50 distinct residential scenes. Every task uses seed 42, dynamic memory, and the same `recovery_v6` controller. Five pickup/place combinations appear in ten houses each.

[All outcomes](analysis/attempts.csv) · [Analysis](analysis/study_analysis.json) · [Task configurations](../../../configs/residential50/task_manifest.json)

The houses were fixed using geometry and initialization checks before policy execution. Controller development used ten separate houses; ProcTHOR-Train-3859 was additionally used to diagnose grasp clearance. The controller and uniform 1,800-second simulation budget were selected during development, then frozen before the final 50-task evaluation. Each task ran once in that evaluation. Earlier controller runs are retained separately. All 50 outcomes contribute to the completion rate.

Task completion requires observing the relocated object, a sustained physical lift, cross-room transport, and stable released placement in the instructed receptacle. Counted successes also pass independent control replay and source, observation, contact, and video checks. Continuous correct target tracking is allowed; loss followed by reacquisition is recorded separately.

The archived video metadata describes the earlier 4× recordings. Those videos are no longer displayed on the project website. The current gallery contains all 50 trials from the separate [38/50 cohort](../residential-fast-return/). This table retains all outcomes from the historical five-recipe study.

| File | Contents |
| --- | --- |
| `protocol.json` | Fixed tasks, sources, and evaluation design |
| `analysis/attempts.csv` | Every task outcome, including failures and audit rejections |
| `attempts/<run>.zip` | Controls, events, trajectories, results, and available independent checks |
| `videos.json` | Displayed video metadata and checksums |
| `recording_archives.json` | Hashes of complete raw recording archives retained by the authors |
| `controller_validation.json` | Controller selection and development checks |

See the [run guide](../../../docs/reproduction.md) to execute the fixed task manifest. These are descriptive results for the Fetch simulation adapter; the real-robot studies use their own hardware and task protocols.
