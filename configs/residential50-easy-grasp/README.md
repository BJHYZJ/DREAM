# Fifty-house tasks with graspable object models

This is the task set behind the current **38/50 (76%)** evaluation and the 50 videos on the project website. It retains all 50 houses, layouts, relocation paths, and seed 42 from the earlier diverse-object configuration, replacing the pickup model in 33 tasks.

The tasks use **21 native-scale models across six categories**, with 8–9 tasks per category. Model selection screened 541 available model records, excluding very thin, large, slender, and unstable candidates. Of 22 candidates, 21 passed the short support/placement checks; Apple_19 was excluded after excessive velocity following an offset release. Models can appear in multiple houses.

Those checks used tray support and plate releases at the center and at ±2 cm. They were object-selection checks, not complete robot pickup trials. Full task outcomes come from the separately recorded policy executions.

| File | Contents |
| --- | --- |
| `task_manifest.json` | All 50 task and room-map hashes |
| `object_changes.csv` | Pickup models before and after substitution |
| `object_selection.json` | Accepted/rejected candidates and selection measurements |
| `tasks50/` | Task definitions and evaluator room maps |

After following the [setup guide](../../docs/reproduction.md), inspect or execute the current protocol:

```bash
DREAM_PYTHON="$(command -v python)" ./scripts/run_residential.sh \
  --gpus 0 --workers-per-gpu 1 --raster-threads 2 \
  --output results/residential50 --execute
```

Use the GPU and worker count in your local resource configuration. Omitting `--execute` prints the plan. The wrapper selects `continuous_return`, 1800 robot-action seconds, and no fixed server deadline. All 38 successes pass independent physics, observation, contact, and both arm-return checks; all 12 failures remain in the denominator.

[Current results and records](../../reproducibility/evidence/residential-fast-return/) · [Historical compact-controller result, 36/50](../../reproducibility/evidence/residential-evaluation/)
