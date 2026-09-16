# Residential manipulation evaluation

DREAM completes **36 of 50 tasks (72%)** with one evaluated Fetch controller. All 36 completed tasks pass independent physics, observation, contact, and arm-return checks. All 14 failures remain in the denominator.

Each house has one native-scale pickup/place task, seed 42, dynamic memory, a 900-second robot-action budget, and a separate 2700-second execution watchdog. The cohort contains 21 pickup models from six categories, selected for graspable geometry. These houses were used during controller development; the result describes this cohort and does not measure held-out generalization or isolate the effect of memory.

| File | Contents |
| --- | --- |
| `results.json` | Every task outcome and the hashes of its original records |
| `records.zip` | Original protocol, task inputs, room maps, results, and all 36 independent review records |
| `recording_checksums.json` | Recorded hashes of the full sensor, control, trajectory, and replay files retained in local experiment storage |
| `manifest.json` | Sizes and SHA256 hashes of the public files |

Verify the public outcomes and reconstruct the evaluated controller:

```bash
python -m dream_sim.evaluate
```

The check confirms task and review record integrity and that the reconstructed controller matches the evaluated Python source. It does not execute a new task or recheck the archived raw sensor arrays. A new simulation run generates its own full observations, controls, and trajectories.

See the [run guide](../../../docs/reproduction.md) for execution and independent replay. The separate [dynamic/static memory comparison](../recovery-v2-study/) and [recorded video cohort](../residential50-seed42/) retain their own protocols and outcomes.
