# Component measurements

These records cover memory pruning, policy scaling, and exploration. `manifest.json` lists the archives and the checksums of their contents.

| Measurement | Inputs and interpretation |
| --- | --- |
| RMP replay | Public RGB-D sequences with injected pose drift and supplied corrected poses; RGB vectors serve as feature proxies. |
| RMP threshold, scaling, and timing | Sensitivity and policy-cost measurements; timing excludes feature extraction. |
| Procedural exploration | Noisy observed-cell semantic-context proxies. |
| HouseExpo exploration | Official floor plans, occlusion-aware simulated sensing, and controlled semantic proxies. |

These component tests measure their stated geometry, memory, and exploration settings. Learned-perception task results are recorded separately in the [ManiSkill comparison](../study/README.md).

Extract an archive into a new directory to restore its `experiments/results/...` layout. The corresponding scripts are in the source archive's engine `experiments/` directory:

```text
rmp_rgbd_replay.py
rmp_threshold_sensitivity.py
rmp_policy_scaling.py
exploration_gridworld.py
exploration_weight_sensitivity.py
houseexpo_cross_room.py
```

Run `python -m dream_sim.sources` to locate the engine. Download helpers identify the upstream datasets and their attribution requirements. The compact archives contain CSV/JSON measurements; original public-dataset image arrays are obtained from their providers.
