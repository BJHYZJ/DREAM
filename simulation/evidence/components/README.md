# Retained component measurements

These CSV/JSON records support the controlled component results in the reviewer
response. They were measured previously; they are not new physical trials or
part of the ten-profile delivery acceptance run. `manifest.json` gives archive
and uncompressed file hashes. No original public-dataset image arrays are
redistributed here.

- RMP replay: real public RGB-D, deliberately injected pose drift and supplied
  corrected poses; RGB vectors are feature proxies, not a VLM-accuracy test.
- RMP threshold/scaling/timing: isolated sensitivity and policy-cost records;
  timing excludes feature extraction and is not total high-level latency.
- Procedural exploration: noisy observed-cell semantic-context proxies.
- HouseExpo: official floor plans with occlusion-aware simulated sensing and a
  controlled semantic proxy, not a learned-perception or physical-robot trial.

Each archive restores `experiments/results/...` relative paths. Extract into a
new work directory for analysis, preserving the original archives. The helper
scripts are preserved under `simulation/_vendor/DREAM_code/experiments/`:
`rmp_rgbd_replay.py`, `rmp_threshold_sensitivity.py`, `rmp_policy_scaling.py`,
`exploration_gridworld.py`, `exploration_weight_sensitivity.py`, and
`houseexpo_cross_room.py`. Their download helpers identify the upstream data;
follow those providers' license/attribution requirements. Re-executing a
component test is separate from reproducing the learned ManiSkill videos.
