# DREAM

**Dynamic Resilient Spatio-Semantic Memory with Hybrid Localization for Mobile Manipulation**

[Project page](https://bjhyzj.github.io/dream-web/) · [Paper](https://arxiv.org/abs/2606.00576) · [Videos](https://bjhyzj.github.io/dream-web/simulation/) · [Real-robot code](https://github.com/BJHYZJ/DREAM/tree/realtime)

DREAM is a mobile manipulation framework for previously unseen indoor environments. Starting from a language instruction, the robot explores its surroundings, builds a 3D semantic memory, finds task objects, and carries out grasping and placement. It uses new observations to update remembered locations when objects move.

The full system combines dynamic spatio-semantic memory, Redundancy-Aware Memory Pruning, multi-sensor SLAM, hybrid target localization, and task-oriented navigation. The ROS implementation and hardware setup are on the [`realtime` branch](https://github.com/BJHYZJ/DREAM/tree/realtime).

![DREAM system overview](https://bjhyzj.github.io/dream-web/media/figures/main.png)

## Indoor simulation

This branch runs cross-room pick-and-place tasks in **ManiSkill / SAPIEN** with a Fetch robot. The robot searches for an object, observes a change in its location, finds it again, and delivers it to the requested receptacle.

- **Perception:** SigLIP visual-language features and OWL-V2 object detection from head-camera RGB-D observations.
- **Memory:** semantic voxel mapping, retrieval of task-relevant observations, and depth-based updates to stale object locations.
- **Navigation:** observed occupancy mapping, frontier exploration, and A* route planning.
- **Manipulation:** grasp and placement poses computed from observed geometry, executed with robot feedback.

The simulator supplies odometry, and task instructions use a bounded English pickup/place grammar. See the [implementation details](docs/reproduction.md#implementation-boundary) for the simulation interfaces and their relation to the real-robot system.

## Getting started

Use Linux with **Python 3.11**, CUDA inference, and a working Vulkan renderer. The [setup guide](docs/reproduction.md) covers dependencies, model downloads, scene assets, and resource requirements.

```bash
git clone --branch simulation https://github.com/BJHYZJ/DREAM.git
cd DREAM
python3.11 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install -r requirements/requirements-learned.txt
python -m pip install --no-deps -e .
python -m pip check
```

Prepare the models and scene assets:

```bash
python -m dream_sim.prepare models --cache-dir .runtime/models --production
python -m dream_sim.prepare assets \
  --reference-lock configs/residential50/assets.lock.json \
  --output-parent .runtime/assets/data/scene_datasets \
  --output-manifest .runtime/assets_download.json
python -m dream_sim.render_assets \
  --source-dir .runtime/assets --output-dir .runtime/render_assets \
  --reference-lock configs/residential50/assets.lock.json \
  --output-lock .runtime/render_assets.lock.json
python -m dream_sim.run --preflight --asset-dir .runtime/render_assets \
  --asset-lock configs/residential50/render_assets.lock.json
```

Run the residential study:

```bash
DREAM_PYTHON="$(command -v python)" \
DREAM_ASSET_DIR="$PWD/.runtime/render_assets" \
DREAM_MODEL_CACHE="$PWD/.runtime/models" \
./scripts/run_residential.sh \
  --gpus 0 1 --workers-per-gpu 4 --raster-threads 4 \
  --output results/residential50 --execute
```

The default `continuous_return` controller passes task completion and independent physics, observation, and arm-return checks in **38/50 scenes (76%)**. All 50 tasks ran with seed 42, dynamic memory, native object scale, an **1800-second robot-action budget**, and **no fixed server deadline**. The easy-grasp cohort uses 21 pickup models across 6 categories. These houses were used for controller development, so the rate describes this fixed cohort. The [portable evidence](reproducibility/evidence/residential-fast-return/) includes all outcomes, original review records, failure analysis, and paired return times. Full local records are in `results/residential-adjusted/`. The [historical compact-controller result](reproducibility/evidence/residential-evaluation/results.json) remains 36/50 (72%) at 900 action seconds with a 2700-second server watchdog. The videos below belong to a separate recorded cohort.

The command uses GPUs 0 and 1 with four workers per GPU; the deployment resource profile must support this allocation. Omit `--execute` to inspect the plan. See the [run guide](docs/reproduction.md#3-run-the-residential-study) for cache paths and the distinct study protocols.

Each attempt saves observations, applied controls, physical trajectories, and its outcome. The [run guide](docs/reproduction.md) explains independent replay checks and complete-cohort analysis.

The [arm-return controller](controllers/continuous_return/README.md) lifts clear of the support and follows measured intermediate postures before confirming the final fold. Independent replay checks both loaded and empty returns, including self-contact and fixture contact. The [return-time comparison](reproducibility/evidence/residential-fast-return/return_times.json) reports the shared successful tasks and timing definitions for the staged and continuous controllers. Historical configurations and their original budgets are documented in the [run guide](docs/reproduction.md).

## Demonstrations and evaluation

In the historical diverse-object residential study, DREAM completed **27/50 tasks (54%)**. The [video gallery](https://bjhyzj.github.io/dream-web/simulation/) includes **all 27 successful runs** and **2 failure cases**, with synchronized scene, robot-camera, semantic-memory, and navigation views. Videos retain the complete recorded sequence at 4× speed. Those historical video checks did not record robot self-contact or require a verified final fold, so their completion labels do not establish collision-free arm retraction.

The [experiment records](reproducibility/evidence/residential50-seed42/README.md) contain all 50 outcomes, task configurations, and independent physics and recording checks. Every attempt contributes to the completion rate. Earlier experiments retain their own records in [`reproducibility/evidence/`](reproducibility/evidence/).

The [extended-search case study](reproducibility/evidence/long-search/README.md) reports all four follow-up outcomes separately from the main cohort, including the 78.7-minute successful task.

## Code structure

| Directory | Contents |
| --- | --- |
| `src/dream_sim/` | Task and batch execution, resource preparation, result evaluation, and video tools |
| `configs/` | Case catalog, task definitions, room maps, and dependency locks |
| `controllers/` | Controller modules for residential evaluation |
| `requirements/` | Python package versions |
| `reproducibility/source_archives/` | Checksum-locked controller source, extracted automatically when used |
| `reproducibility/evidence/` | Experiment records, analysis, and video metadata |
| `tests/` | Runner, configuration, resource-allocation, and result-integrity tests |

To locate the controller for a case:

```bash
python -m dream_sim.sources --case 01
```

The returned directory contains `experiments/instruction_policy.py`, the navigation and manipulation helpers, and `src/dream/`. The [architecture guide](docs/architecture.md) maps these modules to the task pipeline.

## Validation

```bash
python -m dream_sim.run --preflight --dry-run
python -m dream_sim.verify_evidence
python -m dream_sim.evaluate
python -m pytest -q
```

These commands check configuration, source integrity, and stored records without running the simulator. `dream_sim.evaluate` checks the historical compact-controller result; the [current evidence guide](reproducibility/evidence/residential-fast-return/#verify-and-reproduce) provides the 76% archive integrity check. See the [run guide](docs/reproduction.md#4-read-results) for evaluating a new execution.

## Citation

```bibtex
@misc{yan2026dynamicresilientspatiosemanticmemory,
  title={Dynamic Resilient Spatio-Semantic Memory with Hybrid Localization for Mobile Manipulation},
  author={Zhijie Yan and Shufei Li and Ze Zhang and Xin Liu and Yuhang Zheng and Zuoxu Wang},
  year={2026},
  eprint={2606.00576},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2606.00576},
}
```

## License

The DREAM code is released under the [MIT License](LICENSE). SigLIP and OWL-V2 weights, and AI2-THOR scene assets, are obtained separately under the licenses of those projects. Versions and checksums are listed in the [model and asset records](configs/locks/).
