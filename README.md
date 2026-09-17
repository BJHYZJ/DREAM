# DREAM

**Dynamic Resilient Spatio-Semantic Memory with Hybrid Localization for Mobile Manipulation**

[Project page](https://bjhyzj.github.io/dream-web/) · [Paper](https://arxiv.org/abs/2606.00576) · [Simulation demos](https://bjhyzj.github.io/dream-web/simulation/) · [Real-robot code](https://github.com/BJHYZJ/DREAM/tree/realtime)

DREAM starts from a language instruction, explores an unfamiliar environment, builds and updates semantic memory, and performs object pickup and placement. This `simulation` branch provides the **Fetch / ManiSkill / SAPIEN** implementation. The ROS system, SLAM backend, hardware configuration, and AnyGrasp integration are on the [`realtime` branch](https://github.com/BJHYZJ/DREAM/tree/realtime).

![DREAM system overview](https://bjhyzj.github.io/dream-web/media/figures/main.png)

## What to reproduce

| Experiment | Configuration | Published result |
| --- | --- | --- |
| Current residential evaluation and all 50 videos | `continuous_return`, 50 fixed houses, seed 42, dynamic memory | **38/50 (76%)**, including all 12 failures |
| Paired memory comparison | `recovery_v2`, 10 houses × 3 seeds × 2 memory variants | Dynamic 22/30; static 14/30 |
| Extended search | Four selected cases with action/iteration limits removed | One qualified completion; separate from 38/50 |

The current residential tasks use native-scale objects, a **1800-second robot-action budget**, and **no fixed server deadline**. These houses were used during controller development. See the [experiment index](reproducibility/README.md) for protocols, evidence contents, and the earlier comparison cohorts.

## Quick start

Use Linux, Python 3.11, an NVIDIA GPU for inference, and a working Vulkan renderer. The reference system used about 10.5 GB of GPU memory per task. Start with one worker. The batch runner requires **at least 128 GiB free on the recording filesystem before each task**, in addition to space for downloaded assets and models. See the [complete setup and run guide](docs/reproduction.md) for rendering, resource allocation, and troubleshooting.

```bash
git clone --branch simulation https://github.com/BJHYZJ/DREAM.git
cd DREAM
python3.11 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install -r requirements/requirements-learned.txt
python -m pip install --no-deps -e .
python -m pip check
```

Create the local worker configuration on a **new checkout**. Replace `0` with a physical GPU index allocated to you. The command uses available CPU cores and refuses to overwrite an existing allocation.

```bash
python -m dream_sim.configure --gpus 0 --workers-per-gpu 1 --cpu-threads 2
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

Inspect the current 50-task plan, then append `--execute` to run it:

```bash
DREAM_PYTHON="$(command -v python)" ./scripts/run_residential.sh \
  --gpus 0 --workers-per-gpu 1 --raster-threads 2 \
  --output results/residential50
```

Use a fresh output directory for each execution. Models and assets default to the prepared `.runtime` directories. GPU indices and worker counts must match your local configuration. One worker runs the same 50 tasks sequentially; adding workers changes concurrency, not the task definitions or action budget.

After execution, independently review successful tasks and calculate the result over **all 50 attempts**:

```bash
python -m dream_sim.study_review --run results/residential50 \
  --output results/residential50_audits --execute
python -m dream_sim.study_report --run results/residential50 \
  --audits results/residential50_audits --output results/residential50_summary \
  --include-contact-rejections
```

`results/residential50_summary/study_analysis.json` and `attempts.csv` contain the new results. Completion requires the task outcome and independent physics, observation, contact, and both arm-return checks. A new execution is not guaranteed to reproduce the original 38 successes on different hardware.

## Inspect the published evidence without running the simulator

```bash
python -m dream_sim.evaluate
python -m dream_sim.verify_evidence
```

`evaluate` checks the **current 38/50 records** and reconstructs the exact evaluated controller. `verify_evidence` also checks the paired comparison, component measurements, and extended-search archive. These are offline checks of saved files; they do not perform a new policy run or physics replay. Use `evaluate --cohort compact` or `verify_evidence --include-historical` for the earlier cohorts.

The repository contains compact outcomes, protocols, checksums, and review records. **It does not contain the full original RGB-D recordings.** To replay or export a video locally, first run the tasks to generate complete recordings, or obtain a complete original recording separately. Hash lists and compact evidence ZIPs cannot replace the omitted sensor data. [Videos and replay commands](docs/reproduction.md#5-export-a-current-trial-video).

## Implementation and source layout

| Directory | Contents |
| --- | --- |
| `src/dream_sim/` | Preparation, worker allocation, execution, result checks, and video export |
| `controllers/continuous_return/` | Current controller overrides; inherits the staged and compact controllers |
| `configs/residential50-easy-grasp/` | Current 50 tasks, room maps, and fixed input hashes |
| `requirements/` | Pinned Python dependencies |
| `reproducibility/source_archives/` | Required base source, verified and extracted by the loader |
| `reproducibility/evidence/` | Published experiment outcomes and supporting comparison records |
| `tests/` | Regression tests for execution, scheduling, and result integrity |

The simulator supplies odometry. Perception uses SigLIP and OWL-V2; manipulation uses observed RGB-D geometry and feedback-controlled motion. Hosted mLLM verification is disabled in this adapter. See the [implementation boundary](docs/reproduction.md#implementation-boundary) and [module map](docs/architecture.md).

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

DREAM code uses the [MIT License](LICENSE). SigLIP and OWL-V2 weights and AI2-THOR scene assets are downloaded separately under their upstream licenses. Versions are pinned in the [model lock](configs/locks/dream_models.lock.json) and [asset lock](configs/residential50/assets.lock.json).
