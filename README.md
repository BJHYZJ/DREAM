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
python -m dream_sim.study --controller recovery_v6 \
  --task-manifest configs/residential50/task_manifest.json \
  --asset-dir .runtime/render_assets \
  --gpus 0 --output results/residential50 --execute
```

The study runs one cross-room task in each of **50 distinct residential scenes**, with **seed 42 throughout**. Each of five object/receptacle combinations appears in ten houses. All tasks use dynamic memory and the same controller. Add GPU IDs to run tasks concurrently, for example `--gpus 0 1 2 3`.

Each attempt saves observations, applied controls, physical trajectories, and its outcome. The [run guide](docs/reproduction.md) explains independent replay checks and complete-cohort analysis.

## Demonstrations and evaluation

The [simulation gallery](https://bjhyzj.github.io/dream-web/simulation/) shows complete cross-room task recordings with synchronized scene, robot-camera, semantic-memory, and navigation views. Videos play at 4× speed while retaining every recorded frame.

The [task manifest](configs/residential50/task_manifest.json) fixes the 50-house evaluation. Every outcome contributes to the reported completion rate. Earlier experiments and their source versions remain available in [`reproducibility/evidence/`](reproducibility/evidence/).

## Code structure

| Directory | Contents |
| --- | --- |
| `src/dream_sim/` | Task runner, resource preparation, replay evaluation, and video tools |
| `configs/` | Case catalog, task definitions, room maps, and dependency locks |
| `controllers/` | Controller revisions for common-protocol evaluation |
| `requirements/` | Python package versions |
| `reproducibility/source_archives/` | Versioned controller source, extracted automatically when used |
| `reproducibility/evidence/` | Experiment records, analysis, and video metadata |
| `tests/` | Configuration, source-integrity, and entrypoint checks |

To locate the controller for a case:

```bash
python -m dream_sim.sources --case 01
```

The returned directory contains `experiments/instruction_policy.py`, the navigation and manipulation helpers, and `src/dream/`. The [architecture guide](docs/architecture.md) maps these modules to the task pipeline.

## Validation

```bash
python -m dream_sim.run --preflight --dry-run
python -m dream_sim.verify_evidence
python -m pytest -q
```

These commands check configuration, source integrity, and stored records without running the simulator. See the [run guide](docs/reproduction.md#4-read-results) for evaluating a new execution.

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
