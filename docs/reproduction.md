# Running DREAM in ManiSkill

This guide covers environment setup, case execution, evaluation, and video export. Run commands from the root of the `simulation` branch checkout.

## 1. Install the environment

Use Linux with Python 3.11, CUDA inference, and a Vulkan renderer:

```bash
python3.11 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install -r requirements/requirements-learned.txt
python -m pip install --no-deps -e .
python -m pip check
```

The reference environment uses Python 3.11.15, ManiSkill 3.0.1, SAPIEN 3.0.3, PyTorch 2.10.0, CPU PhysX, Mesa Vulkan rendering, and NVIDIA H20 GPUs. A single task used approximately 10.5 GB of GPU memory on that configuration. Start with one worker and check resource use before adding more.

Install the CUDA and Vulkan runtime appropriate to your machine. For the Mesa software renderer used in the reference environment:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```

Set this variable to an installed, compatible ICD. Preflight checks its path; task execution also exercises the renderer. Changes in hardware or numerical libraries can affect the resulting trajectories.

## 2. Prepare models and scene assets

```bash
python -m dream_sim.prepare models \
  --cache-dir .runtime/models --production
python -m dream_sim.prepare assets \
  --reference-lock configs/locks/instruction_asset_lock03.json \
  --output-parent .runtime/assets/data/scene_datasets \
  --output-manifest .runtime/assets_download.json
python -m dream_sim.run --preflight
```

The model lock pins the SigLIP and OWL-V2 production models and their smaller compatibility variants. Use `--production` to download the production weights. The asset lock covers 1,339 files at upstream revision `1a173d5de042aaad8f1af09d4d2bc2ce4004b28a`.

Preflight checks package versions, model revisions and weight presence, scene-file hashes, and CUDA availability. Model weight contents do not have a separate SHA256 manifest. Downloads require network access; task execution uses the prepared model cache offline. Models and assets retain their upstream licenses.

To resume a partial asset download, pass its partial manifest with `--resume-manifest` in place of `--reference-lock`, and choose a new `--output-manifest`. The downloader rejects conflicting existing files.

For caches stored elsewhere, add these options to preflight and run commands:

```bash
--asset-dir /absolute/path/to/assets --model-cache /absolute/path/to/models
```

## 3. Run a task

```bash
python -m dream_sim.run --case 01 --output results/my_case01
python -m dream_sim.run --all --output results/my_ten_cases
```

Each command validates the selected sources and task configuration before launching one policy attempt per case. Results go into a new directory; an existing output directory is rejected. A later rerun uses another output path and the same case configuration.

The default is one GPU worker. Use `--gpus 0 1` for one worker on each device. Repeated device IDs allocate additional workers to that GPU. Cases retain a 1,200-second simulation budget and a 14,400-second policy wall timeout; perception, rendering, and recording make wall time longer than video playback.

| ID | House | Seed | Pickup → destination |
| --- | --- | ---: | --- |
| 01 | ProcTHOR-Train-283 | 100 | Mug → plate on wooden table |
| 02 | ProcTHOR-Train-8361 | 100 | Mug → plate on wooden table |
| 03 | ProcTHOR-Train-9031 | 100 | Egg → plate on wooden table |
| 04 | ProcTHOR-Train-5318 | 101 | Mug → plate on wooden table |
| 05 | ArchitecTHOR-Test-02 | 103 | Bread → plate on white table |
| 06 | ProcTHOR-Train-5080 | 102 | Egg → plate on wooden table |
| 07 | ProcTHOR-Test-244 | 102 | Egg → bowl on white table |
| 08 | ArchitecTHOR-Val-01 | 104 | Egg → bowl on white table |
| 09 | ProcTHOR-Val-632 | 105 | Tomato → bowl on white table |
| 10 | ProcTHOR-Train-7819 | 104 | Bread → plate on white table |

Video numbers match these case IDs. Full instructions, source versions, scene identifiers, and video checksums are in the [gallery catalog](../reproducibility/evidence/gallery/manifest.json). Train/Val/Test are upstream house split names; all cases use inference at runtime.

Six controller versions cover the gallery. Cases 09–10 enable the later heading-navigation adapter; other cases retain their recorded navigation settings. The [architecture guide](architecture.md) explains source selection and how to locate the policy code.

For configuration and integrity checks without loading the simulator:

```bash
python -m dream_sim.run --preflight --dry-run
python -m dream_sim.run --all --dry-run --output results/dry_plan
python -m pytest -q
```

### Evaluate a common controller

`dream_sim.study` runs one controller across the selected houses, seeds, and memory variants. The default design contains ten houses, seeds 100–102, and both dynamic and static memory, for 60 attempts:

```bash
python -m dream_sim.study --controller recovery \
  --gpus 0 1 --output results/recovery_study
```

This command validates the inputs and prints the plan. Add `--execute` to run it. Use `--controller baseline` for the controller from the recorded memory comparison, or restrict a development run with `--cases 03 04 --seeds 100 --variants dynamic`. External caches use the same `--asset-dir` and `--model-cache` options as the task runner.

The `recovery` controller uses 2.5 cm of additional footprint padding and retries an incomplete receptacle fit from higher head-camera views. When a bowl's central floor is occluded, inward-facing depth normals on its visible inner wall can establish cavity evidence. Detection, support grounding, payload clearance, release above the observed rim, and physical task scoring remain required. These changes are under evaluation; the published comparison describes the baseline controller.

Each study directory contains a frozen source tree, task definitions, `protocol.json`, individual run directories, and `attempts.jsonl` with every completed outcome. The final `batch_result.json` records whether all planned attempts were accounted for. Success rates use the complete declared set, including failures. Development runs on previously inspected cases do not estimate performance on new houses.

After the study finishes, replay successful tasks and compute the comparison:

```bash
python results/recovery_study/frozen_workspace/DREAM_code/experiments/audit_instruction_batch.py \
  --batch results/recovery_study --output results/recovery_audits \
  --workers 2 --all-task-successes
python -m dream_sim.study_report --run results/recovery_study \
  --audits results/recovery_audits --output results/recovery_summary
```

The report checks the declared inputs, all 60 outcomes, and successful-task replay records before writing counts and paired house-level uncertainty intervals. Add `--baseline-csv reproducibility/evidence/study/analysis/attempts.csv` to compare with the recorded baseline on matching houses and seeds.

## 4. Read results

The output root contains `planned.json`, `preflight.json`, `progress.json`, and a final `acceptance.json`. Each `case_XX/` contains:

| Path | Contents |
| --- | --- |
| `execution/` | Source snapshot, configuration, observations, detections, memory updates, actions, forces, trajectory, score, and original-speed video |
| `audit/physical/` | Physics replay of the saved controls and forces, with trajectory and contact comparisons |
| `audit/record/` | Checks for task order, target identity, relocation, rediscovery, grasp, transport, placement, and video continuity |
| `acceptance.json` | Per-case execution status, evaluation results, and checksums |
| `spectator/` | Additional presentation-camera render for cases 07 and 10 |

The final `all_passed` value combines execution and evaluation status for every requested case. Check `policy.log`, the case report, and the audit logs when a case fails. The runner records failed outcomes and does not retry them automatically.

The evaluators are selected by the version recorded with each case. Case 05 applies its documented evaluator-v4 correction while retaining the original score. Cases 07 and 10 render the same controls from a spectator view after passing evaluation. The [evaluation notes](../reproducibility/evidence/reproduction/README.md) describe the scoring versions and recorded results.

To re-evaluate complete recorded episodes:

```bash
python -m dream_sim.audit --execution results/my_ten_cases \
  --output results/my_matched_audits --workers 1
```

To render a spectator view:

```bash
python -m dream_sim.render --execution results/my_ten_cases \
  --case 07 --output results/my_case07_spectator
```

Add `--audits results/my_matched_audits` to use separately generated audit results. These two commands operate on recorded controls; a new policy attempt is launched through `dream_sim.run`.

## 5. Export a video

Use the composite video and its corresponding frame record:

```bash
python -m dream_sim.video --input PATH/TO/reviewer_view.mp4 \
  --frames PATH/TO/video_frames.json --output PATH/TO/reviewer_view_4x.mp4
```

The export keeps all frames at 1440×600. Source footage at 5 fps becomes 20 fps for 4× playback, with the simulation clock preserved and the speed label updated. Output uses H.264, yuv420p, and faststart for browser playback.

The gallery encodes case 01 with `--crf 18` and the remaining cases with `--crf 20`. Encoding uses local temporary storage before copying to the destination. The accompanying JSON records input/output hashes, frame counts, and sampled quality measurements.

## Implementation boundary

| Component | Simulation implementation |
| --- | --- |
| Instruction | One pickup/place command in bounded English, with optional support description |
| Observation | Current Fetch head-camera RGB-D and robot state |
| Perception | SigLIP features and OWL-V2 text-conditioned object detection |
| Memory | Observed semantic voxels, visual retrieval, and depth-based stale-location updates |
| Localization | Simulator odometry; the ROS SLAM backend is used in the real-robot implementation |
| Navigation | Observed occupancy, frontier exploration, and DREAM A* logic |
| Manipulation | RGB-D grasp/receptacle geometry and feedback-controlled motion templates; AnyGrasp is used in the real-robot system |
| Language-model verification | Hosted mLLM verification is disabled in this adapter |
| Evaluation | Environment object poses, room geometry, contacts, and recorded trajectories |

The environment initializes the objects and applies a force-driven relocation after verified visual discovery. The policy detects changes through its RGB-D observations. Destination search keeps the existing scene memory and grounds the requested receptacle in a fresh view. During navigation, the arm stays folded and sensing uses the head camera.

## Experiment records

- [Gallery](../reproducibility/evidence/gallery/README.md): ten selected demonstrations and their case/video mappings.
- [Memory comparison](../reproducibility/evidence/study/README.md): 60 attempts using one controller, ten houses, three seeds, and two memory variants.
- [Reproduction](../reproducibility/evidence/reproduction/README.md): repeat executions and evaluation records for the ten gallery cases.
- [Component measurements](../reproducibility/evidence/components/README.md): memory pruning, scaling, and exploration analyses.
- [Entrypoint validation](../reproducibility/evidence/packaging/README.md): source-integrity checks and a case-01 smoke run.

The gallery is a selection of demonstrations. Comparative results use the separate study protocol and include all its attempts. Compact archives contain control, trajectory, and evaluation records; large raw sensor arrays and videos are listed by hash and retained in the authors' archive.
