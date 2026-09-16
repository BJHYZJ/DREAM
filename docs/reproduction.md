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

The reference environment uses Python 3.11.15, ManiSkill 3.0.1, SAPIEN 3.0.3, PyTorch 2.10.0, CPU PhysX, Mesa Vulkan rendering, and NVIDIA H20 GPUs. A single task used approximately 10.5 GB of GPU memory on that configuration. Start with one worker and check resource use before adding more. A snapshot of 18 concurrent tasks showed about 5–8.2 GiB of resident host memory per task; this is not a peak-memory bound. Saved RGB-D frames, controls, and videos require additional storage. Use a disk-backed output directory for routine runs: placing recordings in `/dev/shm` charges their full size to host memory until they are archived and removed.

Install the CUDA and Vulkan runtime appropriate to your machine. For the Mesa software renderer used in the reference environment:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
```

Set this variable to an installed, compatible ICD. Preflight checks its path; task execution also exercises the renderer. Changes in hardware or numerical libraries can affect the resulting trajectories.

## 2. Prepare models and scene assets

```bash
python -m dream_sim.prepare models \
  --cache-dir .runtime/models --production
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

The rendering cache supplies the missing UV0 index in `Desk_Lamp_11.glb` using its existing UV1 coordinates. It preserves the original files, materials, vertex positions, and binary buffers. The [recorded rendering lock](../configs/residential50/render_assets.lock.json) contains both source and derived hashes.

The model lock pins the SigLIP and OWL-V2 production models and their smaller compatibility variants. Use `--production` to download the production weights. The residential asset lock covers 2,415 files at upstream revision `1a173d5de042aaad8f1af09d4d2bc2ce4004b28a`.

Preflight checks package versions, model revisions and weight presence, scene-file hashes, and CUDA availability. Model weight contents do not have a separate SHA256 manifest. Downloads require network access; task execution uses the prepared model cache offline. DREAM source code uses the MIT License. SigLIP and OWL-V2 model weights and AI2-THOR scene assets retain the licenses published with those resources.

To resume a partial asset download, pass its partial manifest with `--resume-manifest` in place of `--reference-lock`, and choose a new `--output-manifest`. The downloader rejects conflicting existing files.

For caches stored elsewhere, add these options to preflight and run commands:

```bash
--asset-dir /absolute/path/to/assets --model-cache /absolute/path/to/models
```

## 3. Run the residential study

### Easy-grasp protocol

The DREAM Fetch controller completes **36/50 easy-grasp tasks (72%)**. All 36 successes pass independent physics and observation checks, including arm returns. All 50 scenes use the original fixed tasks and maps, seed 42, dynamic memory, and native object scale. The cohort contains 21 models across 6 categories; see the [task definition](../configs/residential50-easy-grasp/README.md).

```bash
DREAM_PYTHON="$(command -v python)" \
DREAM_ASSET_DIR="$PWD/.runtime/render_assets" \
DREAM_MODEL_CACHE="$PWD/.runtime/models" \
./scripts/run_residential.sh \
  --gpus 0 1 --workers-per-gpu 4 --raster-threads 4 \
  --output results/residential50
```

This prints the plan; add `--execute` to run it. Each task receives **900 seconds of robot actions** and a separate **2700-second execution watchdog**. The example selects GPUs 0 and 1 with four workers per GPU. The deployment resource profile must assign the corresponding worker slots, CPU cores, and memory limits. Queue time is separate from execution time. Use a new output directory for every run. The three environment variables above select your installed Python and prepared caches; omitting them uses `python3.11` and the repository-local `.runtime/render_assets` and `.runtime/models` caches.

The [sealed all-50 report](../reproducibility/evidence/residential-evaluation/results.json) retains every failure and records source and per-attempt evidence hashes. The offline evaluation command checks that the official loader reconstructs the evaluated Python source from its recorded base and override modules. Video qualification is separate; the existing gallery belongs to the historical diverse-object experiment.

The [experiment directory](../reproducibility/evidence/residential-evaluation/README.md) contains all task outcomes and independent review records; full sensor arrays remain in local experiment storage. To check its 36/50 result and confirm that the installed controller matches the frozen evaluated source, run `python -m dream_sim.evaluate`. This checks the recorded results and reconstructs the controller from the checksum-locked source archive.

### Diverse-object protocol


The earlier diverse-object configuration contains **50 distinct houses**, one cross-room pickup/place task per house, and **seed 42 for every task**. Each house uses a different pickup model: **50 native-scale models across 20 categories**, including one mug. Every instruction requests placement on a plate. All tasks use dynamic memory and the same `compact_v1` controller. The task manifest fixes the houses, instructions, object layouts, initial-state seed, and file checksums. Using the current controller starts a new experiment with that configuration.

```bash
python -m dream_sim.study --controller compact_v1 \
  --task-manifest configs/residential50-diverse/task_manifest.json \
  --asset-dir .runtime/render_assets \
  --gpus 0 1 --output results/residential50
```

This validates the inputs and prints the execution plan. Add `--execute` to run the study. Use `--gpus 0` for one standard worker or `--gpus 0 1` for two. This deployment accepts GPU IDs 0 and 1; repeated IDs are rejected. The parallel batch wrapper above sets multiple workers per GPU explicitly. Use a new output directory for each study. The manifest cannot be combined with overrides of cases, seeds, or memory variants.

Each task has a 1,800-second simulation budget and a 14,400-second policy wall timeout. RGB-D perception, rendering, and recording add wall time. The runner records each outcome once and does not retry failed tasks automatically.

For external caches, add `--asset-dir /absolute/path/to/assets --model-cache /absolute/path/to/models`.

### Robot control

The controller selects exploration candidates reachable through heading-dependent, swept-footprint motion edges, plans through observed free space, verifies the requested receptacle from RGB-D geometry, and uses feedback during grasping and release. The torso follows a smooth position reference shared across arm control modes. Its reference speed and acceleration are limited to 0.04 m/s and 0.10 m/s². Physical joint motion is measured separately.

Initialization and arm returns use a compact, torso-facing arm posture. The joint target is `[-1.253857, 1.35, 0.4, 1.9, -0.013637, 1.200051, 0.000009]`, ordered as the seven Fetch arm joints. Shoulder/elbow flexion and upper-arm roll draw the arm inward while maintaining torso clearance. After grasping or release, the robot lifts clear of the support, folds with the base stationary, and checks measured tracking and settling. Fold/unfold joint references are limited to 0.9 rad/s; physical tracking is checked separately. Before extending again, the robot partly turns the shoulder forward and lifts, then extends forward. Camera-height adjustments preserve folded arm targets. Independent replay checks evaluate the measured arm motion and clearance.

The controller disables optional held-arm extensions for receptacle visibility and uses head and base views during that search. The optional extension method remains available and retraces its measured joint path before navigation resumes, checking payload retention, joint tracking, and base displacement. Search progress also accounts for the remaining length of a route ending near the observed target, so a necessary detour is not abandoned solely because straight-line distance increases. Replanning without measured base movement and routes to unrelated exploration frontiers do not reset the stalled-search budget.

Before grasping, the robot aligns the open gripper above the observed target and approaches vertically. It checks position and orientation before closing the gripper. Placement uses the observed object-center offset relative to the gripper, so the commanded tool pose and release-height check remain consistent for grasps above the object center. If release alignment stalls, the controller makes one bounded torso-height adjustment and rechecks the observed object center before opening the gripper.

### Evaluate the recordings

After execution finishes:

```bash
python results/residential50/frozen_workspace/DREAM_code/experiments/audit_instruction_batch.py \
  --batch results/residential50 --output results/residential50_audits \
  --workers 2 --all-task-successes
python -m dream_sim.study_report --run results/residential50 \
  --audits results/residential50_audits --output results/residential50_summary \
  --include-contact-rejections
```

The report verifies all 50 outcomes against their fixed inputs and checks independent physical replays of successful tasks. It writes `study_analysis.json` and `attempts.csv`. Task completion requires correct visual observation of the moved object, sustained physical grasp and lift, cross-room transport, and stable release in the requested receptacle. Continuous correct tracking is permitted; losing and rediscovering the object is recorded as a separate behavior.

`task_success` is the evaluator outcome. `qualified_task_success` additionally requires independent physics and recording checks, recorded compact-arm returns after grasp and release, and collision checks for those motions. Robot self contacts are recorded at every native physics substep. A one-second joint-position window verifies that the returned arm has settled; native joint velocities are retained separately. The completion rate uses all 50 tasks, including failures. `--include-contact-rejections` counts tasks rejected solely for native-environment contact as unsuccessful qualified outcomes while preserving their original scores. Missing audits, changed sources, or other failed checks stop reporting.

## 4. Read results

| Path within the study | Contents |
| --- | --- |
| `protocol.json` | Fixed tasks, seeds, controller sources, and checksums |
| `frozen_workspace/`, `frozen_tasks/` | Source snapshot and task inputs |
| `<attempt>/result.json` | Outcome, criteria, duration, and any execution error |
| `<attempt>/events.jsonl` | Perception, memory, navigation, and manipulation events |
| `<attempt>/actions.json` | Applied controls and measured arm/torso state |
| `<attempt>/evaluator_trajectory.jsonl` | Physical task trajectory |
| `attempts.jsonl`, `batch_result.json` | Every completed outcome and study completion status |

The audit directory contains a physical replay, video/source checks, and a status report for each successful task. Inspect the attempt's policy log and audit logs when a check fails. Replaying saved controls does not launch another policy attempt.

## 5. Export a video

Use the composite video and its corresponding frame record:

```bash
python -m dream_sim.video --input PATH/TO/reviewer_view.mp4 \
  --frames PATH/TO/video_frames.json --output PATH/TO/reviewer_view_4x.mp4
```

The export retains all frames at 1440×600. Source footage at 5 fps becomes 20 fps for 4× playback, with the simulation clock preserved and the speed label updated. Output uses H.264, yuv420p, and faststart for browser playback. The gallery's [video metadata](https://github.com/BJHYZJ/dream-web/blob/master/simulation/results.json) records source hashes and encoding settings.

### Earlier experiments

The original ten demonstrations remain reproducible through `python -m dream_sim.run --case 01 --output results/original_case01`. Their configurations and source versions are in [`configs/cases.json`](../configs/cases.json). The earlier common-controller studies are stored under [`reproducibility/evidence/`](../reproducibility/evidence/). They use their own task manifests and remain separate from the diverse-object study. The earlier five-recipe, 50-house configuration remains in `configs/residential50/`.

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

Small-object search also uses overlapping calibrated RGB-D crops. Memory retrieval verifies up to four semantically ranked observations and caches inference for each saved image and query. Cached detections remain subject to current stale-location rejection rules.

The environment initializes the objects and applies a force-driven relocation after verified visual discovery. The policy detects changes through its RGB-D observations. Destination search keeps the existing scene memory and grounds the requested receptacle in a fresh view. During navigation, the arm stays folded and sensing uses the head camera. The video’s left panel is a simulator overview; the lower-right panel is a depth-derived navigation map with semantic coloring. Robot self-filtering applies to that navigation map using joint poses and robot geometry. The held-object bound is reconstructed from observed dimensions and the recorded grasp frame, then transformed with the measured gripper pose. It includes a 2 cm uncertainty margin. The navigation self-filter accounts for the held geometry with a uniform 2.5 cm margin. These filters do not mask the raw camera images. The semantic-memory integrator receives the RGB-D observations without this navigation self-filter.

## Experiment records

The [residential task manifest](../configs/residential50-diverse/task_manifest.json) contains the complete 50-house design. Experiment records include every task outcome, controller and asset hashes, and independent replay checks. Compact archives contain control, trajectory, and evaluation records; large raw sensor arrays are retained in the authors' archive.

The [component measurements](../reproducibility/evidence/components/README.md) cover memory pruning, scaling, and exploration. The [architecture guide](architecture.md) maps the implementation to the perception, memory, navigation, and manipulation pipeline.

## Worker resources

Parallel execution reads `.runtime/worker_resource_limits.json`, a machine-specific deployment profile. It must agree with `--gpus`, `--workers-per-gpu`, and the available CPU and memory allocation. This local file is excluded from Git because its CPU IDs, cgroup paths, and GPU assignment belong to the host machine. The batch runner and `worker_limits.py` validate that configuration before task execution. Keep your deployment's existing allocation when using this repository on a shared server.

To inspect the task list without launching workers, omit `--execute`. The standard `study` mode uses one worker per requested GPU; `--parallel` selects the bounded 900-second residential batch with explicit worker slots.
