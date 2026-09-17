# Running DREAM in ManiSkill

This guide covers environment setup, case execution, evaluation, and video export. Run commands from the root of the `simulation` branch checkout.

## Choose the experiment

The commands in Sections 1–5 reproduce the current **50-house, 38/50** experiment using `continuous_return`. The [paired dynamic/static comparison](#evaluate-a-common-controller) and [extended search](#extended-search-video) are separate experiments. The [archive index](../reproducibility/README.md) explains which records support each result.

A clean checkout contains the source, fixed tasks, and compact evidence. It does **not** contain the authors' original RGB-D recordings. Installation and a new policy run produce complete local recordings; unpacking an evidence ZIP alone is insufficient for physics replay or video export.

## 1. Install the environment

Use Linux with Python 3.11, CUDA inference, and a Vulkan renderer:

```bash
python3.11 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install -r requirements/requirements-learned.txt
python -m pip install --no-deps -e .
python -m pip check
```

The reference environment uses Python 3.11.15, ManiSkill 3.0.1, SAPIEN 3.0.3, PyTorch 2.10.0, CPU PhysX, Mesa Vulkan rendering, and NVIDIA H20 GPUs. A single task used approximately 10.5 GB of GPU memory on that configuration. Start with one worker and check resource use before adding more. A snapshot of 18 concurrent tasks showed about 5–8.2 GiB of resident host memory per task; this is not a peak-memory bound. The batch runner requires at least **128 GiB free on the output filesystem before each task**; reserve additional space for model and scene downloads. Saved RGB-D frames, controls, and videos require additional storage. Use a disk-backed output directory for routine runs: placing recordings in `/dev/shm` charges their full size to host memory until they are archived and removed.

Install the CUDA and Vulkan runtime appropriate to your machine. For the Mesa software renderer used in the reference environment:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
```

Set this variable to an installed, compatible ICD. Preflight checks its path; task execution also exercises the renderer. Changes in hardware or numerical libraries can affect the resulting trajectories.

Before the first run, select a physical GPU allocated to you (`nvidia-smi -L` lists device indices) and create the machine-local configuration:

```bash
python -m dream_sim.configure --gpus 0 --workers-per-gpu 1 --cpu-threads 2
```

The command assigns distinct cores from the process's available CPU affinity and refuses to overwrite an existing configuration. It does not install a RAM or GPU-memory cap. On a shared machine, retain the existing allocation. See [Worker resources](#worker-resources) for multiple workers.

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

The published run uses `continuous_return`, all 50 tasks in `configs/residential50-easy-grasp/task_manifest.json`, seed 42, dynamic memory, native object scale, an **1800-second action budget**, and **no fixed server deadline**. Its 21 pickup models span six categories. The houses were used during controller development.

Inspect the plan first:

```bash
DREAM_PYTHON="$(command -v python)" ./scripts/run_residential.sh \
  --gpus 0 --workers-per-gpu 1 --raster-threads 2 \
  --output results/residential50
```

Add `--execute` to the same command to start. Without it, the command checks inputs and prints the plan; it does not run a task. The one-worker configuration runs the same 50 tasks sequentially. Use a new output directory for each experiment. All terminal outcomes, including failures, remain in the denominator.

The wrapper selects the current controller and budgets explicitly. Direct `dream_sim.study` calls have older defaults, so the equivalent current command is:

```bash
python -m dream_sim.study --controller continuous_return \
  --task-manifest configs/residential50-easy-grasp/task_manifest.json \
  --asset-dir .runtime/render_assets --model-cache .runtime/models \
  --parallel --gpus 0 --workers-per-gpu 1 --raster-threads 2 \
  --robot-time-limit-seconds 1800 --wall-timeout-seconds 0 \
  --wait-for-slot-seconds 1200 --output results/residential50 --execute
```

Robot-action time counts executed controls. It excludes inference, loading, saving, and queue waits. `--wall-timeout-seconds 0` disables the separate server deadline. The worker wait limit applies before execution begins. Navigation iteration and stagnation rules still apply. Changed budgets or controllers define a new experiment, not a rescore of the original recordings.

For external caches, set `DREAM_ASSET_DIR` and `DREAM_MODEL_CACHE` for the shell wrapper, or pass `--asset-dir` and `--model-cache` to the Python entry point. The default caches are `.runtime/render_assets` and `.runtime/models`.

The current [results](../reproducibility/evidence/residential-fast-return/results.json) contain every outcome. Run `python -m dream_sim.evaluate` to check those records and the exact evaluated source offline. This does not launch a new simulation. To check the earlier 36/50 compact-controller archive, use `python -m dream_sim.evaluate --cohort compact`.

### Historical residential protocols

| Controller | Tasks | Action / server limits | Published outcome |
| --- | --- | --- | --- |
| `continuous_return` | Easy-grasp 50 | 1800 s / disabled | 38/50 |
| `staged_return` | Easy-grasp 50 | 900 s / 2700 s | 32/50 |
| `compact_v1` | Easy-grasp 50 | 900 s / 2700 s | 36/50 |

To run either historical easy-grasp protocol, set `DREAM_CONTROLLER` to that controller and append `--robot-time-limit-seconds 900 --wall-timeout-seconds 2700` to the wrapper command, with a new output directory. Command-line options override the wrapper defaults. The separate `configs/residential50-diverse/` configuration uses 50 distinct object models; it is not the task set behind the current gallery.

### Robot control

The controller selects exploration candidates reachable through heading-dependent, swept-footprint motion edges, plans through observed free space, verifies the requested receptacle from RGB-D geometry, and uses feedback during grasping and release. The torso follows a smooth position reference shared across arm control modes. Its reference speed and acceleration are limited to 0.04 m/s and 0.10 m/s². Physical joint motion is measured separately.

Initialization and arm returns use a compact, torso-facing arm posture. The joint target is `[-1.253857, 1.35, 0.4, 1.9, -0.013637, 1.200051, 0.000009]`, ordered as the seven Fetch arm joints. Shoulder/elbow flexion and upper-arm roll draw the arm inward while maintaining torso clearance. After grasping or release, the robot lifts clear of the support, folds with the base stationary, and checks measured tracking and settling. The selected controller sets the return speed and acceleration bounds. `continuous_return` uses 0.85 rad/s and 1.5 rad/s² while holding an object, and 1.1 rad/s and 2.0 rad/s² with an empty gripper; physical tracking is checked separately. Before extending again, the robot partly turns the shoulder forward and lifts, then extends forward. Camera-height adjustments preserve folded arm targets. Independent replay checks evaluate the measured arm motion and clearance.

The controller disables optional held-arm extensions for receptacle visibility and uses head and base views during that search. The optional extension method remains available and retraces its measured joint path before navigation resumes, checking payload retention, joint tracking, and base displacement. Search progress also accounts for the remaining length of a route ending near the observed target, so a necessary detour is not abandoned solely because straight-line distance increases. Replanning without measured base movement and routes to unrelated exploration frontiers do not reset the stalled-search budget.

Before grasping, the robot aligns the open gripper above the observed target and approaches vertically. It checks position and orientation before closing the gripper. Placement uses the observed object-center offset relative to the gripper, so the commanded tool pose and release-height check remain consistent for grasps above the object center. If release alignment stalls, the controller makes one bounded torso-height adjustment and rechecks the observed object center before opening the gripper.

### Evaluate the recordings

After all 50 tasks finish, review the corrected `staged_return` or `continuous_return` recordings:

```bash
python -m dream_sim.study_review --run results/residential50 \
  --output results/residential50_audits --execute
python -m dream_sim.study_report --run results/residential50 \
  --audits results/residential50_audits --output results/residential50_summary \
  --include-contact-rejections
```

`study_review` reads the parallel runner's `screen_results` and `screen_complete.json`, binds every task to its frozen inputs, and reviews successful tasks one at a time in separate CPU processes. Omit `--execute` to inspect the review plan. If exact physical replays already exist, add `--physical-audits /path/to/audits`; their source and input checks still apply, and the saved observations are checked again. Review output directories must be new. Video export is separate from these task and arm-return checks.

The report verifies all 50 outcomes against their fixed inputs and checks independent physical replays of successful tasks. It writes `study_analysis.json` and `attempts.csv`. Task completion requires correct visual observation of the moved object, sustained physical grasp and lift, cross-room transport, and stable release in the requested receptacle. Continuous correct tracking is permitted; losing and rediscovering the object is recorded as a separate behavior.

`task_success` is the evaluator outcome. `qualified_task_success` additionally requires independent physics and recording checks, recorded compact-arm returns after grasp and release, and collision checks for those motions. Robot self contacts are recorded at every native physics substep. The staged-return reviewer checks the position and native-velocity settling windows. The derived `continuous_return` controller also records positions at every native physics step; after a final-stow native-velocity timeout, its explicit displacement-based check must be independently replayed before acceptance. Earlier cohorts retain their original measurement method and scores. The completion rate uses all 50 tasks, including failures. Valid review records that reject task completion or arm return count as unsuccessful qualified outcomes. `--include-contact-rejections` also retains native-environment contact rejections as unsuccessful outcomes while preserving their original scores. Missing audits, changed sources, or failed recording-integrity checks stop reporting.

## 4. Read results

| Path within the study | Contents |
| --- | --- |
| `protocol.json` | Fixed tasks, seeds, controller sources, and checksums |
| `frozen_workspace/`, `frozen_tasks/` | Source snapshot and task inputs |
| `<attempt>/result.json` | Outcome, criteria, duration, and any execution error |
| `<attempt>/events.jsonl` | Perception, memory, navigation, and manipulation events |
| `<attempt>/actions.json` | Applied controls and measured arm/torso state |
| `<attempt>/evaluator_trajectory.jsonl` | Physical task trajectory |
| `screen_results/`, `screen_complete.json` | Every completed outcome and completion status for parallel execution |
| `attempts.jsonl`, `batch_result.json` | Equivalent outcome records for the serial study runner |

The audit directory contains a physical replay, video/source checks, and a status report for each successful task. Inspect the attempt's policy log and audit logs when a check fails. Replaying saved controls does not launch another policy attempt.

## 5. Export a current trial video

Use the complete run you created in Section 3, including its frozen workspace and task records. Keep the separate independent reviews from Section 3 as well. The compact published evidence ZIP is not a replacement for this directory:

```bash
python -m dream_sim.render_cohort_trial \
  --run results/residential50 --audits results/residential50_audits \
  --name 01_ProcTHOR-Test-722_seed42_dynamic \
  --output /path/to/new-render-directory --manipulation-clips
```

The exporter physically replays every control and disturbance. It supports both successful and failed tasks, checks that the original task criteria and outcome are preserved, and rejects state mismatches. The complete timeline is sampled once per robot second and encoded at 12 fps for 12× playback. The optional grasp and placement excerpts use five frames per robot second at 1×. This first export supplies the physically verified external view and saved observations. To reconstruct semantic heatmaps and compose all panels, use the same recording:

```bash
python -m dream_sim.render_head_view \
  --run results/residential50 \
  --name 01_ProcTHOR-Test-722_seed42_dynamic \
  --render /path/to/new-render-directory --output /path/to/new-camera-directory
CUDA_VISIBLE_DEVICES=0 python -m dream_sim.semantic_history \
  --run results/residential50 \
  --name 01_ProcTHOR-Test-722_seed42_dynamic --output /path/to/new-semantic-directory
python -m dream_sim.observation_panels \
  --record results/residential50/01_ProcTHOR-Test-722_seed42_dynamic \
  --render /path/to/new-render-directory --semantic /path/to/new-semantic-directory \
  --head /path/to/new-camera-directory \
  --output /path/to/new-composite-directory
```

For semantic reconstruction, replace `CUDA_VISIBLE_DEVICES=0` with your allocated physical GPU. This offline step uses CUDA and is separate from the batch worker scheduler.

Semantic reconstruction uses the original frozen encoder, mean voxel-feature pooling and depth clearing, and compares every update's voxel counts with the recording. The heatmap displays raw text-feature alignment before candidate rejection. The overlaid target and planned path come from the original controller log. RGB-D spatial history is a display reconstruction, not an exact copy of the controller's collision map. The current first-person pane is newly rendered during exact physical replay and updates with robot motion. The adjacent Saved observation pane retains the most recent actual semantic observation and its capture time. Newly rendered camera images are display-only and never enter the semantic reconstruction. Every memory/path panel uses only original observations and events at or before its displayed simulation time. These steps perform no new navigation or task decisions.

Encoding uses a local temporary directory, H.264, yuv420p, and faststart. Every frame must decode before the output is accepted. The rendering receipt contains input, observation, protocol, and video hashes. The gallery's [video metadata](https://github.com/BJHYZJ/dream-web/blob/master/simulation/videos.json) links all 50 recordings to the 38/50 evaluation. Exporting them does not execute the policy again or change the denominator.

Outputs include encoded MP4s and JSON receipts. Keep the run directory until all external-view, head-camera, and semantic reconstruction steps finish. These commands replay saved controls; they do not rerun policy decisions.

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
| Arm-return safety feedback | `staged_return` additionally reads robot-link self-contact impulses at every simulator substep; no task-object pose enters its return planner |
| Language-model verification | Hosted mLLM verification is disabled in this adapter |
| Evaluation | Environment object poses, room geometry, contacts, and recorded trajectories |

Small-object search also uses overlapping calibrated RGB-D crops. Memory retrieval verifies up to four semantically ranked observations and caches inference for each saved image and query. Cached detections remain subject to current stale-location rejection rules.

The environment initializes the objects and applies a force-driven relocation after verified visual discovery. The policy detects changes through its RGB-D observations. Destination search keeps the existing scene memory and grounds the requested receptacle in a fresh view. During navigation, the arm stays folded and sensing uses the head camera. The controller constructs a depth-derived navigation map. The replay gallery separately shows the current camera, saved semantic observations, reconstructed semantic heatmaps, and logged routes. Robot self-filtering applies to that navigation map using joint poses and robot geometry. The held-object bound is reconstructed from observed dimensions and the recorded grasp frame, then transformed with the measured gripper pose. It includes a 2 cm uncertainty margin. The navigation self-filter accounts for the held geometry with a uniform 2.5 cm margin. These filters do not mask the raw camera images. The semantic-memory integrator receives the RGB-D observations without this navigation self-filter.

## Experiment records

The [current residential task manifest](../configs/residential50-easy-grasp/task_manifest.json) contains the complete 50-house design. Experiment records include every task outcome, controller and asset hashes, and independent replay checks. Compact archives contain control, trajectory, and evaluation records; large raw sensor arrays are retained in the authors' archive.

The [component measurements](../reproducibility/evidence/components/README.md) cover memory pruning, scaling, and exploration. The [architecture guide](architecture.md) maps the implementation to the perception, memory, navigation, and manipulation pipeline.

## Worker resources

Execution reads `.runtime/worker_resource_limits.json`. The generated one-worker profile contains physical GPU indices, maximum worker count, and per-worker CPU affinity. The file is ignored by Git because device and CPU indices belong to the host. Never copy the authors' cgroup paths or CPU indices onto another machine.

On a machine with sufficient GPU/host memory and at least eight allocated CPU cores, an example four-worker allocation is:

```bash
python -m dream_sim.configure --gpus 0 1 --workers-per-gpu 2 --cpu-threads 2 \
  --output .runtime/proposed_worker_resources.json
```

Review that file and install it as `.runtime/worker_resource_limits.json` while no DREAM workers are active. Then use `--gpus 0 1 --workers-per-gpu 2 --raster-threads 2`. A GPU needs enough memory for every concurrent worker; the measured 10.5 GB per task is not a universal upper bound. Up to four workers per GPU and eight total are supported. More workers reduce available RAM and GPU memory per task.

The generator selects CPU cores but does not reserve devices or install kernel memory limits. Its memory-limit fields are `null`. Existing deployments may additionally specify a writable cgroup-v1 `memory_cgroup_parent`; only then does `WorkerLimits` apply and monitor those kernel limits. Runtime receipts report the actual applied limits. Configure such isolation through the machine's administrator or job scheduler.

## Evaluate a common controller

The reported memory comparison uses `recovery_v2`, ten houses, seeds 100/101/102, and both dynamic and static memory: **60 attempts**. These houses need the earlier asset lock in addition to the current residential scene assets. Prepare a separate cache:

```bash
python -m dream_sim.prepare assets \
  --reference-lock configs/locks/instruction_asset_lock03.json \
  --output-parent .runtime/comparison-assets/data/scene_datasets \
  --output-manifest .runtime/comparison-assets.json
python -m dream_sim.study --controller recovery_v2 \
  --gpus 0 --asset-dir .runtime/comparison-assets --model-cache .runtime/models \
  --output results/memory-comparison --execute
```

The default comparison cases, seeds, and variants are fixed by the profile catalog. Do not pass the residential task manifest for this experiment. Review and summarize the new recordings:

```bash
MS_ASSET_DIR="$PWD/.runtime/comparison-assets" \
HF_HUB_CACHE="$PWD/.runtime/models" \
python results/memory-comparison/frozen_workspace/DREAM_code/experiments/audit_instruction_batch.py \
  --batch results/memory-comparison --output results/memory-comparison-audits \
  --workers 1 --all-task-successes --input-bound
python -m dream_sim.study_report --run results/memory-comparison \
  --audits results/memory-comparison-audits \
  --include-contact-rejections --output results/memory-comparison-summary
```

The paired comparison uses its frozen batch reviewer above. `dream_sim.study_review` is specifically for the newer residential controller with measured intermediate arm returns.

The [paired-comparison README](../reproducibility/evidence/recovery-v2-study/README.md) describes how to recompute the published statistics from its compact archives. This does not require downloading original RGB-D images, whereas a new physical replay does. The earlier `study` and `recovery-study` archives retain their own controllers and outcomes.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `invalid choice` for a GPU index | Create the local allocation with `dream_sim.configure` before launching another CLI process; match `--gpus` to `allowed_gpu_ids`. |
| Worker-count mismatch or missing resource file | Run `configure` once and use its `workers_per_gpu`; existing profiles are never overwritten automatically. |
| Missing model files while offline | Complete `prepare models --production` using the same model cache passed to the runner. |
| Vulkan or asset errors | Check the installed Vulkan ICD, prepare the rendering cache, and pass its path to preflight and execution. |
| `Insufficient disk headroom` | The recording filesystem needs at least 128 GiB free before each task. Use disk-backed storage. |
| GPU or host memory exhaustion | Start with one worker per GPU and inspect actual resource use. Worker concurrency is not part of the task definition. |
| Output directory already exists | Choose a new directory; do not overwrite completed failures to improve the aggregate. See [recovery](recovery.md) for interrupted runs. |
| Missing sensor arrays during replay | Use a complete generated recording. Compact evidence archives omit the large arrays. |
| Source checksum mismatch | Use a clean checkout and a new `DREAM_SIM_SOURCE_CACHE`; do not edit generated, checksum-locked sources. |

## Extended-search video



The exporter uses the frozen source and saved control/force trace. It requires a passing task, observation and arm-return review in the run directory. No policy decisions are rerun and the original result is not rescored.

```bash
python -m dream_sim.render_trial \
  --run /path/to/run \
  --name 07_ProcTHOR-Train-1402_seed42_dynamic \
  --output /path/to/new-video-directory
```

The output contains a complete-timeline overview at 24× playback (one frame per two robot seconds), 1× grasp and placement excerpts, frame timestamps, and a checksum-bound rendering receipt. Each control is physically re-executed even when it is not displayed. The exporter rejects control/state mismatches and rechecks both arm returns before writing its completion receipt. Use the simulation environment and asset caches described above; rendering uses the CPU and H.264 encoding. The [extended-search records](../reproducibility/evidence/long-search/README.md) document the different termination and memory settings used for that follow-up. Its one qualified completion among four selected cases does not replace the main 38/50 result.

To add the synchronous head-camera replay, use `render_head_view` from Section 5 with the same extended-search run and render directory. Reconstruct its semantic history with `semantic_history`, then compose the views with `observation_panels --semantic`, as in the current 50-trial gallery. The published long-search recording includes all 2,552 original semantic observations, a synchronized current camera view, and logged navigation paths. The frame timeline and 24×/1× playback rates are inherited from the physical replay export. `--geometry-only` remains available for a geometry map without semantic scores.
