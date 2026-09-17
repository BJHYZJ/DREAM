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

The default `continuous_return` controller completes **38/50 easy-grasp tasks (76%)** with independent physics, observation, and both arm-return checks. All 50 scenes use fixed tasks and maps, seed 42, dynamic memory, and native object scale. The cohort contains 21 models across 6 categories; see the [task definition](../configs/residential50-easy-grasp/README.md). These are controller-development houses, so the rate describes this cohort.

```bash
DREAM_PYTHON="$(command -v python)" \
DREAM_ASSET_DIR="$PWD/.runtime/render_assets" \
DREAM_MODEL_CACHE="$PWD/.runtime/models" \
./scripts/run_residential.sh \
  --gpus 0 1 --workers-per-gpu 4 --raster-threads 4 \
  --output results/residential50
```

This prints the plan for `continuous_return`; add `--execute` to run it. The script defaults to **1800 seconds of robot actions**, **no fixed server deadline**, and **1200 seconds of worker-slot waiting**. Each arm-return state and the final fold must pass the configured settling checks. The example selects GPUs 0 and 1 with four workers per GPU. The deployment resource profile must assign the corresponding worker slots, CPU cores, and memory limits. Queue time is separate from execution time. Use a new output directory for every run. The three cache/runtime environment variables above select your installed Python and prepared caches; omitting them uses `python3.11` and the repository-local `.runtime/render_assets` and `.runtime/models` caches.

The historical `compact_v1` controller passed 36/50 (72%) and the slower corrected `staged_return` passed 32/50 (64%). To select either historical protocol through the script, set `DREAM_CONTROLLER` to its name and pass `--robot-time-limit-seconds 900 --wall-timeout-seconds 2700`. Explicit command-line values override the script defaults.

The current controller includes bounded release-alignment recovery. To reproduce its 30-minute action protocol with no fixed server deadline:

```bash
DREAM_CONTROLLER=continuous_return ./scripts/run_residential.sh \
  --gpus 0 1 --workers-per-gpu 4 --raster-threads 4 \
  --robot-time-limit-seconds 1800 --wall-timeout-seconds 0 \
  --wait-for-slot-seconds 1200 --output results/residential50-1800
```

Add `--execute` to run it, and use the Python and cache environment variables above for your installation. The limit is frozen in `protocol.json` before workers start and must match every recorded result. The runner accepts up to 1800 action seconds. `--wall-timeout-seconds 0` disables the separate server deadline; completed tasks finish immediately and robot-action timeouts still count as failures. A positive server limit can be set explicitly, up to 5400 seconds. `--wait-for-slot-seconds` controls only the worker queue wait (script default 1200, maximum 3600); it does not extend either execution clock. Direct calls to `python -m dream_sim.study` retain the CLI defaults of `staged_return`, 900 action seconds, a 2700-second server watchdog, and 120 seconds of queue waiting; pass the full options above to select the current residential protocol without the wrapper. The completed 1800-second evaluation and its public review both report 38/50 (76%); its records and failure analysis are in `results/residential-adjusted/`. The controller, action budget, and server deadline changed together relative to the preceding cohort. Changed time budgets define separate experiments; old task records cannot be rescored against a longer limit.

The [current all-50 report](../reproducibility/evidence/residential-fast-return/results.json) retains all 38 qualified successes and 12 failures. Its [evidence directory](../reproducibility/evidence/residential-fast-return/) includes the original task and review records, failure analysis, paired return times, and file/member checksums. Follow that directory's integrity-check instructions to verify the portable records. The current gallery includes all 50 of these attempts; failures are exported without requiring a successful-task review.

The [historical compact-controller directory](../reproducibility/evidence/residential-evaluation/README.md) retains its 36/50 result and independent review records. Run `python -m dream_sim.evaluate` to check that archive and reconstruct its evaluated source. This command is specific to the historical compact controller. Large raw sensor arrays from both experiments remain in local experiment storage; the portable integrity check does not perform another physics replay.

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

Use a complete recorded run directory, including its frozen workspace, task records, and independent reviews:

```bash
python -m dream_sim.render_cohort_trial \
  --run results/residential-adjusted \
  --name 01_ProcTHOR-Test-722_seed42_dynamic \
  --output /path/to/new-render-directory --manipulation-clips
```

The exporter physically replays every control and disturbance. It supports both successful and failed tasks, checks that the original task criteria and outcome are preserved, and rejects state mismatches. The complete timeline is sampled once per robot second and encoded at 12 fps for 12× playback. The optional grasp and placement excerpts use five frames per robot second at 1×. This first export supplies the physically verified external view and saved observations. To reconstruct semantic heatmaps and compose all panels, use the same recording:


```bash
python -m dream_sim.render_head_view \
  --run results/residential-adjusted \
  --name 01_ProcTHOR-Test-722_seed42_dynamic \
  --render /path/to/new-render-directory --output /path/to/new-camera-directory
python -m dream_sim.semantic_history \
  --run results/residential-adjusted \
  --name 01_ProcTHOR-Test-722_seed42_dynamic --output /path/to/new-semantic-directory
python -m dream_sim.observation_panels \
  --record results/residential-adjusted/01_ProcTHOR-Test-722_seed42_dynamic \
  --render /path/to/new-render-directory --semantic /path/to/new-semantic-directory \
  --head /path/to/new-camera-directory \
  --output /path/to/new-composite-directory
```

Semantic reconstruction uses the original frozen encoder, mean voxel-feature pooling and depth clearing, and compares every update's voxel counts with the recording. The heatmap displays raw text-feature alignment before candidate rejection. The overlaid target and planned path come from the original controller log. RGB-D spatial history is a display reconstruction, not an exact copy of the controller's collision map. The current first-person pane is newly rendered during exact physical replay and updates with robot motion. The adjacent Saved observation pane retains the most recent actual semantic observation and its capture time. Newly rendered camera images are display-only and never enter the semantic reconstruction. Every memory/path panel uses only original observations and events at or before its displayed simulation time. These steps perform no new navigation or task decisions.

Encoding uses a local temporary directory, H.264, yuv420p, and faststart. Every frame must decode before the output is accepted. The rendering receipt contains input, observation, protocol, and video hashes. The gallery's [video metadata](https://github.com/BJHYZJ/dream-web/blob/master/simulation/videos.json) links all 50 recordings to the 38/50 evaluation. Exporting them does not execute the policy again or change the denominator.

`dream_sim.video` remains available for converting older archived composite recordings. Those profiles and their video format are distinct from the current gallery.

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

Parallel execution reads `.runtime/worker_resource_limits.json`, a machine-specific deployment profile. It must agree with `--gpus`, `--workers-per-gpu`, and the available CPU and memory allocation. This local file is excluded from Git because its CPU IDs, cgroup paths, and GPU assignment belong to the host machine. The batch runner and `worker_limits.py` validate that configuration before task execution. Keep your deployment's existing allocation when using this repository on a shared server.

To inspect the task list without launching workers, omit `--execute`. The standard `study` mode uses one worker per requested GPU; `--parallel` selects the residential batch with explicit worker slots and a recorded action limit (900 seconds by default).

## Render an independently reviewed trial

The exporter uses the frozen source and saved control/force trace. It requires a passing task, observation and arm-return review in the run directory. No policy decisions are rerun and the original result is not rescored.

```bash
python -m dream_sim.render_trial \
  --run /path/to/run \
  --name 07_ProcTHOR-Train-1402_seed42_dynamic \
  --output /path/to/new-video-directory
```

The output contains a complete-timeline overview at 24× playback (one frame per two robot seconds), 1× grasp and placement excerpts, frame timestamps, and a checksum-bound rendering receipt. Each control is physically re-executed even when it is not displayed. The exporter rejects control/state mismatches and rechecks both arm returns before writing its completion receipt. Use the simulation environment and asset caches described above; rendering uses the CPU and H.264 encoding. The [extended-search records](../reproducibility/evidence/long-search/README.md) document the different termination and memory settings used for that follow-up. Its one qualified completion among four selected cases does not replace the main 38/50 result.

To add the synchronous head-camera replay, use `render_head_view` from Section 5 with the same extended-search run and render directory. Compose it with `observation_panels --geometry-only` in place of `--semantic`; this produces the published long-search layout with the saved observation and an RGB-D geometry/navigation map. The frame timeline and 24×/1× playback rates are inherited from that export. Optional semantic reconstruction uses `semantic_history` and `--semantic` as in the current 50-trial gallery.
