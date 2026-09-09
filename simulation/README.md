# DREAM cross-room dynamic pick-and-place in ManiSkill

This directory provides the task configurations, exact source versions and
execution/audit tools corresponding to ten selected demonstrations in ten
distinct indoor houses. It is separate from DREAM's original ROS hardware
workflow. The real-robot source and instructions remain unchanged.

The ten cases are development-selected demonstrations, not a held-out benchmark
or a claim of a 100% success rate. Train/Val/Test in a house name comes from the
upstream scene split; it does not switch the controller into training mode.
Keep every result produced by a new run, including failures.

## 1. Install the isolated environment

From the repository root, on Linux:

```bash
python3.11 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install -r simulation/requirements.txt
python -m pip check
```

The recorded configuration uses Python 3.11.15, ManiSkill 3.0.1, SAPIEN 3.0.3,
PyTorch 2.10.0, CPU PhysX, Mesa Vulkan rendering and CUDA learned inference.
Exact Python package versions are in the requirements files. A working CUDA
installation and Vulkan ICD are required; these commands do not install or
replace system drivers. The server used NVIDIA H20 GPUs. About 10.5 GB of GPU
memory was observed for a single task on that server, not established as a
portable minimum. Start with one worker.

The original server used this installed software-renderer ICD:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
```

Use the actual compatible ICD on your machine. Preflight checks the file exists;
only actual task execution checks that rendering works. Other hardware or
library versions can change numerical behavior and are not guaranteed to
reproduce identical outcomes. Do not modify a frozen controller to make a
configuration check pass.

## 2. Download the locked models and scene assets

The following commands download from the original upstream sources. No model
weights or third-party household meshes are redistributed in this repository.
Observe each upstream license in addition to DREAM's MIT license.

```bash
python simulation/_vendor/DREAM_code/experiments/prepare_learned_models.py \
  --cache-dir simulation/.runtime/models --production
python simulation/_vendor/DREAM_code/experiments/prepare_instruction_assets.py \
  --reference-lock simulation/_vendor/DREAM_code/experiments/repro_profiles/locks/instruction_asset_lock03.json \
  --output-parent simulation/.runtime/assets/data/scene_datasets \
  --output-manifest simulation/.runtime/assets_download.json
python -m simulation.run --preflight
```

The complete asset lock contains 1,339 files at upstream revision
`1a173d5de042aaad8f1af09d4d2bc2ce4004b28a`. Both production models (large SigLIP
and OWL-V2) and the two small compatibility models have immutable revisions in
`repro_profiles/locks/`. The `--production` flag is necessary. A model-lock JSON
alone is not a downloaded model; the actual weights/configuration must exist.
Preflight compares every scene asset's bytes/hash, model revisions and installed
Python package versions. It checks weight presence, not a separate weight-file
checksum manifest. Execution uses the prepared model cache offline.

Network access is needed only for installation/download. If an asset download is
partial, use the downloader's `--resume-manifest` with the partial manifest,
instead of `--reference-lock`, and choose a new output manifest. Conflicting
existing assets are never overwritten. For existing caches elsewhere, pass
`--asset-dir /absolute/path/to/assets --model-cache /absolute/path/to/models`
to preflight and each run. Do not rely on a symlink to the authors' workspace.

## 3. Execute a case or all ten

```bash
python -m simulation.run --case 01 --output simulation/results/my_case01
python -m simulation.run --all --output simulation/results/my_ten_cases
```

These are new learned-policy executions, not action replay. The default is one
GPU worker. On a machine with sufficient resources, `--gpus 0 1` uses one worker
per listed device. Repeating a device number allocates additional workers, so
only do this deliberately. The ten-case acceptance run on the original server
uses `--gpus 0 1 0 1 0 1`; it is not a required minimum.

The runner validates and freezes the chosen source/task before launch. Each
profile has its recorded scene, seed, instruction, threshold and navigation
configuration. It creates exactly one attempt per requested case. It never
changes a seed, retries a failed task, overwrites an existing output, or replaces
a scorer after observing a new failure. Use a new output directory for each
intentional rerun. A 1,200-second simulation budget and a 14,400-second policy
wall timeout are retained. Wall time is much longer than video playback because
of perception, rendering and saved evidence.

| ID | Native house | Seed | Pickup → destination |
|---|---|---:|---|
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

The full instruction is authoritative in the profile catalog. The unified entry
point is new packaging around the original source versions, not a claim that
all historical videos used one latest controller. Cases 01–08 retain their
original heading-navigation setting; cases 09–10 retain the later heading
adapter. Six deduplicated controller versions cover the ten demonstrations;
the seventh snapshot belongs to the separate frozen comparison.

To check code/profile integrity without loading runtime dependencies or running
a task:

```bash
python -m simulation.run --preflight --dry-run
python -m simulation.run --all --dry-run --output simulation/results/dry_plan
python -m pytest simulation/tests -q
```

These checks do not establish task success.

## 4. Read the result and audit

Each output contains `planned.json`, `preflight.json`, live `progress.json` and
terminal `acceptance.json`. Each `case_XX/` contains:

- `execution/`: immutable source/configuration, one saved episode and its original
  score; RGB-D observations, detection identities, events, controls, external
  forces, simulator trajectories and original 1× rendered video are retained.
- `audit/physical/`: a new physics instance executes those saved controls and
  forces, comparing states and measuring contact. It never assigns recorded
  robot/target states to actors. This audit is not another learned-policy trial.
- `audit/record/`: checks instruction order, correctly observed target identity,
  disturbance timing, measured empty old location, subsequent rediscovery,
  physical grasp/transport, placement and continuous recording provenance.
- `acceptance.json`: per-case original verdict, independent checks and hashes.

`all_passed` requires every requested fresh execution and its declared audits to
pass. A simulator process exiting normally is not sufficient. Historical
compatibility fields are preserved; do not reinterpret an absent newer endpoint
as a new success or failure.

Case 05 has a pre-declared evaluator-v4 correction: qualified held transport does
not disappear when a grasped object is lowered, and room assignment uses
wall-derived regions rather than only eroded room cores. Its original false
verdict remains, alongside reassessment and independent physical evidence.
Case 07 uses the already documented v4 physical-audit accounting without changing
its original true verdict. Record-review scripts are independently selected by
the SHA256 recorded with each original video: v1 for 01, v2 for 02–04 and 06–08,
v3 for 05, and v4 for 09–10. The extra historical v3 audit snapshot is kept in
`_audit_sources/`, separately from the seven policy/comparison snapshots.
Neither accounting nor review-version selection modifies actions or counts as
a new policy trial.

Cases 07 and 10 use display-only spectator re-renders to make placement visible.
After passing policy/record checks, the unified runner automatically creates
`case_XX/spectator/` for these two cases. It executes the same saved controls,
not a second learned policy. Original head/map panels and simulation timestamps
are preserved. To repeat only this display step, use a new output directory:

```bash
python -m simulation.render --execution simulation/results/my_ten_cases \
  --case 07 --output simulation/results/my_case07_spectator
```

To independently audit already complete episodes without another policy trial:

```bash
python -m simulation.audit --execution simulation/results/my_ten_cases \
  --output simulation/results/my_matched_audits --workers 1
```

For display rendering from those separate audits, add
`--audits simulation/results/my_matched_audits` to the render command.

## 5. Export a compact 4× video

After the reviewed composite and matching `video_frames.json` exist:

```bash
python -m simulation.video --input PATH/TO/reviewer_view.mp4 \
  --frames PATH/TO/video_frames.json --output PATH/TO/reviewer_view_4x.mp4
```

This retains 1440×600 resolution and every frame: 5 fps at original simulation
speed becomes 20 fps at 4× playback. There is no frame interpolation or removal
of slow/failed stages. The header says `4x playback`; its clock remains original
simulation time. H.264/yuv420p and faststart support ordinary browser playback.
Only the speed header is redrawn. Published case 01 uses `--crf 18`; the other
nine use `--crf 20`, as recorded in their encoding manifests. Encoding happens on local temporary storage
before a checked copy to the destination, to avoid MP4 relocation problems on
some shared filesystems. JSON records hashes, frame counts and sampled quality.

## Implementation boundary

The policy receives a bounded-English pickup/place instruction, current head
RGB-D observations and simulator odometry. It builds an observed map, rotates
to scan, selects exploration frontiers, navigates using DREAM's A* logic,
checks stale target locations with new depth, and searches again. The destination
is a second language-grounding problem with retained memory. Hidden object
coordinates and evaluator room geometry belong to simulator initialization and
evaluation, not the search policy. An environment-owned force-driven target
relocation is armed only after a verified visible discovery during approach.

Grasp/place use current RGB-D detections, support/footprint geometry and measured
TCP feedback with fixed pregrasp/approach/close/lift/release templates. They are
automated geometric heuristics, not per-house prerecorded full trajectories,
not AnyGrasp and not a learned grasp model. Navigation keeps the arm folded and
uses the head camera; there is no navigation wrist-camera sweep. Simulator
odometry replaces physical SLAM, hosted mLLM verification is disabled, and this
workflow does not independently validate pose-graph RMP or arbitrary language.

The code layout keeps the public wrapper small. `_vendor/DREAM_code/` contains
the release's original helpers and adapted DREAM components; its
`experiments/repro_profiles/sources/` preserves exact historical bytes and
hashes. Do not autoformat these snapshots. New work should use a new profile,
not silently edit one associated with a published recording.

## Evidence and interpretation

The selected video gallery is qualitative evidence of executed cross-room
dynamic tasks. A separate, completed 60-attempt comparison uses one frozen
controller and retains every outcome; its numerical analysis and scope are
documented with the evidence, not inferred from the ten selected videos.
It does not establish general superiority or outdoor performance.

The final delivery check newly executed all ten profiles once in a separate
pinned Python environment on the same server. All ten action and evaluator
trajectory files match their original recordings byte-for-byte. Matching-version
physical/record audits passed for all ten; cases 07 and 10 also passed the exact
original spectator-renderer checks. The original wrapper's reviewer-version
errors for 02 and 06 and the pre-declared case-05 scoring correction are retained
and explained in [`evidence/reproduction/`](evidence/reproduction/README.md).
Audits/re-renders did not rerun policies, change seeds or add comparison trials.

Delivery acceptance records must be read separately from the historical video
checks. A successful execution in the documented same-server environment does
not guarantee identical behavior on all hardware. The author must review the
final videos and publication links before pushing the repositories or submitting
the response. This directory does not itself publish anything.
