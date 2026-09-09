# Architecture

## Task pipeline

The simulation runner connects instruction parsing, RGB-D perception, semantic memory, navigation, and manipulation:

```text
Instruction → scene exploration → target localization → grasp
                    ↑                    ↓
              memory update ← fresh observation

Grasp → destination search with retained memory → placement → evaluation
```

The environment initializes the house and task objects and moves the target after visual discovery. The policy uses its observations to detect this change and update its search. Evaluator trajectories record object poses and contacts for scoring and replay checks.

## Entry points

| Module | Responsibility |
| --- | --- |
| `dream_sim.run` | Validate the runtime, execute cases, and collect evaluation results |
| `dream_sim.prepare` | Download the required models and scene assets |
| `dream_sim.sources` | Verify and extract controller sources; locate a case's implementation |
| `dream_sim.profile` | Launch versioned profiles and the 60-attempt comparison |
| `dream_sim.audit` | Re-evaluate recorded episodes through physics replay and record checks |
| `dream_sim.render` | Render an episode from a spectator camera using its saved controls |
| `dream_sim.video` | Export the composite video at 4× playback |
| `dream_sim.verify_evidence` | Check experiment archive hashes and video-to-case mappings |

## Controller modules

Run `python -m dream_sim.sources --case 01` to print the source directory for a case. Paths below are relative to that directory.

| Source file | Responsibility |
| --- | --- |
| `experiments/run_instruction_task.py` | Create the environment, run the task stages, and save observations and results |
| `experiments/instruction_task.py` | Parse a pickup/place instruction and define the visual-discovery gate |
| `experiments/instruction_policy.py` | Coordinate search, target verification, grasping, and destination search |
| `experiments/dream_learned_core.py` | SigLIP/OWL-V2 perception, semantic memory, and observed occupancy |
| `experiments/maniskill_crossroom_policy.py` | Exploration, route following, and approach behavior |
| `experiments/dream_fetch_navigation.py` | Fetch occupancy map and A* navigation |
| `experiments/instruction_geometry.py` | Grasp geometry and receptacle placement regions |
| `experiments/maniskill_learned_probe.py` | Robot state, sensor observations, and simulator control interface |
| `src/dream/dynamic_memory.py` | Shared memory-update and focused-observation logic |
| `src/dream/semantic_retrieval.py` | Text-to-voxel feature alignment |

The [simulation interfaces](reproduction.md#implementation-boundary) describe the policy's inputs and the components adapted from the real robot.

## Source versions

The ten gallery cases use six controller versions; the comparison study uses a seventh. Each case records its controller, task, seed, evaluation version, and video identity. The runtime selects these through the profile catalog.

The source code is stored in `reproducibility/source_archives/selected_profiles_v1.zip`. Its lock file contains the archive hash and the size and SHA256 of every member. Model weights and house meshes are downloaded separately.

On first use, `dream_sim.sources` verifies the ZIP and extracts it to:

```text
.runtime/sources/<archive-sha256>/
```

Extraction uses a process lock and a temporary directory before publishing the completed tree. Later processes verify the cached files before loading them. Modified files cause a checksum error. Set `DREAM_SIM_SOURCE_CACHE` to use another cache location.

The extracted `DREAM_code` directory is the controller's workspace root. The helper scripts use this name to resolve their relative imports and resources. The project therefore uses an editable installation from a complete checkout, including `configs/` and `reproducibility/`.

## Configuration and outputs

`configs/cases.json` lists the case profiles. `configs/tasks/` and `configs/locks/` contain the corresponding task inputs and dependency locks. The runner checks these files against the source catalog before execution. Record paths inside the catalog resolve relative to that catalog in the extracted source tree.

Each run creates its own output directory with the launch plan, environment checks, saved episode, and evaluation results. The [run guide](reproduction.md#4-read-results) describes the output files.

For algorithm development, copy a controller into a development workspace and record the modified source and task configuration in a new profile. Existing gallery profiles are checksum-locked to their recorded implementations. Changes to the generated source cache will be rejected by the runner.

## Related repositories

- [`realtime`](https://github.com/BJHYZJ/DREAM/tree/realtime): ROS implementation, hardware drivers, robot model, and calibration guides.
- [`dream-web`](https://github.com/BJHYZJ/dream-web): project website, figures, and demonstration videos.
