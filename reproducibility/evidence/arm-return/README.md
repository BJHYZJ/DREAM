# Arm-return checks

These are four physical regression checks of [`staged_return`](../../../controllers/staged_return/README.md), recorded on 16 September 2026. They use development scenes and test only arm retraction. They do not measure a full-task success rate.

| Initial condition | House | Arm controls | Duration | Result |
| --- | --- | ---: | ---: | --- |
| Holding an apple | ProcTHOR-Train-8921 | 917 | 45.85 s | Passed |
| Holding a mug | ProcTHOR-Test-722 | 936 | 46.80 s | Passed |
| Holding a soap bottle | ProcTHOR-Train-4300 | 955 | 47.75 s | Passed |
| Empty gripper after placement | ProcTHOR-Test-722 | 1022 | 51.10 s | Passed |

Duration includes the initial lift and a two-second hold after completing the return. All measured intermediate postures passed a separate check against the saved joint positions and velocities. Each final posture remained within the configured settling tolerances. The three loaded checks retained the object throughout. No robot self-contact or counted environment contact was recorded during any return.

## What was executed

Each scene was initialized from its original task configuration. Original controls and disturbance forces were replayed up to the recorded grasp or release. Measured gripper and payload positions matched the original prefix exactly in these runs (maximum recorded difference: 0 m). The corrected controller then executed new actions from that physical state. No joint or object pose was reassigned after replay began.

The arm controller received robot feedback and the previously observed grasp geometry. Ground-truth payload poses and grasp-contact checks were used only by the evaluator. Contact sampling ran at every native physics substep. The inherited audit excludes robot–target contacts and low horizontal floor support; its collision counts cover robot self-contact and robot/payload contacts with the environment at or above 0.5 N. In the empty-gripper check, the released object's continuing support contact was allowed.

[`checks.json`](checks.json) identifies the exact source hashes, outcomes, independent stage checks, and the two normal-speed videos in the website repository. [`records.zip`](records.zip) contains the new actions, measured trajectories, contact reports, fold receipts, and the original prefix controls and configuration needed to inspect the branch point. The videos show synchronized overhead and oblique views of the apple and mug checks.

The historical 36/50 (72%) task result remains associated with `compact_v1`. It does not apply to this changed controller.
