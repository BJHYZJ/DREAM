# Arm retraction

This controller adds a measured sequence for returning the Fetch arm after pickup and release. It inherits the checksum-locked `compact_v1` implementation and replaces only the three modules listed in `controller.json`.

## Motion sequence

1. Hold the base and lift the gripper clear of the support.
2. Compute an inward path from the robot URDF, measured joint positions, and the observed payload dimensions and grasp frame. The inverse-kinematics solver respects joint limits, preserves the gripper orientation, and keeps conservative arm bounds clear of the head collision meshes. Reject an infeasible path before executing it.
3. Withdraw above the observed support, then raise and shorten the arm before turning toward the body.
4. Lower through the outside approach posture, align the shoulder, and reach the compact transport posture.
5. Confirm the final posture for one second before starting transport or completing the task.

Each joint waypoint must reach an error below 0.025 rad and settle before the next starts. Intermediate settling requires 0.3 s, velocity below 0.12 rad/s, and joint range below 0.005 rad. The final window requires 1 s, velocity below 0.08 rad/s, and joint range below 0.002 rad. For withdrawal and folding, quintic interpolation limits joint reference speed to 0.45 rad/s and acceleration to 0.7 rad/s².

A tracking error above 0.12 rad during withdrawal or folding, a stalled waypoint, lost grip feedback, or base displacement above 0.02 m stops the sequence. Simulated robot self-contact is checked at every physics substep; force at or above 0.5 N also stops execution. The controller holds measured joint positions on exit, and a failed return cannot emit a successful fold receipt or start transport.

The initial shoulder lift retains the parent controller's 0.6 rad/s reference limit and 0.25 rad tracking guard; it is also covered by the substep self-contact monitor.

Self-contact feedback is a simulator interface used by this controller. Task-object poses and object contact handles remain evaluator inputs. Independent checks must still verify payload retention and environment contacts. These runtime guards detect faults; they do not certify collision-free motion in every unseen scene.

## Evaluation

The completed fixed-cohort run passes **32/50 scenes (64%)** after independent physics, observation, and both arm-return checks. There were 33 provisional task completions; one lacked the final empty-arm fold and was rejected. All 50 scenes remain in the denominator. Full local recordings are in `results/residential-return/`, including `failure_analysis.md` and per-scene review reports.

The historical **36/50 (72%)** result belongs to `compact_v1`. It does not measure the corrected return controller.

The [four physical arm checks](../../reproducibility/evidence/arm-return/README.md) cover an apple, a mug, a soap bottle, and an empty gripper after placement.

Use `--controller staged_return` with `python -m dream_sim.study`. Use `--controller compact_v1` to reproduce the historical controller. The historical source and archived evidence remain unchanged.
