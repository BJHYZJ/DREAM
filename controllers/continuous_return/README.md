# Fast arm return with measured intermediate states

This controller inherits `staged_return`. `controller.json` identifies every changed module and its checksum.

## Motion

The arm lifts clear, withdraws inward, and follows the raise, shorten, turn, lower, approach, align, and stow states. Each intermediate state must settle for 0.3 seconds; final stow requires one second. The base stays stationary, and self-contact monitoring runs at every physics step.

Loaded motion is limited to 0.85 rad/s and 1.5 rad/s²; empty motion to 1.1 rad/s and 2.0 rad/s². A continuous curve through the IK points avoids stopping at every small waypoint. Timing respects local speed and acceleration bounds. If cubic interpolation crosses the reserved joint-limit margin, shape-preserving interpolation retains the IK coordinates' bounds. Every resulting reference is checked against joint limits, the observed support height, and the original 1.5 cm head-clearance requirement before execution.

Initialization preserves the original staged controller's 40-control sequence before camera scanning. An unreachable open-hand grasp uses the same lift and clearance checks before recovery. If no feasible path exists, recovery stops and the task remains unsuccessful.

## Settling measurements

Native joint velocities remain recorded and are checked first. PhysX TGS can report nonzero joint velocities at a stationary pose because of the solver's force/drive update timing; see the [PhysX documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html#tgs-steady-state-velocity-and-position-discrepancy).

Only after the final native-velocity check has waited its full five seconds may a displacement-based check confirm stow. It requires a complete second of positions sampled at every native physics step: actual speed below 0.08 rad/s, target error below 0.025 rad, and position range below 0.002 rad. Native velocities remain in the trace. Independent physical replay must match the native position samples. Missing samples, movement between control endpoints, or a replay mismatch reject the return. Physics settings, contact checks, and payload-retention checks remain unchanged. The experiment protocol records the task budget separately.

This is an explicit measurement-method change. Earlier cohorts retain their original scores.

## Placement recovery

Placement-center correction uses observed receptacle clearance, with a maximum command bias of 3 cm moving at no more than 3 mm/s. The alignment target adapts to the available clearance and stays within the original 2.5 cm tolerance. After rejecting a geometrically unusable receptacle before placement motion, the controller confirms the retained object's compact posture and resumes search. It tries at most three observed candidates within the declared task budget.

## Evaluation

The complete 50-house cohort passes **38/50 (76%)** with an **1800-second robot-action budget and no fixed server deadline**. All 38 counted successes pass independent physics, observation, contact and both arm-return checks. All 12 unsuccessful tasks remain in the denominator. The houses were used during controller development; this is not a held-out generalization estimate.

[Results and original records](../../reproducibility/evidence/residential-fast-return/README.md) include every task outcome and the independent reviews. Across the 32 jointly successful tasks with the staged controller, median complete return time changes from 44.55 to 20.975 seconds after grasp and from 52.40 to 24.375 seconds after placement. These durations include lift, intermediate states and final settling. Controller and budget changes prevent attributing the full-task success difference to return speed alone.

Use `scripts/run_residential.sh` with a fresh output directory; it defaults to this controller, 1800 action seconds and no server deadline. Direct Python calls require the explicit settings in the [run guide](../../docs/reproduction.md). Historical protocols retain their original outcomes. The [extended searches](../../reproducibility/evidence/long-search/README.md) are separate from the main cohort.
