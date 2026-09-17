"""Read-only arm motion sampling at the simulator's native physics frequency.

TGS drive velocities can remain nonzero at a stationary pose. Keep those native
velocities in the control trace and separately measure displacement at *every*
physics step, so movement between control endpoints cannot be missed.
"""

import math

import numpy as np

ARM_INDICES = (5, 7, 8, 9, 10, 11, 12)
CONTINUOUS = (2, 4, 6)
MEASUREMENT = "joint_positions_at_physics_steps"


def difference(actual, reference):
    delta = np.asarray(actual, dtype=float) - reference
    delta[..., list(CONTINUOUS)] = np.arctan2(
        np.sin(delta[..., list(CONTINUOUS)]), np.cos(delta[..., list(CONTINUOUS)])
    )
    return delta


class JointMotionSampler:
    """Temporarily chain a read-only callback after each physics step."""

    def __init__(self, io, *, control_step=None, selected_steps=None):
        self.io = io
        self.control_step = control_step or (lambda: io.step_id + 1)
        self.selected_steps = selected_steps
        self.dt = float(io.base.scene.px.timestep)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Invalid physics timestep")
        self.records = {}

    def positions(self):
        value = self.io.robot.robot.get_qpos()
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)[0, list(ARM_INDICES)].astype(float, copy=True)

    def after_step(self):
        self.original()
        current = self.positions()
        step = self.control_step()
        if self.selected_steps is None or step in self.selected_steps:
            record = self.records.setdefault(step, dict(
                physics_dt_s=self.dt,
                initial_joint_positions_rad=self.previous.tolist(),
                joint_positions_rad=[],
            ))
            record["joint_positions_rad"].append(current.tolist())
        self.previous = current

    def __enter__(self):
        self.original = self.io.base._after_simulation_step
        self.previous = self.positions()
        self.io.base._after_simulation_step = self.after_step
        return self

    def __exit__(self, *_):
        self.io.base._after_simulation_step = self.original

    def settling_window(self, end_step, target, hz):
        """Require a full second below the original pose/range/speed limits."""
        count = max(2, math.ceil(hz))
        substeps = round(1 / (hz * self.dt))
        if substeps < 1 or not math.isclose(substeps * self.dt * hz, 1., abs_tol=1e-5):
            return None
        points, previous = [], None
        maximum_speed = 0.
        for step in range(end_step - count + 1, end_step + 1):
            record = self.records.get(step, {})
            initial = np.asarray(record.get("initial_joint_positions_rad", []))
            samples = np.asarray(record.get("joint_positions_rad", []))
            if (initial.shape != (7,) or samples.shape != (substeps, 7)
                    or not np.isfinite(initial).all() or not np.isfinite(samples).all()
                    or record.get("physics_dt_s") != self.dt):
                return None
            if previous is not None and np.max(np.abs(initial - previous)) > 1e-7:
                return None
            all_points = np.vstack([initial, samples])
            speed = difference(all_points[1:], all_points[:-1]) / self.dt
            maximum_speed = max(maximum_speed, float(np.max(np.abs(speed))))
            if maximum_speed >= .08 or np.max(np.abs(difference(all_points, target))) >= .025:
                return None
            points.extend(all_points if previous is None else samples)
            previous = samples[-1]
        spread = float(np.max(np.ptp(difference(points, target), axis=0)))
        if spread >= .002:
            return None
        return dict(settling_measurement=MEASUREMENT,
                    measured_maximum_speed_rad_s=maximum_speed,
                    native_physics_samples=count * substeps,
                    measured_window_s=count * substeps * self.dt,
                    settled_joint_range_rad=spread)
