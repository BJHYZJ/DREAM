"""The robot action budget advances only when the simulator executes a control."""
import math


class RobotActionBudget:
    def __init__(self, seconds, control_frequency_hz):
        if not (math.isfinite(seconds) and seconds>0
                and math.isfinite(control_frequency_hz) and control_frequency_hz>0):
            raise ValueError('Action duration and control frequency must be positive and finite')
        self.seconds=float(seconds)
        self.control_frequency_hz=float(control_frequency_hz)
        self.maximum_steps=math.floor(self.seconds*self.control_frequency_hz)
        if self.maximum_steps<1:raise ValueError('Action budget must allow at least one control')
        self.exceeded=False

    def before_step(self, executed_steps):
        if executed_steps>=self.maximum_steps:
            self.exceeded=True
            raise TimeoutError('Robot action-time budget exceeded')

    def report(self, executed_steps):
        return dict(robot_action_seconds=executed_steps/self.control_frequency_hz,
            robot_action_limit_seconds=self.seconds,robot_action_steps=executed_steps,
            control_frequency_hz=self.control_frequency_hz,robot_action_timeout=self.exceeded,
            time_limit_boundary='Executed control steps / actual control frequency; loading and inference waits do not advance this clock')
