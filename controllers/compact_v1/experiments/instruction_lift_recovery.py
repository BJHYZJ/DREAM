"""Bounded measured Cartesian lift when one shoulder joint reaches its limit."""
import numpy as np


def recover_vertical_lift(io,target_height,check=None,maximum_steps=240):
    def array(value):
        return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
    start=array(io.robot.tcp_pose.p)[0].copy()
    if not np.isfinite(target_height) or target_height<start[2] or maximum_steps<=0:
        raise ValueError('Recovery requires a finite upward goal and positive action budget')
    goal=start.copy();goal[2]=target_height
    io.robot.set_control_mode('pd_ee_delta_pose');io.robot.controller.reset()
    controls=0
    try:
        for _ in range(maximum_steps):
            if check is not None:check()
            error=goal-array(io.robot.tcp_pose.p)[0]
            if np.linalg.norm(error)<.015:break
            rotation=array(io.robot.torso_lift_link.pose.to_transformation_matrix())[0,:3,:3]
            io.command_ee(np.r_[np.clip(rotation.T@error*3.,-.05,.05),0.,0.,0.])
            controls+=1
    finally:
        io.hold_measured_arm()
    return dict(target_height_m=float(target_height),
        measured_height_m=float(array(io.robot.tcp_pose.p)[0,2]),
        controls=controls,maximum_controls=maximum_steps,requested_inward_motion_m=0.)
