#!/usr/bin/env python3
"""Measure the robot collision envelope after physically executed arm postures."""
import argparse
import json
from pathlib import Path

import numpy as np
from maniskill_learned_probe import make_env, SimulatorIO, array


def measure(io):
    points=[]
    details=[]
    for link in io.robot.robot.get_links():
        for shape in link._objs[0].get_collision_shapes():
            if not hasattr(shape,"vertices"):
                details.append(dict(link=link.name,shape=type(shape).__name__,unsupported=True))
                continue
            pose=link._objs[0].pose*shape.local_pose
            vertices=np.asarray(shape.vertices)
            if hasattr(shape,"scale"):
                vertices=vertices*np.asarray(shape.scale)
            matrix=pose.to_transformation_matrix()
            world=vertices@matrix[:3,:3].T+matrix[:3,3]
            xy=world[:,:2]-io.pose()[:2]
            points.append(xy)
            details.append(dict(link=link.name,max_radius=float(np.linalg.norm(xy,axis=1).max()),
                                z_bounds=[float(world[:,2].min()),float(world[:,2].max())]))
    all_xy=np.vstack(points)
    return dict(max_radius=float(np.linalg.norm(all_xy,axis=1).max()),
                xy_min=all_xy.min(0).tolist(),xy_max=all_xy.max(0).tolist(),links=details)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    env=make_env("ArchitecTHOR-Test-03")
    try:
        env.reset(seed=17)
        io=SimulatorIO(env)
        io.prepare_scan()
        results={"v2_scan_posture":measure(io)}
        for name,joints in (("compact_a",[-1.2,1.1,0,1.8,0,1.2,0]),
                            ("compact_b",[-1.2,1.3,0,1.8,0,1.2,0])):
            initial=io.arm.copy()
            for alpha in np.linspace(0,1,150):
                io.arm=initial*(1-alpha)+np.asarray(joints)*alpha
                io.command()
            results[name]=dict(command=joints,**measure(io))
        (args.output/"envelopes.json").write_text(json.dumps(results,indent=2)+"\n")
        print(json.dumps(results,indent=2))
    finally:
        env.close()


if __name__=="__main__":
    main()
