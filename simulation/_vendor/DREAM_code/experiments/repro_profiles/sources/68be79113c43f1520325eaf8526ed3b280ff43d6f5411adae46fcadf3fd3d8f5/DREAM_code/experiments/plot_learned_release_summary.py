#!/usr/bin/env python3
"""Paper-sized plot derived only from the saved ten-scene paired replay."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=json.loads(args.result.read_text())
    runs=[r for r in result["runs"] if r["status"]=="replayed"]
    values={v:[next(f["region_voxels"] for f in r["final"] if f["variant"]==v) for r in runs] for v in ("dynamic","static")}
    plt.rcParams.update({"font.size":12,"axes.labelsize":12,"xtick.labelsize":11,"ytick.labelsize":11})
    fig,ax=plt.subplots(figsize=(5.5,2.2),constrained_layout=True)
    positions=np.arange(len(runs))
    ax.bar(positions-.18,values["static"],.36,color="#8797a5",label="No ray clearing")
    ax.bar(positions+.18,values["dynamic"],.36,color="#2d64a5",label="Ray clearing")
    ax.scatter(positions+.18,values["dynamic"],s=9,color="#2d64a5",zorder=4)
    ax.set_xticks(positions,np.arange(1,len(runs)+1))
    ax.set(xlabel="Scene index (per-scene table)",ylabel="Former-region voxels",ylim=(-1,62),yticks=[0,20,40,60])
    ax.legend(frameon=False,fontsize=10,ncol=2,loc="upper center")
    ax.grid(axis="y",alpha=.15)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"),dpi=200)
    plt.close(fig)


if __name__=="__main__":
    main()
