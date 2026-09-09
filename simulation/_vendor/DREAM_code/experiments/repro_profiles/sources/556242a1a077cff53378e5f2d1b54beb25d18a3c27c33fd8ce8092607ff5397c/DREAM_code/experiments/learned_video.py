"""Uncluttered visualization of actually rendered frames, streamed at 1x.

No source labels to crop out, no held/duplicated explanatory frames, no generated
scene pixels. Raw camera videos and frame-to-simulation-step provenance accompany
the composite. Map colors encode the agent's observations, not scene truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def semantic_colors(occupancy,memory,query):
    """Max measured text-feature alignment per cell, fixed scale across runs."""
    field=np.full(occupancy.known.shape,-np.inf,dtype=np.float32)
    if memory is not None and len(memory) and query:
        scores=memory.alignments(query).numpy()
        cells=occupancy.cells(memory.cloud.points.numpy()[:,:2])
        valid=occupancy.inside(cells)&np.isfinite(scores)
        np.maximum.at(field,(cells[valid,0],cells[valid,1]),scores[valid])
    return field


def map_image(occupancy, memory=None, query=None, base_xy=None, trajectory=(), cached=None, semantic=None,
              planned_path=(),waypoint=None,navigation_goal=None):
    known = occupancy.known
    view = np.full((*known.shape, 3), (236,239,241), dtype=np.uint8)
    view[known == 1] = (213,222,224)
    view[known == -1] = (67,81,86)
    if semantic is None and memory is not None:
        semantic=semantic_colors(occupancy,memory,query)
    if semantic is not None:
        valid=np.isfinite(semantic)&(semantic>=.17)
        weights=np.clip((semantic[valid]-.17)/.09,0,1)[:,None]
        hot=np.array([242,130,40])
        view[valid]=(view[valid]*(1-weights*.85)+hot*weights*.85).astype(np.uint8)
    if len(trajectory) > 1:
        cells = occupancy.cells(np.asarray(trajectory))[:, ::-1]
        cv2.polylines(view, [cells.astype(np.int32)], False, (45,100,165), 1)
    if len(planned_path)>1:
        cells=occupancy.cells(np.asarray(planned_path))[:,::-1]
        cv2.polylines(view,[cells.astype(np.int32)],False,(25,155,115),2)
    if navigation_goal is not None:
        cell=tuple(occupancy.cells(navigation_goal)[::-1])
        cv2.drawMarker(view,cell,(25,155,115),cv2.MARKER_DIAMOND,8,1)
    if waypoint is not None:
        cv2.circle(view,tuple(occupancy.cells(waypoint)[::-1]),2,(25,155,115),-1)
    if cached is not None:
        cv2.circle(view, tuple(occupancy.cells(cached[:2])[::-1]), 3, (230,130,40), 1)
    if base_xy is not None:
        cv2.circle(view, tuple(occupancy.cells(base_xy)[::-1]), 3, (35,90,175), -1)
    observed = np.argwhere(known != 0)
    if len(observed):
        low = np.maximum(observed.min(axis=0)-5, 0)
        high = np.minimum(observed.max(axis=0)+6, known.shape)
        view = view[low[0]:high[0], low[1]:high[1]]
    # Rows increase with world +y in this observation grid.
    return view[::-1].copy()


def letterbox(rgb, width, height, color=(245,247,248)):
    scale = min(width/rgb.shape[1], height/rgb.shape[0])
    resized = cv2.resize(rgb, (max(1,round(rgb.shape[1]*scale)), max(1,round(rgb.shape[0]*scale))),
                         interpolation=cv2.INTER_NEAREST)
    result = np.full((height,width,3), color, np.uint8)
    top, left = (height-resized.shape[0])//2, (width-resized.shape[1])//2
    result[top:top+resized.shape[0],left:left+resized.shape[1]] = resized
    return result


class EvidenceVideo:
    def __init__(self, output: Path, hz=20, stride=4):
        self.output = output
        self.stride = stride
        self.hz = hz
        self.rows = []
        self.last_step = None
        self.semantic_stamp=None
        self.semantic=None
        self.writers = {name: imageio.get_writer(output / f"{name}.mp4", fps=hz/stride,
            codec="libx264", quality=8, macro_block_size=2, ffmpeg_log_level="error")
            for name in ("overview_raw", "observation_raw", "reviewer_view")}

    def capture(self, io, occupancy, phase, memory=None, query=None, cached=None,
                planned_path=(),waypoint=None,navigation_goal=None,disturbance_label=None):
        if io.step_id % self.stride or io.step_id == self.last_step:
            return
        if self.last_step is not None and io.step_id-self.last_step != self.stride:
            raise RuntimeError("Video frame gap would invalidate continuous 1x playback")
        self.last_step = io.step_id
        observation = io.capture(getattr(io,"review_sensor",io.display_sensor),keyframe=False)
        overview = io.overview()
        trail = [r["base_xyyaw"][:2] for r in io.trace[::10]]
        # Recompute language coloring at keyframe cadence, not every video
        # frame: inference latency and simulation time are separate quantities.
        if memory is not None:
            stamp=(query,len(memory.updates),len(memory.rejected_regions),getattr(memory,"rejection_version",0))
            if stamp!=self.semantic_stamp:
                self.semantic=semantic_colors(occupancy,memory,query)
                self.semantic_stamp=stamp
        mapview = map_image(occupancy, base_xy=io.pose()[:2], trajectory=trail,
                            cached=cached,semantic=self.semantic,planned_path=planned_path,
                            waypoint=waypoint,navigation_goal=navigation_goal)
        composite = np.full((600,1440,3), (245,247,248), np.uint8)
        composite[36:576,:960] = overview
        composite[60:330,960:] = letterbox(observation.rgb,480,270)
        composite[360:600,960:] = letterbox(mapview,480,240)
        color = (32,46,53)
        for label, xy, scale in ((phase, (20,26), .65),
                (f"{io.step_id/self.hz:.1f} s | sim 1x",(750,26),.5),
                ("Wrist observation" if observation.sensor=="fetch_hand" else "Head observation", (980,54),.5),
                (f"Memory: {query}" if memory is not None else "Observed depth map",(980,354),.5)):
            cv2.putText(composite,label,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
        labels=([disturbance_label] if isinstance(disturbance_label,str) else list(disturbance_label or []))
        for i,label in enumerate(labels):
            top=46+34*i
            cv2.rectangle(composite,(12,top),(650,top+30),(245,235,204),-1)
            cv2.putText(composite,label,(22,top+21),cv2.FONT_HERSHEY_SIMPLEX,.55,color,1,cv2.LINE_AA)
        if hasattr(occupancy,"occupied_xyz"):
            cv2.putText(composite,"Blue: traveled  Green: A* plan / goal",(980,594),
                        cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
        self.writers["overview_raw"].append_data(overview)
        self.writers["observation_raw"].append_data(observation.rgb)
        self.writers["reviewer_view"].append_data(composite)
        self.rows.append(dict(frame=len(self.rows),sim_step=io.step_id,sim_time_s=io.step_id/self.hz,
                             phase=phase,sensor=observation.sensor,
                             disturbance_caption=" | ".join(labels) if labels else None,display_captions=labels,
                             waypoint_xy=waypoint,navigation_goal_xy=navigation_goal))
        if len(self.rows) == 1 or len(self.rows) % 100 == 0:
            cv2.imwrite(str(self.output / "latest_frame.png"),cv2.cvtColor(composite,cv2.COLOR_RGB2BGR))

    def close(self):
        for writer in self.writers.values():
            writer.close()
        (self.output / "video_frames.json").write_text(json.dumps(self.rows,indent=2)+"\n")
