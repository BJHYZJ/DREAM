"""Recheck small egg imagery with fixed calibrated crops and a subtype query."""
from dataclasses import replace
import numpy as np


def egg_detail_detections(perception, observation, query):
    if str(query).strip().lower() != 'egg':
        return []
    height, width = observation.depth_m.shape
    side = min(224, height, width)
    if side < 64:
        return []
    detections = []
    for y in sorted(set(np.linspace(0, height - side, 3).round().astype(int))):
        for x in sorted(set(np.linspace(0, width - side, 4).round().astype(int))):
            intrinsics = observation.intrinsics.copy()
            intrinsics[0, 2] -= x
            intrinsics[1, 2] -= y
            crop = replace(observation,
                rgb=observation.rgb[y:y + side, x:x + side].copy(),
                depth_m=observation.depth_m[y:y + side, x:x + side].copy(),
                intrinsics=intrinsics)
            for detection in perception.detect(crop, 'Easter egg'):
                detections.append(replace(detection, query=query,
                    box_xyxy=(np.asarray(detection.box_xyxy) + [x, y, x, y]).tolist()))
    kept = []
    for detection in sorted(detections, key=lambda item: -item.score):
        if not any(np.linalg.norm(np.asarray(detection.point_world) - old.point_world) < .05
                   for old in kept):
            kept.append(detection)
    return kept
