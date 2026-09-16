"""Calibrated image tiles for language detection of small visible objects."""
from dataclasses import replace
import numpy as np


def image_tiles(observation):
    height, width = observation.depth_m.shape
    tile_width, tile_height = min(width, 384), min(height, 320)
    for y in sorted({0, height - tile_height}):
        for x in sorted({0, width - tile_width}):
            intrinsics = observation.intrinsics.copy()
            intrinsics[0, 2] -= x
            intrinsics[1, 2] -= y
            tile = replace(observation,
                           rgb=observation.rgb[y:y + tile_height, x:x + tile_width].copy(),
                           depth_m=observation.depth_m[y:y + tile_height, x:x + tile_width].copy(),
                           intrinsics=intrinsics)
            yield tile, (x, y, x + tile_width, y + tile_height)


def tiled_detections(perception, observation, query):
    detections = []
    for tile, (x, y, _, _) in image_tiles(observation):
        for detection in perception.detect(tile, query):
            detections.append(replace(detection,
                box_xyxy=(np.asarray(detection.box_xyxy) + [x, y, x, y]).tolist()))
    # Overlapping tiles can rediscover the same surface. Keep the strongest view.
    kept = []
    for detection in sorted(detections, key=lambda item: -item.score):
        if not any(np.linalg.norm(np.asarray(detection.point_world) - old.point_world) < .05
                   for old in kept):
            kept.append(detection)
    return kept
