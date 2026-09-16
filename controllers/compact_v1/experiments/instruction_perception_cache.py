"""Reuse OWLv2 image features across independent noun queries on one image.

Each query keeps its original independent sigmoid scores and postprocessing.
Image content, including crop dimensions, keys a bounded GPU feature cache.
"""
from collections import OrderedDict
import hashlib
from types import SimpleNamespace

import numpy as np
import torch


class CachedOwlQueries:
    def __init__(self, detector, max_bytes=128 * 1024**2):
        self.detector = detector
        self.max_bytes = max_bytes
        self.resident_bytes = 0
        self.images = OrderedDict()
        self.texts = OrderedDict()
        self.image_encodes = 0
        self.image_reuses = 0

    @torch.inference_mode()
    def predict(self, rgb, query, threshold):
        detector = self.detector
        model = detector.model
        image = np.ascontiguousarray(rgb)
        key = (image.shape, hashlib.sha256(image).digest())
        if query not in self.texts:
            inputs = detector.processor(text=[["a photo of a " + query]], return_tensors="pt")
            inputs = {name: value.to(detector.device) for name, value in inputs.items()}
            features = model.owlv2.get_text_features(**inputs)
            features = features / torch.linalg.norm(features, dim=-1, keepdim=True)
            self.texts[query] = (features[None], inputs["input_ids"][None, ..., 0] > 0)
            if len(self.texts) > 32:
                self.texts.popitem(last=False)
        self.texts.move_to_end(query)
        if key not in self.images:
            inputs = detector.processor(images=image, return_tensors="pt")
            pixels = inputs["pixel_values"].to(detector.device)
            feature_map, _ = model.image_embedder(pixels)
            features = feature_map.flatten(1, 2)
            boxes = model.box_predictor(features, feature_map)
            size = sum(t.numel() * t.element_size() for t in (features, boxes))
            while self.images and self.resident_bytes + size > self.max_bytes:
                _, (_, _, old_size) = self.images.popitem(last=False)
                self.resident_bytes -= old_size
            if size <= self.max_bytes:
                self.images[key] = (features, boxes, size)
                self.resident_bytes += size
            self.image_encodes += 1
        else:
            features, boxes, _ = self.images[key]
            self.images.move_to_end(key)
            self.image_reuses += 1
        query_features, query_mask = self.texts[query]
        logits, _ = model.class_predictor(features, query_features, query_mask)
        outputs = SimpleNamespace(logits=logits, pred_boxes=boxes)
        sizes = torch.tensor([image.shape[:2]], device=detector.device)
        return detector.processor.image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=sizes)[0]

    def stats(self):
        return dict(image_encodes=self.image_encodes, image_reuses=self.image_reuses,
                    resident_bytes=self.resident_bytes, budget_bytes=self.max_bytes)
