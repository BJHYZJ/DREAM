"""Observation-only bridge to production DREAM perception and feature memory.

No simulator import is allowed in this module. There is deliberately no actor
ID, actor pose, scene geometry, oracle navigation grid, or relocation endpoint
in its input interface. Simulated camera/robot odometry is a declared input.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dream.semantic_retrieval import feature_alignment


def production_module(name, relative_path):
    """Load the actual source without importing physical-robot package init."""
    spec = importlib.util.spec_from_file_location(name, REPO / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VoxelizedPointcloud = production_module(
    "dream_learned_voxel_util", "src/dream/mapping/voxel/voxel_util.py"
).VoxelizedPointcloud
select_frontier_goal = production_module(
    "dream_learned_frontier", "src/dream/mapping/exploration_policy.py"
).select_frontier_goal


@dataclass(frozen=True)
class RGBDObservation:
    frame_id: int
    sim_step: int
    sensor: str
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    camera_to_world_cv: np.ndarray
    base_xyyaw: np.ndarray

    def __post_init__(self):
        if self.rgb.dtype != np.uint8 or self.rgb.shape != (*self.depth_m.shape, 3):
            raise ValueError("Expected aligned uint8 RGB HxWx3 and metric depth HxW")
        if self.intrinsics.shape != (3, 3) or self.camera_to_world_cv.shape != (4, 4):
            raise ValueError("Camera calibration must be 3x3 K and 4x4 CV camera pose")
        if not np.isfinite(self.intrinsics).all() or not np.isfinite(self.camera_to_world_cv).all():
            raise ValueError("Non-finite camera calibration")

    def world_points(self):
        v, u = np.indices(self.depth_m.shape)
        pixels = np.stack((u, v, np.ones_like(u)), axis=-1).astype(np.float32)
        camera = (pixels @ np.linalg.inv(self.intrinsics).T) * self.depth_m[..., None]
        return camera @ self.camera_to_world_cv[:3, :3].T + self.camera_to_world_cv[:3, 3]

    def save(self, path: Path):
        from instruction_observation_efficiency import save_observation
        save_observation(path, vars(self))


@dataclass(frozen=True)
class Detection:
    observation_id: int
    query: str
    score: float
    box_xyxy: list[float]
    point_world: list[float]
    grounding_query: str | None = None


class LearnedPerception:
    def __init__(self, device="cuda", threshold=0.15, model_set="production"):
        import os

        cache=Path(os.environ["HF_HUB_CACHE"])
        lock=cache / "dream_models.lock.json"
        if not lock.is_file():
            raise FileNotFoundError("Run prepare_learned_models.py before offline inference")
        os.environ["DREAM_MODEL_LOCK_FILE"]=str(lock)
        from dream.perception.encoders.siglip_encoder import MaskSiglipEncoder
        from dream.perception.detection.owl.owl_perception import OwlPerception

        if model_set not in {"base","production"}:
            raise ValueError(model_set)
        self.model_set=model_set
        self.encoder = MaskSiglipEncoder(device=device, version="so400m" if model_set=="production" else "base")
        self.detector = OwlPerception(
            version="owlv2-L-p14-ensemble" if model_set=="production" else "owlv2-B-p16-ensemble",
            device=device, confidence_threshold=threshold,
        )
        self.encoder.model.eval()
        self.detector.model.eval()
        from instruction_owl_attention import configure_owl_vision_attention
        self.owl_attention_runtime = configure_owl_vision_attention(self.detector.model)
        from instruction_owl_precision import configure_owl_vision_precision
        self.owl_precision_runtime = configure_owl_vision_precision(self.detector.model)
        self.owl_attention_runtime["precision"] = "managed_by_recorded_owl_vision_precision"
        from instruction_perception_cache import CachedOwlQueries
        self.cached_detector = CachedOwlQueries(self.detector)
        self.threshold = threshold
        self.text_cache = {}

    def text(self, query):
        if query not in self.text_cache:
            self.text_cache[query] = self.encoder.encode_text(query).detach().cpu()
        return self.text_cache[query]

    def dense(self, obs):
        # Keep the production dense-feature calculation bounded while retaining
        # full-resolution RGB/depth for OWL and geometric localization.
        h, w = obs.depth_m.shape
        scale = min(1., 320/max(h,w))
        image_shape = (round(h*scale), round(w*scale))
        with torch.inference_mode():
            _, features = self.encoder.run_mask_siglip(
                torch.from_numpy(obs.rgb.copy()).permute(2, 0, 1),
                image_shape=image_shape if scale < 1 else None,
            )
        return features[0].detach().cpu()

    def dense_sampled(self, obs, stride):
        from instruction_feature_sampling import sampled_mask_siglip
        with torch.inference_mode():
            return sampled_mask_siglip(self.encoder,obs.rgb,stride)

    def detect(self, obs, query, max_depth=4.0, *, threshold=None):
        """Original OWL inference; median valid bbox depth for simulator RGB-D.

        The original compute_obj_coord takes a median over the whole bbox,
        including invalid zero depth. Here invalid depths are excluded, but no
        instance mask, color classifier or ground-truth object size is used.
        The output is a visible-surface estimate, not a known object centre.
        """
        score_threshold=self.threshold if threshold is None else float(threshold)
        if not np.isfinite(score_threshold) or not 0<=score_threshold<=1:
            raise ValueError("Detection threshold must be finite and within [0, 1]")
        result = self.cached_detector.predict(obs.rgb, query, score_threshold)
        scores = result["scores"].detach().cpu().numpy()
        boxes = result["boxes"].detach().cpu().numpy()
        world = obs.world_points()
        h, w = obs.depth_m.shape
        detections = []
        for index in np.argsort(-scores):
            box = boxes[index]
            x0, y0 = np.maximum(np.floor(box[:2]).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(box[2:]).astype(int), (w, h))
            if x1 <= x0 or y1 <= y0:
                continue
            depth = obs.depth_m[y0:y1, x0:x1]
            valid = np.isfinite(depth) & (depth >= 0.10) & (depth <= max_depth)
            if valid.sum() < 6:
                continue
            xyz = np.median(world[y0:y1, x0:x1][valid], axis=0)
            from instruction_detection_extent import refine_detected_extent
            refined = refine_detected_extent(world, obs.depth_m, box)
            if refined is not None:
                box = np.asarray(refined['box_xyxy'])
                xyz = np.asarray(refined['point_world'])
            detections.append(Detection(obs.frame_id, query, float(scores[index]), box.tolist(), xyz.tolist()))
        # One narrower wording may recover a plate at the ordinary threshold.
        # Keep the instruction noun for memory and record the actual model query.
        # Existing detections always take precedence; weak tracking never expands.
        if not detections and query == "plate" and score_threshold >= self.threshold:
            for candidate in self.detect(obs, "round plate", max_depth,
                                         threshold=score_threshold):
                detections.append(Detection(candidate.observation_id, query,
                    candidate.score, candidate.box_xyxy, candidate.point_world,
                    grounding_query=candidate.query))
        return detections


class SemanticMemory:
    def __init__(self, perception, *, dynamic=True, voxel_size=0.05, stride=6, max_depth=4.0,
                 observation_directory=None):
        self.perception = perception
        self.dynamic = dynamic
        self.stride = stride
        self.max_depth = max_depth
        self.cloud = VoxelizedPointcloud(voxel_size=voxel_size, feature_pool_method="mean")
        self.observations: dict[int, RGBDObservation] = {}
        if observation_directory is not None:
            from instruction_observation_store import ObservationStore
            self.observations = ObservationStore(observation_directory, RGBDObservation)
        self.retrieval_verification_cache = {}
        self.rejected_regions = []
        self.rejected_observations=set()
        self.query_rejections={}
        self.rejection_version=0
        self.retrieval_exclusions={}
        self.updates = []

    def __len__(self):
        return 0 if self.cloud.points is None else len(self.cloud.points)

    def integrate(self, obs, features=None):
        started = time.monotonic()
        if obs.frame_id in self.observations:
            raise ValueError("Observation IDs must be unique")
        before = len(self)
        if self.dynamic:
            self.cloud.clear_points(
                torch.from_numpy(obs.depth_m.copy()), torch.from_numpy(obs.intrinsics.copy()),
                torch.from_numpy(obs.camera_to_world_cv.copy()), min_samples_clear=0,
                depth_in_view_max_distance=self.max_depth,
            )
        after_clear = len(self)
        sampled_features = None
        if features is None:
            sampler = getattr(self.perception,"dense_sampled",None)
            if sampler is None:
                features = self.perception.dense(obs)
            else:
                sampled_features = sampler(obs,self.stride)
        stride = self.stride
        points = obs.world_points()[::stride, ::stride]
        depth = obs.depth_m[::stride, ::stride]
        valid = np.isfinite(depth) & (depth >= 0.10) & (depth <= self.max_depth)
        valid &= np.isfinite(points).all(axis=-1)
        if valid.any():
            # The feature grid is aligned to the same camera image, not an
            # independently projected or simulator-provided semantic image.
            if sampled_features is None:
                h, w = obs.depth_m.shape
                fh, fw = features.shape[:2]
                ys = np.rint(np.arange(0,h,stride)*(fh-1)/max(1,h-1)).astype(int)
                xs = np.rint(np.arange(0,w,stride)*(fw-1)/max(1,w-1)).astype(int)
                sampled_features = features[ys[:,None],xs[None,:]]
            if sampled_features.shape[:2] != points.shape[:2]:
                raise ValueError("Sampled features must align with observed depth pixels")
            self.cloud.add(
                torch.as_tensor(points[valid], dtype=torch.float32), obs_id=obs.frame_id,
                features=sampled_features[valid].float(),
                rgb=torch.as_tensor(obs.rgb[::stride, ::stride][valid], dtype=torch.float32),
            )
        self.observations[obs.frame_id] = obs
        update = dict(frame_id=obs.frame_id, sim_step=obs.sim_step, sensor=obs.sensor,
                      before=before, removed=before-after_clear, after=len(self),
                      input_points=int(valid.sum()), elapsed_s=time.monotonic()-started)
        self.updates.append(update)
        return update

    def alignments(self, query):
        if not len(self):
            return torch.empty(0)
        scores = feature_alignment(self.perception.text(query), self.cloud.features).reshape(-1)
        regions,observations=self.rejections_for(query)
        for obs_id in observations:
            scores[self.cloud._obs_counts==obs_id]=-float("inf")
        # This adapter mirrors the production timestamp-limited rejected region
        # rule: a later observation of the same area is not banned forever.
        for center, radius, max_id in regions:
            rejected = torch.linalg.norm(self.cloud.points[:, :2] - torch.tensor(center), dim=1) <= radius
            rejected &= self.cloud._obs_counts <= max_id
            scores[rejected] = -float("inf")
        return scores

    def ranked_observations(self, query, limit=8):
        scores = self.alignments(query)
        if not len(scores):
            return []
        # Temporary navigation priorities do not delete semantic evidence or
        # alter the displayed alignment field. Other saved candidates remain
        # available while an unproductive approach is deferred.
        for center,radius in getattr(self,"retrieval_exclusions",{}).get(query,[]):
            near=torch.linalg.norm(self.cloud.points[:,:2]-torch.tensor(center),dim=1)<=radius
            scores[near]=-float("inf")
        selected = []
        for i in torch.argsort(scores, descending=True).tolist():
            if not torch.isfinite(scores[i]):
                break
            obs_id = int(self.cloud._obs_counts[i])
            if any(row[0] == obs_id for row in selected):
                continue
            selected.append((obs_id, float(scores[i])))
            if len(selected) >= limit:
                break
        return selected

    def retrieve(self, query):
        """Verify up to four semantically ranked saved RGB-D observations.

        Each immutable image/query pair is inferred once. Rejection rules are
        applied on every call so a cached visual result cannot revive a stale
        or explicitly rejected location.
        """
        candidates = self.ranked_observations(query, limit=4)
        first_candidate = None
        for obs_id, score in candidates:
            candidate = dict(observation_id=obs_id, alignment=score)
            if first_candidate is None:
                first_candidate = candidate
            key = (obs_id, query)
            if key not in self.retrieval_verification_cache:
                observation = self.observations[obs_id]
                detections = self.perception.detect(observation, query, self.max_depth)
                if not detections:
                    from instruction_visual_search import tiled_detections
                    detections = tiled_detections(self.perception, observation, query)
                self.retrieval_verification_cache[key] = detections
            detections = [d for d in self.retrieval_verification_cache[key]
                          if not self.detection_rejected(d)
                          and not any(np.linalg.norm(np.asarray(d.point_world[:2])-center)<=radius
                                      for center,radius in getattr(self,"retrieval_exclusions",{}).get(query,[]))]
            if detections:
                return detections[0], candidate
        return None, first_candidate

    def detection_rejected(self,detection):
        regions,observations=self.rejections_for(detection.query)
        if detection.observation_id in observations:
            return True
        return any(detection.observation_id<=max_id and
            np.linalg.norm(np.asarray(detection.point_world[:2])-center)<=radius
            for center,radius,max_id in regions)

    def rejections_for(self,query):
        regions,observations=self.query_rejections.get(query,([],set()))
        return self.rejected_regions+regions,self.rejected_observations|observations

    def reject(self, xy, max_observation_id, radius=0.30, observation_id=None,query=None):
        if self.dynamic:
            if query is None:
                regions,observations=self.rejected_regions,self.rejected_observations
            else:
                regions,observations=self.query_rejections.setdefault(query,([],set()))
            if radius is not None:
                regions.append((np.asarray(xy, dtype=np.float32).tolist(), radius, max_observation_id))
            self.rejection_version+=1
            # RobotAgent rejects the candidate observation as well as its
            # timestamp-limited spatial region. Otherwise a background voxel
            # in the SAME old image can resurrect the rejected object via OWL.
            if observation_id is not None:
                observations.add(int(observation_id))


def select_live_or_memory_target(live, retrieved):
    """Fresh verified vision may supersede memory in BOTH ablation variants.

    A memory ablation must not ignore an observed moved target just to create a
    wrong revisit. This follows RobotAgent's focused target refresh ordering.
    """
    return live if live is not None else retrieved


class ObservedOccupancy:
    """Fixed empty metric canvas; every occupied/free cell comes from RGB-D.

    Floor height and robot radius are embodiment calibration, not scene maps.
    Unknown space is never supplied as traversable to the route planner.
    """
    def __init__(self, origin=(-16., -16.), size=320, resolution=0.10, radius=0.30):
        self.origin = np.asarray(origin, dtype=float)
        self.resolution = resolution
        self.radius = radius
        self.known = np.zeros((size, size), dtype=np.int8)
        self.last_seen = np.zeros((size, size), dtype=np.int32)

    def cells(self, xy):
        return np.floor((np.asarray(xy)-self.origin)/self.resolution).astype(int)[..., ::-1]

    def world(self, rc):
        return self.origin + (np.asarray(rc)[..., ::-1]+0.5)*self.resolution

    def inside(self, rc):
        return (rc >= 0).all(axis=-1) & (rc < np.asarray(self.known.shape)).all(axis=-1)

    def integrate(self, obs, stride=8, self_spheres=()):
        xyz = obs.world_points()[::stride, ::stride].reshape(-1, 3)
        depth = obs.depth_m[::stride, ::stride].reshape(-1)
        valid = np.isfinite(xyz).all(axis=1) & (depth >= .1) & (depth <= 4.0)
        xyz = xyz[valid]
        camera = obs.camera_to_world_cv[:3, 3]
        free = np.zeros_like(self.known, dtype=np.uint8)
        occupied = np.zeros_like(free)
        # Project only ray intervals inside the calibrated collision-height
        # slab; rays above the robot do not certify free ground underneath.
        for endpoint in xyz:
            length = np.linalg.norm(endpoint[:2]-camera[:2])
            t = np.linspace(0, 1, max(2, int(length/self.resolution)+1))
            ray = camera + t[:, None]*(endpoint-camera)
            ray = ray[(ray[:, 2] >= .08) & (ray[:, 2] <= 1.60)]
            rc = self.cells(ray[:, :2])
            rc = rc[self.inside(rc)]
            if len(rc):
                free[rc[:, 0], rc[:, 1]] = 1
        endpoints = self.cells(xyz[:, :2])
        valid = self.inside(endpoints) & (xyz[:, 2] >= .12) & (xyz[:, 2] <= 1.60)
        # Do not map Fetch's own body as scene clutter. This uses robot pose,
        # not rendered instance segmentation or scene-specific coordinates.
        valid &= np.linalg.norm(xyz[:, :2]-obs.base_xyyaw[:2], axis=1) > .36
        # Proprioceptive self filtering, not renderer segmentation. In particular
        # an actually held payload must not be rasterized as a wall travelling
        # in front of the robot. The spheres come from measured robot links.
        for center, radius in self_spheres:
            valid &= np.linalg.norm(xyz-np.asarray(center),axis=1)>radius
        endpoints = endpoints[valid]
        if len(endpoints):
            occupied[endpoints[:, 0], endpoints[:, 1]] = 1
        # The physical robot's current footprint is proprioceptive evidence.
        base = self.cells(obs.base_xyyaw[:2])
        cv2.circle(free, tuple(base[::-1]), max(1, int(self.radius/self.resolution)), 1, -1)
        self.known[free.astype(bool)] = 1
        self.known[occupied.astype(bool)] = -1
        self.last_seen[(free | occupied).astype(bool)] = obs.frame_id

    def traversable(self):
        blocked = self.known == -1
        clearance = distance_transform_edt(~blocked)*self.resolution
        return (self.known == 1) & (clearance >= self.radius)

    def frontier(self, base_xy, semantic_scores=None):
        from houseexpo_cross_room import weighted_distances

        safe = self.traversable()
        start = tuple(self.cells(base_xy))
        if not self.inside(np.array(start)) or not safe[start]:
            return None, None
        distances, parents = weighted_distances(safe, start)
        frontier = safe & binary_dilation(self.known == 0, iterations=2) & np.isfinite(distances)
        frontier &= distances >= 4
        age = self.last_seen.max() - self.last_seen.astype(float)
        age /= max(1., age.max())
        decision = select_frontier_goal(age, semantic_scores, frontier, np.array(start), semantic_rate=.1)
        return decision.index, parents

    def semantic_field(self, memory, query):
        """Observed language-feature alignment, with no target-centred prior."""
        from scipy.ndimage import maximum_filter

        field = np.zeros_like(self.known, dtype=float)
        if not len(memory):
            return field
        cells = self.cells(memory.cloud.points.numpy()[:,:2])
        scores = memory.alignments(query).numpy()
        valid = self.inside(cells) & np.isfinite(scores)
        cells, scores = cells[valid], np.maximum(scores[valid],0.)
        np.maximum.at(field,(cells[:,0],cells[:,1]),scores)
        # Local context propagation uses only observed features. It does not
        # populate an oracle disk around any initial or current target pose.
        return maximum_filter(field,size=11,mode="constant")
